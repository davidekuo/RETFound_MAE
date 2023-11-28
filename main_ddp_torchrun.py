# Adapted from PyTorch DistributedDataParallel tutorial:
# https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu_torchrun.py
# and from the PyTorch implementation of "Masked Autoencoders Are Scalable Vision Learners"
# https://arxiv.org/abs/2111.06377
# https://github.com/facebookresearch/mae/tree/main


import argparse
from contextlib import nullcontext
import os
import traceback

import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
import torchmetrics.classification as metrics
from torchvision import datasets, transforms
# import torchvision.transforms.v2 as transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_

import models_vit
from util.lars import LARS
from util.pos_embed import interpolate_pos_embed
import wandb

CFP_CHKPT_PATH = "/home/dkuo/RetFound/model_checkpoints/RETFound_cfp_weights.pth"
OCT_CHKPT_PATH = "/home/dkuo/RetFound/model_checkpoints/RETFound_oct_weights.pth"

CFP_DATASET_PATH = "/home/dkuo/RetFound/tasks/referable_CFP_OCT/dataset/CFP"
OCT_DATASET_PATH = "/home/dkuo/RetFound/tasks/referable_CFP_OCT/dataset/OCT"

RETFOUND_EXPECTED_IMAGE_SIZE = 224
NUM_CLASSES = 2


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            test_dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            loss_fn: torch.nn.Module,
            save_every: int,
            last_snapshot_path: str,
            best_snapshot_path: str,
            test_snapshot_path: str = None,
            mixed_precision: bool = False,
    ) -> None:
        # model, optimizer, loss
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn

        # mixed precision
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler()  

        # dataloaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        # training snapshots
        self.epochs_run = 0
        self.save_every = save_every
        self.last_snapshot_path = last_snapshot_path
        self.best_snapshot_path = best_snapshot_path
        self.test_snapshot_path = test_snapshot_path
        self.best_val_ap = 0.0
        if os.path.exists(last_snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(last_snapshot_path)

    def _load_snapshot(self, snapshot_path: str) -> None:
        """ Load snapshot of model """
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.best_val_ap = snapshot["BEST_VAL_AP"]
        print(f"Successfully loaded snapshot from Epoch {self.epochs_run}")
    
    def _save_snapshot(self, epoch: int, snapshot_path: str):
        """ Save snapshot of current model """
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),  # actual model is module wrapped by DDP
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
            "BEST_VAL_AP": self.best_val_ap,
        }
        torch.save(snapshot, snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}")
    
    def _train_batch(self, inputs: torch.Tensor, targets: torch.Tensor):
        """ Train on one batch """
        self.model.train()
        self.optimizer.zero_grad()
        
        with torch.autocast(device_type='cuda', dtype=torch.float16) if self.mixed_precision else nullcontext():
            outputs = self.model(inputs)  # (batch_size, num_classes)
            loss = self.loss_fn(outputs, targets)
            preds = torch.argmax(outputs, dim=-1)  # (batch_size)
            try:
                train_acc = torch.sum(preds == targets) / len(targets)
            except Exception as e:
                print(f"Exception: {e}")  #  most likely due to CutMix/MixUp
                train_acc = torch.sum(preds == torch.argmax(outputs, dim=-1)) / len(preds)
            print(f"[GPU{self.gpu_id}] Training loss: {loss}, Training accuracy: {train_acc}")

            # avg across all GPUs and send to rank 0 process
            torch.distributed.reduce(loss, op=torch.distributed.ReduceOp.AVG, dst=0)
            torch.distributed.reduce(train_acc, op=torch.distributed.ReduceOp.AVG, dst=0)
            if self.gpu_id == 0:
                print(f"[MEAN] Training loss: {loss}, Training accuracy: {train_acc} \n")
                wandb.log({"train_loss": loss, "train_acc": train_acc})
        
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
    
    def _validate(self, test: bool = False):
        """ Validate current model on validation or final test dataset"""
        metrics_dict = {"avg_loss": torch.tensor([0.0]).to(self.gpu_id)}
        torchmetrics_dict = torch.nn.ModuleDict({
            "auroc": metrics.AUROC(task="multiclass", num_classes=NUM_CLASSES, average="macro"),
            "ap": metrics.AveragePrecision(task="multiclass", num_classes=NUM_CLASSES, average="macro"),
            "precision": metrics.Precision(task="multiclass", num_classes=NUM_CLASSES, average="macro"),
            "recall": metrics.Recall(task="multiclass", num_classes=NUM_CLASSES, average="macro"),
            "f1": metrics.F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro"),
            "accuracy": metrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES, average="macro"),
            # set average="macro" per https://github.com/Lightning-AI/torchmetrics/issues/543
        }).to(self.gpu_id)
        
        dataloader = self.test_dataloader if test else self.val_dataloader
        
        self.model.eval()
        with torch.autocast(device_type='cuda', dtype=torch.float16) if self.mixed_precision else nullcontext():
            with torch.no_grad():
                for batch in dataloader:
                    inputs, targets = batch
                    inputs = inputs.to(self.gpu_id)
                    targets = targets.to(self.gpu_id)
                    outputs = self.model(inputs)  # (batch_size, num_classes)
                    probs = F.softmax(outputs, dim=-1)

                    metrics_dict["avg_loss"] += self.loss_fn(outputs, targets)  
                    for metric, fn in torchmetrics_dict.items():
                        fn(probs, targets)
        
        metrics_dict["avg_loss"] /= len(dataloader)
        for metric, fn in torchmetrics_dict.items():
            metrics_dict[metric] = fn.compute()
            fn.reset()

        for metric, value in metrics_dict.items():
            torch.distributed.reduce(value, op=torch.distributed.ReduceOp.AVG, dst=0) # avg across all GPUs and send to rank 0 process
        
        if self.gpu_id == 0:
            for metric, value in metrics_dict.items():
                print(f"- {metric}: {value.item()}")
            if not test:
                metrics_dict = {f"val_{metric}": value.item() for metric, value in metrics_dict.items()}
            else:  # test
                metrics_dict = {f"test_{metric}": value.item() for metric, value in metrics_dict.items()}
            wandb.log(metrics_dict)

        return metrics_dict

    def _train_epoch(self, epoch: int):
        """ Train for one epoch """
        batch_size = len(next(iter(self.train_dataloader))[0])
        print(f"\n[GPU{self.gpu_id}] Epoch {epoch} | Batch size {batch_size} | Steps: {len(self.train_dataloader)}")
        self.train_dataloader.sampler.set_epoch(epoch)
        
        for batch in self.train_dataloader:
            inputs, targets = batch
            inputs = inputs.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._train_batch(inputs, targets)
        
        if self.scheduler:
            self.scheduler.step()  # update learning rate every epoch if using LR scheduler (alternatively, update every step)

    def train(self, max_epochs: int):
        """ Train for max_epochs """
        for epoch in range(self.epochs_run, max_epochs):
            self._train_epoch(epoch)
            self.epochs_run += 1
            if self.gpu_id == 0:
                print(f"=== Epoch {epoch} | Validation Metrics ===")
            metrics_dict = self._validate()
            if self.gpu_id == 0:
                print("==================================== \n")
                if epoch % self.save_every == 0:
                    self._save_snapshot(epoch, self.last_snapshot_path)
                if self.best_val_ap < metrics_dict["val_ap"]:
                    self._save_snapshot(epoch, self.best_snapshot_path)
    
    def test(self):
        """ Test on final test set """
        self._load_snapshot(self.test_snapshot_path)
        metrics_dict = self._validate(test=True)
        if self.gpu_id == 0:
            print(f"Test Set Performance: \n{metrics_dict}")


def setup_ddp():
    """ Set up distributed data parallel (DDP) training """
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def default_data_transform():
    """ Return default data transforms (based on ImageNet dataset) - for val and test datasets """
    return transforms.Compose([
        transforms.Resize(RETFOUND_EXPECTED_IMAGE_SIZE, 
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(RETFOUND_EXPECTED_IMAGE_SIZE),  # or Resize([224, 224])? 
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    
    # for torchvision.transforms.v2
    # return transforms.Compose([
    #     transforms.Resize(RETFOUND_EXPECTED_IMAGE_SIZE, 
    #                       interpolation=transforms.InterpolationMode.BICUBIC),
    #     transforms.CenterCrop(RETFOUND_EXPECTED_IMAGE_SIZE),  # or Resize([224, 224])? 
    #     transforms.ToDtype(torch.float32, scale=True),
    #     transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    # ])


def augment_data_transform(augment: str):
    """
    Parameters
    ----------
    augment : str
        Data augmentation strategy: "trivialaugment", "randaugment", "autoaugment", "augmix", "deit3"

    Returns
    -------
    Data transforms for specified data augmentation strategy
    """
    # Re-implementation / approximation of DeiT3 3-Augment
    deit3_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomChoice([
            transforms.RandomGrayscale(p=0.2), 
            transforms.RandomSolarize(threshold=130, p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        ]),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ])

    augmentations = {
        "trivialaugment": transforms.TrivialAugmentWide(),
        "randaugment": transforms.RandAugment(),
        "autoaugment": transforms.AutoAugment(),
        "augmix": transforms.AugMix(),
        "deit3": deit3_transforms,
    }
    assert augment in augmentations, f"Data augmentation strategy must be 'none' or one of {list(augmentations.keys())}"

    return transforms.Compose([
        augmentations[augment],
        transforms.Resize(RETFOUND_EXPECTED_IMAGE_SIZE, 
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(RETFOUND_EXPECTED_IMAGE_SIZE),  # or Resize([224, 224])? 
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    # for torchvision.transforms.v2
    # return transforms.Compose([
    #     augmentations[augment],
    #     transforms.Resize(RETFOUND_EXPECTED_IMAGE_SIZE, 
    #                       interpolation=transforms.InterpolationMode.BICUBIC),
    #     transforms.CenterCrop(RETFOUND_EXPECTED_IMAGE_SIZE),  # or Resize([224, 224])? 
    #     transforms.ToDtype(torch.float32, scale=True),
    #     transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    # ])


def cutmixup_collate_fn(batch):
    """ Returns DataLoader collate function with CutMix and MixUp at random - for training dataloader only """
    cutmix = transforms.CutMix(num_classes=2)
    mixup = transforms.MixUp(num_classes=2)
    cutmix_or_mixup = transforms.RandomChoice([cutmix, mixup])
    return cutmix_or_mixup(*default_collate(batch))


def setup_dataloader(dataset: Dataset, batch_size: int, collate_fn=None):
    """
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to load
    batch_size : int
        Batch size for dataloader
    collate_fn : callable, optional
        Function to collate batch of data, by default None
        Use cutmixup_collate_fn for CutMix/MixUp data augmentation
    
    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        Dataloader for dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,  # no shuffling as we want to preserve the order of the images
        sampler=DistributedSampler(dataset),  # for DDP
        collate_fn=collate_fn,
    )


def setup_data(dataset_path: str, batch_size: int, dataset_size: int = None, augment: str = "none", cutmixup: bool = False):
    """
    Parameters
    ----------
    dataset_path : str
        Path to root directory of PyTorch ImageFolder dataset
    batch_size : int
        Batch size for dataloaders
    dataset_size : int, optional
        Number of images to sample from the training set. Use all images if not specified.
    augment : str, optional
        Data augmentation strategy: "none", "trivial_augment", "randaugment", "augmix"; by default "none"
    cutmixup : bool, optional
        Whether to use CutMix/MixUp data augmentation, by default False
    
    Returns
    -------
    train_dataloader, val_dataloader, test_dataloader : torch.utils.data.DataLoader
        Dataloaders for training, validation, and test sets
    """
    # Set up data augmentations
    train_transform = default_data_transform() if augment == "none" else augment_data_transform(augment)
    train_collate_fn = cutmixup_collate_fn if cutmixup else None

    # Set up datasets
    train_dataset = datasets.ImageFolder(dataset_path + "/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(dataset_path + "/val", transform=default_data_transform())
    test_dataset = datasets.ImageFolder(dataset_path + "/test", transform=default_data_transform())

    # If specified, take a random subset of the training dataset of size 'dataset_size' 
    if dataset_size:
        train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset))[:dataset_size])
    
    # Set up and return dataloaders
    train_dataloader = setup_dataloader(train_dataset, args.batch_size, train_collate_fn)
    val_dataloader = setup_dataloader(val_dataset, args.batch_size)
    test_dataloader = setup_dataloader(test_dataset, args.batch_size)

    return train_dataloader, val_dataloader, test_dataloader


def setup_model(
        mae_checkpoint_path: str, 
        training_strategy: str,
        arch: str = 'vit_large_patch16', 
        num_classes: int = 2, 
        global_pool: bool = False,
    ):
    """
    Parameters
    ----------
    mae_checkpoint_path : str
        Path to checkpoint of MAE model
    training_strategy : str
        Training strategy: full_finetune, linear_probe
    arch : str, optional
        Architecture of ViT model, by default 'vit_large_patch16'
    num_classes : int, optional
        Number of classes, by default 2
    global_pool : bool, optional
        Whether to use global average pooling, by default False
    
    Returns
    -------
    vit_model : torch.nn.Module
        ViT model
    """
    assert args.training_strategy in ["full_finetune", "linear_probe"], "Training strategy must be full_finetune, or linear_probe"

    # instantiate model
    vit_model = models_vit.__dict__[arch](
        num_classes=num_classes,
        global_pool=global_pool,
    )
    
    # load mae checkpoint
    print(f"Loading model from {mae_checkpoint_path}")
    mae_checkpoint = torch.load(mae_checkpoint_path, map_location='cpu')
    mae_checkpoint_model = mae_checkpoint["model"]
    state_dict = vit_model.state_dict()
    
    # adapt mae model to vit model
    for k in ["head.weight", "head.bias"]:
        if k in mae_checkpoint_model and mae_checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from checkpoint due to shape mismatch")
            del mae_checkpoint_model[k]

    # interpolate position embedding following DeiT
    interpolate_pos_embed(vit_model, mae_checkpoint_model)
    
    msg = vit_model.load_state_dict(mae_checkpoint['model'], strict=False)
    # print(msg)

    if global_pool:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    else:
        assert set(msg.missing_keys) == {"head.weight", "head.bias"}

    # manually initialize head fc layer following MoCo v3
    trunc_normal_(vit_model.head.weight, std=.02)

    if training_strategy == "linear_probe":
        # hack for linear probe: revise model head with batchnorm
        vit_model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(vit_model.head.in_features, affine=False, eps=1e-6), vit_model.head)
    
        # freeze all parameters but the head
        for name, param in vit_model.named_parameters():
            param.requires_grad = False
        for name, param in vit_model.head.named_parameters():
            param.requires_grad = True
    
    return vit_model


def setup_optimizer(
        model: torch.nn.Module, 
        algo: str,
        training_strategy: str, 
        lr: float, 
        weight_decay: float
    ):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    algo : str
        Optimization algorithm to use: adamw, lars
    training_strategy : str
        Training strategy: full_finetune, linear_probe
    lr : float
        Learning rate
    weight_decay : float
        Weight decay
    
    Returns
    -------
    optimizer : torch.optim.Optimizer
        Optimizer for training
    """
    optimization_algorithms = {
        "adamw": torch.optim.AdamW,
        "lars": LARS,
    }
    assert algo in optimization_algorithms, f"Optimizer must be one of {list(optimization_algorithms.keys())}"
    optim_algo = optimization_algorithms[algo]

    if training_strategy == "linear_probe":
        optimizer = optim_algo(params=model.head.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # training_strategy == "full_finetune"
        optimizer = optim_algo(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    
    return optimizer


def setup_lr_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int, algorithm: str = None):
    """
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer for training
    total_epochs : int
        Total number of epochs to train for
    algorithm : str, optional
        Learning rate scheduler algorithm: "cosine_linear_warmup", by default None
    
    Returns
    -------
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler
    """
    scheduler = None
    
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    if algorithm == "cosine_linear_warmup":
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[5])
    
    return scheduler


def setup_loss_fn():
    """
    Returns
    -------
    loss_fn : torch.nn.Module
        Loss function for training
    """
    # CFP training set has 548 normal and 102 referable (0.19)
    # OCT training set has 782 normal and 132 referable (0.17)
    class_weights = torch.tensor([1.0, 5.0]).cuda()  # [normal, referable]
    return torch.nn.CrossEntropyLoss(weight=class_weights)  


def main(args):
    setup_ddp()
    torch.manual_seed(0)

    # Set up correct pretrained model and dataset paths for modality
    assert args.modality in ["CFP", "OCT"], "Modality must be CFP or OCT"
    mae_checkpoint_path = CFP_CHKPT_PATH if args.modality == "CFP" else OCT_CHKPT_PATH
    dataset_path = args.dataset_path if args.dataset_path else (CFP_DATASET_PATH if args.modality == "CFP" else OCT_DATASET_PATH) 

    # Set up model checkpoint paths
    last_snapshot_path = args.last_snapshot_path + "/last.pth"
    best_snapshot_path = args.best_snapshot_path + "/best.pth"

    # Set up training objects
    train_dataloader, val_dataloader, test_dataloader = setup_data(dataset_path, args.batch_size, args.dataset_size, args.augment, args.cutmixup)
    model = setup_model(mae_checkpoint_path, training_strategy=args.training_strategy, global_pool=args.global_average_pooling)
    optimizer = setup_optimizer(model, algo=args.optimizer, training_strategy=args.training_strategy, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = setup_lr_scheduler(optimizer, args.total_epochs)
    loss_fn = setup_loss_fn()

    # Set up W&B logging 
    gpu_id = int(os.environ["LOCAL_RANK"])
    if gpu_id == 0:  # only log on main process
        wandb.init(
            project="retfound_referable_cfp_oct",
            config=args,
            resume=True,  # continue logging from previous run if did not finish
        )

    # Set up trainer
    trainer = Trainer(
        model=model, 
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        mixed_precision=args.mixed_precision,
        save_every=args.save_every,
        last_snapshot_path=last_snapshot_path,
        best_snapshot_path=best_snapshot_path,
        test_snapshot_path=args.final_test_snapshot_path,
    )
    
    # Test if given a final test snapshot
    if args.final_test_snapshot_path:
        trainer.test()
    
    # Otherwise, train model
    else: 
        try:
            trainer.train(args.total_epochs)
        except Exception as e:
            print("An exception occurred during training: ", e)
            traceback.print_exc()
        else:  
            # If training finishes successfully, delete the last.pth model snapshot using gpu 0
            # and rename best.pth to [wandb_run_name].pth
            if gpu_id == 0:
                print(f"\nTraining completed successfully! Deleting last.pth and saving best.pth as {wandb.run.name}.pth\n")
                if os.path.exists(last_snapshot_path): 
                    os.remove(last_snapshot_path)
                if os.path.exists(best_snapshot_path):
                    os.rename(best_snapshot_path, f"{args.best_snapshot_path}/{wandb.run.name}.pth")

    # Clean up
    wandb.finish()
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Task parameters
    parser.add_argument('--modality', type=str, default="OCT", help='Must be CFP or OCT')
    parser.add_argument('--dataset_path', type=str, help='Path to root directory of PyTorch ImageFolder dataset')   
        
    # Data Hyperparameters
    parser.add_argument('--dataset_size', type=int, help='Number of training images to train the model with.')  # CFP: 650, OCT: 914
    parser.add_argument('--augment', type=str, default="trivialaugment", help='Data augmentation strategy: none, trivialaugment, randaugment, autoaugment, augmix, deit3') 
    parser.add_argument('--cutmixup', type=bool, default=False, help='Whether to use CutMix/MixUp data augmentation')
    
    # Training Hyperparameters
    parser.add_argument('--total_epochs', type=int, default=50, help='Total epochs to train the model')  # 10 epochs with full dataset
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size on each device')  # max batch_size with AMP {full_finetune: 64, linear_probe: 1024+}
    
    parser.add_argument('--optimizer', type=str, default="lars", help='Optimizer: adamw, lars')
    parser.add_argument('--lr_scheduler', type=str, help='Learning rate scheduler: none, cosine_linear_warmup')  
    parser.add_argument('--training_strategy', type=str, default="full_finetune", help="Training strategy: full_finetune, linear_probe, lora")
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    
    parser.add_argument('--global_average_pooling', type=bool, default=False, help='Use global average pooling')
    parser.add_argument('--mixed_precision', type=bool, default=True, help='Use automatic mixed precision')
    parser.add_argument('--save_every', type=int, default=1, help='Save a snapshot every _ epochs')

    # Model snapshot paths
    parser.add_argument('--last_snapshot_path', type=str, 
                        default="/home/dkuo/RetFound/tasks/referable_CFP_OCT/snapshots",
                        help='Path to directory for saving last training snapshot (will be saved as last.pth)')
    parser.add_argument('--best_snapshot_path', type=str, 
                        default="/home/dkuo/RetFound/tasks/referable_CFP_OCT/snapshots",
                        help='Path to directory for saving best snapshot so far (will be saved as best.pth)')
    
    # Evaluate on final test set
    parser.add_argument('--final_test_snapshot_path', type=str, help='Evaluate snapshot at provided path on final test set.')

    args = parser.parse_args()
    main(args)


# Run on all of 3 visible GPUs on 1 machine: 
# > CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=gpu main_ddp_torchrun.py [args]
