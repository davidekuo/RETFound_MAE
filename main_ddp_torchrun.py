# Adapted from PyTorch DistributedDataParallel tutorial:
# https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu_torchrun.py
# and from the PyTorch implementation of "Masked Autoencoders Are Scalable Vision Learners"
# https://arxiv.org/abs/2111.06377
# https://github.com/facebookresearch/mae/tree/main


import argparse
from contextlib import nullcontext
import os
import pprint

import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchmetrics.classification as metrics
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_

import models_mae, models_vit
from util.lars import LARS
from util.pos_embed import interpolate_pos_embed
import wandb

CFP_CHKPT_PATH = "/home/dkuo/RetFound/model_checkpoints/RETFound_cfp_weights.pth"
OCT_CHKPT_PATH = "/home/dkuo/RetFound/model_checkpoints/RETFound_oct_weights.pth"

CFP_DATASET_PATH = "/home/dkuo/RetFound/tasks/referable_CFP_OCT/dataset/CFP"
OCT_DATASET_PATH = "/home/dkuo/RetFound/tasks/referable_CFP_OCT/dataset/OCT"

RETFOUND_EXPECTED_IMAGE_SIZE = 224


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            test_dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            loss_fn,
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
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.best_val_ap = snapshot["BEST_VAL_AP"]
        print(f"Successfully loaded snapshot from Epoch {self.epochs_run}")
    
    def _save_snapshot(self, epoch: int, snapshot_path: str):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),  # actual model is module wrapped by DDP
            "EPOCHS_RUN": epoch,
            "BEST_VAL_AP": self.best_val_ap,
        }
        torch.save(snapshot, snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}")
    
    def _train_batch(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.model.train()
        self.optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16) if self.mixed_precision else nullcontext():
            outputs = self.model(inputs)  # (batch_size, num_classes)
            loss = self.loss_fn(outputs, targets)
            torch.distributed.reduce(loss, op=torch.distributed.ReduceOp.AVG, dst=0)  # avg loss across all GPUs and send to rank 0 process
            print(f"Training loss after reduce:[GPU{self.gpu_id}] {loss}")
            if self.gpu_id == 0:
                wandb.log({"train_loss": loss})
        
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
    
    def _validate(self, test: bool = False):
        metrics_dict = {"avg_loss": torch.tensor([0.0]).to(self.gpu_id)}
        torchmetrics_dict = torch.nn.ModuleDict({
            "auroc": metrics.AUROC(task="multiclass", num_classes=2, thresholds=None),
            "ap": metrics.AveragePrecision(task="multiclass", num_classes=2, thresholds=None),
            "precision": metrics.Precision(task="multiclass", num_classes=2, thresholds=None),
            "recall": metrics.Recall(task="multiclass", num_classes=2, thresholds=None),
            "f1": metrics.F1Score(task="multiclass", num_classes=2, thresholds=None),
            "accuracy": metrics.Accuracy(task="multiclass", num_classes=2, thresholds=None),
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
            pprint(f"Validation metrics: [GPU{self.gpu_id}] {metrics_dict}")
            if not test:
                metrics_dict = {f"val_{metric}": value.item() for metric, value in metrics_dict.items()}
            else:  # test
                metrics_dict = {f"test_{metric}": value.item() for metric, value in metrics_dict.items()}
            wandb.log(metrics_dict)

        return metrics_dict

    def _train_epoch(self, epoch: int):
        batch_size = len(next(iter(self.train_dataloader))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batch size {batch_size} | Steps: {len(self.train_dataloader)}")
        self.train_dataloader.sampler.set_epoch(epoch)
        
        for batch in self.train_dataloader:
            inputs, targets = batch
            inputs = inputs.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._train_batch(inputs, targets)

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._train_epoch(epoch)
            self.epochs_run += 1
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Running Validation")
            metrics_dict = self._validate()
            if self.gpu_id == 0:
                if epoch % self.save_every == 0:
                    self._save_snapshot(epoch, self.last_snapshot_path)
                if self.best_val_ap < metrics_dict["val_ap"]:
                    self._save_snapshot(epoch, self.best_snapshot_path)
    
    def test(self):
        self._load_snapshot(self.test_snapshot_path)
        metrics_dict = self._validate(test=True)
        if self.gpu_id == 0:
            print(f"Test Set Performance: \n{metrics_dict}")


def setup_ddp():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def setup_data_transforms():
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    t = []
    t.append(
        transforms.Resize(RETFOUND_EXPECTED_IMAGE_SIZE, 
                          interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(RETFOUND_EXPECTED_IMAGE_SIZE))  # or Resize([224, 224])? 
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))

    return transforms.Compose(t)


def setup_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,  # no shuffling as we want to preserve the order of the images
        sampler=DistributedSampler(dataset),  # for DDP
    )


def setup_data(dataset_path: str, batch_size: int):
    # Set up datasets
    train_dataset = datasets.ImageFolder(dataset_path + "/train", transform=setup_data_transforms())
    val_dataset = datasets.ImageFolder(dataset_path + "/val", transform=setup_data_transforms())
    test_dataset = datasets.ImageFolder(dataset_path + "/test", transform=setup_data_transforms())

    # Set up dataloaders
    train_dataloader = setup_dataloader(train_dataset, args.batch_size)
    val_dataloader = setup_dataloader(val_dataset, args.batch_size)
    test_dataloader = setup_dataloader(test_dataset, args.batch_size)

    return train_dataloader, val_dataloader, test_dataloader


def setup_model(
        mae_checkpoint_path: str, 
        training_strategy: str,
        arch: str ='vit_large_patch16', 
        num_classes: int = 2, 
        global_pool: bool = False,
    ):

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


def setup_optimizer(model, training_strategy: str, lr=0.03, weight_decay=0):
    if training_strategy == "linear_probe":
        optimizer = LARS(model.head.parameters(), lr, weight_decay)
    else:  # training_strategy == "full_finetune"
        optimizer = LARS(model.parameters(), lr, weight_decay)
        # torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    # print(optimizer)
    return optimizer


def setup_loss_fn():
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

    # Set up training objects
    train_dataloader, val_dataloader, test_dataloader = setup_data(dataset_path, args.batch_size)
    model = setup_model(mae_checkpoint_path, training_strategy=args.training_strategy)
    optimizer = setup_optimizer(model, training_strategy=args.training_strategy, lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = setup_loss_fn()

    #  Set up W&B logging 
    gpu_id = int(os.environ["LOCAL_RANK"])
    if gpu_id == 0:  # only log on main process
        wandb.init(
            project="retfound_referable_cfp_oct",
            config=args,
        )

    # Set up trainer
    trainer = Trainer(
        model=model, 
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        mixed_precision=args.mixed_precision,
        save_every=args.save_every,
        last_snapshot_path=args.last_snapshot_path,
        best_snapshot_path=args.best_snapshot_path,
        test_snapshot_path=args.final_test_snapshot_path,
    )
    
    # Test if given a final test snapshot
    if args.final_test_snapshot_path:
        trainer.test()
    # Train
    else: 
        trainer.train(args.total_epochs)

    # Clean up
    wandb.finish()
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Task parameters
    parser.add_argument('--modality', type=str, default="CFP", help='Must be CFP or OCT')
    parser.add_argument('--dataset_path', type=str, help='Path to root directory of PyTorch ImageFolder dataset')   
        
    # Training Hyperparameters
    parser.add_argument('--total_epochs', type=int, default=10, help='Total epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size on each device')  # max batch_size with AMP {full_finetune: 64, linear_probe: 1024+}
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--mixed_precision', type=bool, default=True, help='Use automatic mixed precision')
    parser.add_argument('--training_strategy', type=str, default="full_finetune", help="Training strategy: full_finetune, linear_probe")
    parser.add_argument('--save_every', type=int, default=1, help='Save a snapshot every _ epochs')

    # Paths
    parser.add_argument('--last_snapshot_path', type=str,
                        default="/home/dkuo/RetFound/tasks/referable_CFP_OCT/snapshots/last.pth", 
                        help='Path to last training snapshot')
    parser.add_argument('--best_snapshot_path', type=str, 
                        default="/home/dkuo/RetFound/tasks/referable_CFP_OCT/snapshots/best.pth",
                        help='Path to best snapshot so far')
    parser.add_argument('--final_test_snapshot_path', type=str, help='Evaluate snapshot at provided path on final test set.')

    args = parser.parse_args()
    main(args)

# Run on all of 3 visible GPUs on 1 machine: 
# > CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=gpu main_ddp_torchrun.py [args]