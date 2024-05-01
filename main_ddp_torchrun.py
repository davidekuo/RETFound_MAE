# Adapted from PyTorch DistributedDataParallel tutorial:
# https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu_torchrun.py
# and from the PyTorch implementation of "Masked Autoencoders Are Scalable Vision Learners"
# https://arxiv.org/abs/2111.06377
# https://github.com/facebookresearch/mae/tree/main


import argparse
from contextlib import nullcontext
import numpy as np
import os
import random
from typing import Tuple, Union
import wandb

import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torchmetrics.classification as metrics
from torchvision import datasets, transforms
# import torchvision.transforms.v2 as transforms_v2

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_
from timm.optim import Lamb
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.poly_lr import PolyLRScheduler
from timm.scheduler.step_lr import StepLRScheduler

import models_vit
from util.lars import LARS
from util.pos_embed import interpolate_pos_embed
from util.sam import SAM, disable_running_stats, enable_running_stats, SAMOptimizer, ClosureGradScaler
from util.samplers import DistributedSamplerWrapper, RepeatedAugmentationSampler

CFP_CHKPT_PATH = "/home/dkuo/RetFound/model_checkpoints/RETFound_cfp_weights.pth"
OCT_CHKPT_PATH = "/home/dkuo/RetFound/model_checkpoints/RETFound_oct_weights.pth"

CFP_DATASET_PATH = "/home/dkuo/RetFound/tasks/referable_CFP_OCT/dataset/CFP"
OCT_DATASET_PATH = "/home/dkuo/RetFound/tasks/referable_CFP_OCT/dataset/OCT"
PEDS_OPTOS_DATASET_PATH = "/home/dkuo/RetFound/tasks/peds_optos/dataset" 

RETFOUND_EXPECTED_IMAGE_SIZE = 224
NUM_CLASSES = 2


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            test_dataloader: DataLoader,
            grad_accum_steps: int,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            loss_fn: torch.nn.Module,
            mixed_precision: bool,
            grad_clip_max_norm: float,
            sam: bool,
            save_every: int,
            last_snapshot_path: str,
            best_snapshot_path: str,
            test_snapshot_path: str = None,
            evaluate_on_final_test_set: bool = False,
    ) -> None:
        # model, optimizer, loss
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.model = DDP(
            self.model, 
            device_ids=[self.gpu_id],
            gradient_as_bucket_view=True,
            static_graph=False,  # errors when set to True
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        
        self.grad_accum_steps = grad_accum_steps
        self.grad_clip_max_norm = grad_clip_max_norm

        # mixed precision
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler()  

        # SAM 
        self.sam = sam
        if self.sam:
            self.sam_optimizer = SAM(
                base_optimizer=self.optimizer,
                rho=0.05,
                adaptive=True,  
                # Suggested by https://github.com/davda54/sam/blob/main/example/train.py
            )
            """
            self.optimizer = SAMOptimizer(
                self.optimizer, 
                rho=0.05, 
                epsilon=1e-12, 
                interval=1
            )
            self.scaler = ClosureGradScaler()
            """
            
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
        self.best_val_f1 = 0.0  # given imbalanced dataset

        if os.path.exists(last_snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(last_snapshot_path)
        
        # evaluate on final test set
        self.evaluate_on_final_test_set = evaluate_on_final_test_set

    def _load_snapshot(self, snapshot_path: str) -> None:
        """ Load snapshot of model """
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.module.load_state_dict(snapshot["model"])  # alternatively, use self.model.load_state_dict(...) but save self.model.state_dict() in _save_snapshot()
        self.optimizer.load_state_dict(snapshot["optimizer"])
        self.epochs_run = snapshot["epoch"]
        self.best_val_f1 = snapshot["best_val_f1"]
        if self.scheduler is not None:
            self.scheduler.load_state_dict(snapshot["scheduler"])
        if self.mixed_precision:
            self.scaler.load_state_dict(snapshot["scaler"]) 
        print(f"Successfully loaded snapshot from Epoch {self.epochs_run}. Best validation F1 so far: {self.best_val_f1}")
    
    def _save_snapshot(self, epoch: int, snapshot_path: str):
        """ Save snapshot of current model """
        snapshot = {
            "model": self.model.module.state_dict(),  # actual model is module wrapped by DDP
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_val_f1": self.best_val_f1,
        }
        if self.scheduler is not None:
            snapshot["scheduler"] = self.scheduler.state_dict()    
        if self.mixed_precision:
            snapshot["scaler"] = self.scaler.state_dict()
        torch.save(snapshot, snapshot_path)
        # print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}")
    
    def _train_batch(self, index: int, inputs: torch.Tensor, targets: torch.Tensor):
        """ Train on one batch """
        take_gradient_step = ((index + 1) % self.grad_accum_steps == 0) or ((index + 1) == len(self.train_dataloader))
        # Take gradient step only every self.grad_accum_steps steps or at the end of the epoch

        # Check class balance (for resampling)
        print(f"[GPU{self.gpu_id}] Batch {index} | Class balance: {torch.bincount(targets)}")
        
        with torch.autocast(device_type='cuda', dtype=torch.float16) if self.mixed_precision else nullcontext():
            with self.model.no_sync() if not take_gradient_step else nullcontext():               
                outputs = self.model(inputs)  
                # global_pool="avg": (batch_size, num_classes) with global average pooling
                # global_pool="": (batch_size, # tokens, num_classes) with no pooling -> outputs[:,0,:] gives cls token output (batch_size, num_classes)
                # global_pool="token": (batch_size, num_classes) e.g. gives cls token output as above
                # global_pool="map": ???
                
                loss = self.loss_fn(outputs, targets) / self.grad_accum_steps
                preds = torch.argmax(outputs, dim=-1)  # (batch_size)
                train_acc = torch.sum(preds == targets) / len(targets)
                print(f"[GPU{self.gpu_id}] Training loss: {loss}, Training accuracy: {train_acc}")

                # avg across all GPUs and send to rank 0 process
                torch.distributed.reduce(loss, op=torch.distributed.ReduceOp.AVG, dst=0)
                torch.distributed.reduce(train_acc, op=torch.distributed.ReduceOp.AVG, dst=0)
                if self.gpu_id == 0:
                    print(f"[MEAN] Training loss: {loss}, Training accuracy: {train_acc} \n")
                    wandb.log({"train_loss": loss, "train_acc": train_acc})
                
                self.scaler.scale(loss).backward() if self.mixed_precision else loss.backward()
        
        if take_gradient_step:
            if self.mixed_precision:
                if self.grad_clip_max_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.grad_clip_max_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                self.optimizer.step()
            self.optimizer.zero_grad()
    
    def _train_sam_batch(self, index: int, inputs: torch.Tensor, targets: torch.Tensor):
        """ 
        Train on one batch with SAM optimizer (Sharpness-Aware Minimization) which takes 2 forward-backward passes 
        NOTE: NO GRADIENT ACCUMULATION OR MIXED PRECISION WITH SAM OPTIMIZER!
        """
        # Check class balance (for resampling)
        print(f"[GPU{self.gpu_id}] Batch {index} | Class balance: {torch.bincount(targets)}")
        
        # first forward-backward pass
        enable_running_stats(self.model)  # compute running stats for batchnorm only in 1st pass
        with self.model.no_sync():  # do not sync 1st pass gradients across GPUs
            outputs = self.model(inputs)  # (batch_size, num_classes)
            loss = self.loss_fn(outputs, targets)
            preds = torch.argmax(outputs, dim=-1)  # (batch_size)
            train_acc = torch.sum(preds == targets) / len(targets)
            print(f"[GPU{self.gpu_id}] Training loss: {loss}, Training accuracy: {train_acc}")

            # avg loss and accuracy across all GPUs and send to rank 0 process
            torch.distributed.reduce(loss, op=torch.distributed.ReduceOp.AVG, dst=0)
            torch.distributed.reduce(train_acc, op=torch.distributed.ReduceOp.AVG, dst=0)
            if self.gpu_id == 0:
                print(f"[MEAN] Training loss: {loss}, Training accuracy: {train_acc} \n")
                wandb.log({"train_loss": loss, "train_acc": train_acc})
            
            loss.backward()
            if self.grad_clip_max_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
            self.sam_optimizer.first_step(zero_grad=True)
        
        # second forward-backward pass
        disable_running_stats(self.model) # don't compute running stats for batchnorm in 2nd pass
        outputs = self.model(inputs)  # (batch_size, num_classes)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        if self.grad_clip_max_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
        self.sam_optimizer.second_step(zero_grad=True)

    def _train_sam_batch_closure(self, index: int, inputs: torch.Tensor, targets: torch.Tensor):
        """ Train on one batch with SAM optimizer (Sharpness-Aware Minimization) using closure """
        
        # Check class balance (for resampling)
        print(f"[GPU{self.gpu_id}] Batch {index} | Class balance: {torch.bincount(targets)}")

        def closure():
            with self.model.no_sync():
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                self.scaler.scale(loss).backward()
                return loss
        
        with torch.autocast(device_type='cuda', dtype=torch.float16) if self.mixed_precision else nullcontext():
            outputs = self.model(inputs)  
            # global_pool="avg": (batch_size, num_classes) with global average pooling
            # global_pool="": (batch_size, # tokens, num_classes) with no pooling -> outputs[:,0,:] gives cls token output (batch_size, num_classes)
            # global_pool="token": (batch_size, num_classes) e.g. gives cls token output as above
            # global_pool="map": ???
            
            loss = self.loss_fn(outputs, targets)
            preds = torch.argmax(outputs, dim=-1)  # (batch_size)
            train_acc = torch.sum(preds == targets) / len(targets)
            print(f"[GPU{self.gpu_id}] Training loss: {loss}, Training accuracy: {train_acc}")

            # avg across all GPUs and send to rank 0 process
            torch.distributed.reduce(loss, op=torch.distributed.ReduceOp.AVG, dst=0)
            torch.distributed.reduce(train_acc, op=torch.distributed.ReduceOp.AVG, dst=0)
            if self.gpu_id == 0:
                print(f"[MEAN] Training loss: {loss}, Training accuracy: {train_acc} \n")
                wandb.log({"train_loss": loss, "train_acc": train_acc})
            
            self.scaler.scale(loss).backward() if self.mixed_precision else loss.backward()
        
        if self.mixed_precision:
            if self.grad_clip_max_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
            self.scaler.step(self.optimizer, closure=closure)
            self.scaler.update()
        else:
            if self.grad_clip_max_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
            self.optimizer.step()
        self.optimizer.zero_grad()

    def _validate(self, test: bool = False):
        """ Validate current model on validation or final test dataset"""
        metrics_dict = {"avg_loss": torch.tensor([0.0]).to(self.gpu_id)}
        torchmetrics_dict = torch.nn.ModuleDict({
            "auroc": metrics.BinaryAUROC(),
            "ap": metrics.BinaryAveragePrecision(),
            "precision": metrics.BinaryPrecision(),
            "recall": metrics.BinaryRecall(),
            "specificity": metrics.BinarySpecificity(),
            "f1": metrics.BinaryF1Score(),
            "accuracy": metrics.BinaryAccuracy(),
        }).to(self.gpu_id)
        
        dataloader = self.test_dataloader if test else self.val_dataloader
        
        self.model.eval()
        with torch.autocast(device_type='cuda', dtype=torch.float16) if self.mixed_precision else nullcontext():
            with torch.no_grad():
                for idx, batch in enumerate(dataloader):
                    inputs, targets = batch
                    inputs = inputs.to(self.gpu_id)
                    targets = targets.to(self.gpu_id)  # (batch_size,)
                    outputs = self.model(inputs)  # (batch_size, num_classes)
                    probs = F.softmax(outputs, dim=-1)  # (batch_size, num_classes)
                    probs_positive_class = probs[:,-1]  # (batch_size,)
                    print(f"[GPU{self.gpu_id}] Validation Batch {idx}: \nTargets: {targets} \nOutputs: {torch.argmax(probs, dim=-1)}")

                    metrics_dict["avg_loss"] += self.loss_fn(outputs, targets)  
                    for metric, fn in torchmetrics_dict.items():
                        fn(probs_positive_class, targets)
        
        metrics_dict["avg_loss"] /= len(dataloader)
        torch.distributed.reduce(metrics_dict["avg_loss"], op=torch.distributed.ReduceOp.AVG, dst=0) # avg across all GPUs and send to rank 0 process

        for metric, fn in torchmetrics_dict.items():
            metrics_dict[metric] = fn.compute()  # .compute() syncs across all GPUs
            fn.reset()

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
        # In distributed mode, calling the set_epoch() method at the beginning of each epoch before creating the DataLoader 
        # iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be always used.
        
        for index, batch in enumerate(self.train_dataloader):
            inputs, targets = batch
            inputs = inputs.to(self.gpu_id).to(memory_format=torch.channels_last)
            targets = targets.to(self.gpu_id)
            
            self.model.train()
            if self.sam:
                self._train_sam_batch(index, inputs, targets)
            else:
                self._train_batch(index, inputs, targets)
        
        if self.scheduler:
            self.scheduler.step(epoch + 1)  # update learning rate every epoch if using LR scheduler (alternatively, update every step)
            print(f"\n[GPU{self.gpu_id}] Epoch {epoch + 1} | Updating learning rate: ", self.scheduler._get_lr(epoch + 1))

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
                if self.best_val_f1 < metrics_dict["val_f1"]:
                    self._save_snapshot(epoch, self.best_snapshot_path)
                    self.best_val_f1 = metrics_dict["val_f1"]
        
        if self.evaluate_on_final_test_set:
            print(f"Finished training for {max_epochs} epochs. Evaluating best model snapshot on final test set")
            self.test_snapshot_path = self.best_snapshot_path
            self.test()
    
    def test(self):
        """ Test on final test set """
        self._load_snapshot(self.test_snapshot_path)
        metrics_dict = self._validate(test=True)
        if self.gpu_id == 0:
            print("\n=== Test Set Performance Metrics ===")
            for metric, value in metrics_dict.items():
                print(f"- {metric}: {value}")
            print("====================================\n")


def setup_ddp():
    """ Set up distributed data parallel (DDP) training """
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def set_seed_everywhere(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # True gives faster training for fixed input size but nondeterministic
    os.environ['PYTHONHASHSEED'] = str(seed)


def _data_transform(
        image_size: Union[int, Tuple[int, int]], 
        augment: str = None,
        randaugment_num_ops: int = 2,
        randaugment_mag: int = 9,
    ):
    """
    Parameters
    ----------
    image_size : Union[int, Tuple[int, int]]
        Input image size for model
            If int size is provided, input image size will be (size, size)
            If Tuple (height, width) is provided, input image size will be (height, width)
    augment : str
        Data augmentation strategy: None, trivialaugment, randaugment, autoaugment, augmix; by default None
    randaugment_num_ops : int
        Number of operations for RandAugment; by default 2 to match PyTorch default
    randaugment_mag : int
        Magnitude of each operation for RandAugment (max 31); by default 9 to match PyTorch default

    Returns
    -------
    Data transforms with or without data augmentation strategy
    """

    transforms_list = [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.CenterCrop(image_size),  # used for ImageNet and in original RETFound code but don't want to lose peripheral pathology
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ]

    if augment is not None:
        # torchvision data augmentations
        augmentations = {
            "trivialaugment": transforms.TrivialAugmentWide(),
            "randaugment": transforms.RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_mag),
            "autoaugment": transforms.AutoAugment(),
            "augmix": transforms.AugMix(),
        }
        assert augment in augmentations, f"Data augmentation strategy must be one of {list(augmentations.keys())}"
        transforms_list.insert(1, augmentations[augment])  #  insert after Resize

        """
        # timm data augmentations
        augment_strategy = augment.split("-")[0]
        assert augment_strategy in ["rand", "augmix"], "timm data augmentation config string must start with rand- or augmix-"
        RGB_IMAGENET_DEFAULT_MEAN = tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN])
        RGB_IMAGENET_DEFAULT_STD = tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_STD])
        if augment_strategy == "rand":  # insert RandAugment after Resize
            transforms_list.insert(1, rand_augment_transform(config_str=augment, hparams={"img_mean": RGB_IMAGENET_DEFAULT_MEAN, "img_std": RGB_IMAGENET_DEFAULT_STD}))
        else:  # augment_strategy == "augmix" - insert AugMix after Resize
            transforms_list.insert(1, augment_and_mix_transform(config_string=augment, hparams={"img_mean": RGB_IMAGENET_DEFAULT_MEAN, "img_std": RGB_IMAGENET_DEFAULT_STD}))
        """
    return transforms.Compose(transforms_list)


def _cutmixup_collate_fn(batch):
    """ Returns DataLoader collate function with CutMix and MixUp at random - for training dataloader only """
    # cutmix = transforms_v2.CutMix(num_classes=2)
    # mixup = transforms_v2.MixUp(num_classes=2)
    # cutmix_or_mixup = transforms_v2.RandomChoice([cutmix, mixup])
    # return cutmix_or_mixup(*default_collate(batch))


def _setup_dataloader(dataset: Dataset, batch_size: int, sampler: torch.utils.data.Sampler = None, collate_fn=None):
    """
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to load
    batch_size : int
        Batch size for dataloader
    sampler : torch.utils.data.Sampler, optional
        Sampler for dataloader, by default None which defaults to DistributedSampler
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
        num_workers=4,
        pin_memory=True,
        shuffle=False,  # DistributedSampler will handle shuffling
        sampler=DistributedSampler(dataset) if sampler is None else sampler,  # for DDP
        collate_fn=collate_fn,
    )


def setup_data(
        dataset_path: str, 
        image_size: Union[int, Tuple[int, int]],
        batch_size: int, 
        random_seed: int, 
        resample: bool, 
        num_augment_repeats: int,
        dataset_size: int, 
        augment: str, 
        randaugment_num_ops: int,
        randaugment_mag: int,
        # cutmixup: bool,
    ):
    """
    Parameters
    ----------
    dataset_path : str
        Path to root directory of PyTorch ImageFolder dataset
    image_size : Union[int, Tuple[int, int]]
        Input image size for model
            If int size is provided, input image size will be (size, size)
            If Tuple (height, width) is provided, input image size will be (height, width)
    batch_size : int
        Batch size for dataloaders
    random_seed : int
        Random seed for reproducibility
    resample : bool
        Whether to resample training dataset to balance classes
    num_augment_repeats : int
        Number of times to augment each image in the training dataset
    dataset_size : int
        Number of images to sample from the training set. Use all images if not specified.
    augment : str
        Data augmentation strategy: None, trivialaugment, randaugment, autoaugment, augmix
    randaugment_num_ops : int
        Number of operations for RandAugment
    randaugment_mag : int
        Magnitude of each operation for RandAugment
    cutmixup : bool
        Whether to use CutMix/MixUp data augmentation
    
    Returns
    -------
    train_dataloader, val_dataloader, test_dataloader : torch.utils.data.DataLoader
        Dataloaders for training, validation, and test sets
    """
    # Set up data augmentations
    train_collate_fn = None  # _cutmixup_collate_fn if cutmixup else None

    # Set up datasets
    train_dataset = datasets.ImageFolder(
        dataset_path + "/train", 
        transform=_data_transform(
            image_size=image_size, 
            augment=augment, 
            randaugment_num_ops=randaugment_num_ops, 
            randaugment_mag=randaugment_mag,
        )
    )
    val_dataset = datasets.ImageFolder(dataset_path + "/val", transform=_data_transform(image_size, augment=None))
    test_dataset = datasets.ImageFolder(dataset_path + "/test", transform=_data_transform(image_size, augment=None))

    # If specified, take a random subset of the training dataset of size 'dataset_size' 
    if dataset_size:
        indices = torch.randperm(len(train_dataset))[:dataset_size]
        print(indices)
        train_dataset = Subset(train_dataset, indices)

    # If specified, resample training dataset to balance classes with WeightedRandomSampler and wrap with DistributedSamplerWrapper for DDP
    # https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264
    # https://discuss.pytorch.org/t/how-to-get-no-of-images-in-every-class-by-datasets-imagefolder/81599 
    # https://github.com/pytorch/pytorch/issues/23430#issuecomment-750037457 
    if resample:  
        # Set up random number generator for reproducibility
        prng = torch.Generator()
        prng.manual_seed(random_seed)
        # Calculate weights for each sample in training dataset
        targets = torch.Tensor(train_dataset.targets).int()
        _, class_counts = torch.unique(targets, return_counts=True)
        class_weights = 1./class_counts
        sample_weights = torch.Tensor([class_weights[t].item() for t in targets])
        # Set up sampler
        train_sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), generator=prng)
        train_sampler = DistributedSamplerWrapper(train_sampler)  # for DDP
    
    elif num_augment_repeats is not None:
        train_sampler = RepeatedAugmentationSampler(
            train_dataset, 
            shuffle=True, 
            num_repeats=num_augment_repeats
        )
        # TODO: enforce class balance with repeated augmentations
    
    else:
        train_sampler = None
    
    # Set up and return dataloaders
    train_dataloader = _setup_dataloader(train_dataset, batch_size, train_sampler, train_collate_fn)
    val_dataloader = _setup_dataloader(val_dataset, batch_size)
    test_dataloader = _setup_dataloader(test_dataset, batch_size)

    return train_dataloader, val_dataloader, test_dataloader


def setup_model(
        model_arch: str, 
        modality: str, 
        training_strategy: str,
        checkpoint_path: str,
        num_classes: int, 
        image_size: int,
        layerscale_init_values: float,
        stochastic_depth_rate: float,
        patch_dropout_rate: float,
        position_embedding_dropout_rate: float,     
        attention_dropout_rate: float,
        projection_dropout_rate: float, 
        head_dropout_rate: float,
        global_pool: str,
    ):
    """
    Parameters
    ----------
    model_arch : str
        Model architecture: resnet50, vit_large, retfound
    modality : str
        Modality: CFP, OCT
    training_strategy : str
        Training strategy: full_finetune, linear_probe, surgical_finetune, sft_and_lp
    checkpoint_path : str
        Path to model checkpoint to finetune from,
    num_classes : int
        Number of classes
    image_size : int
        Input image size
    layerscale_init_values : float
        Layer scale initialization values for ViT
    stochastic_depth_rate : float
        Stochastic depth rate for ViT
    patch_dropout_rate : float
        Patch dropout rate for ViT
    position_embedding_dropout_rate : float
        Position embedding dropout rate for ViT
    attention_dropout_rate : float
        Attention dropout rate for ViT
    projection_dropout_rate : float
        Projection layer dropout rate for ViT
    head_dropout_rate : float
        Classification head dropout rate for ViT
    global_pool : str
        Which global pooling to use for timm ViT (>=0.6.5): '', 'avg', 'token', 'map'
    
    Returns
    -------
    model : torch.nn.Module
        Model to train
    """
    assert model_arch in ["resnet50", "vit_large", "retfound"], "Model architecture must be resnet50, vit_large, or retfound"
    assert args.training_strategy in ["full_finetune", "linear_probe", "surgical_finetune", "sft_and_lp"], "Training strategy must be full_finetune, linear_probe, surgical_finetune, or sft_and_lp"

    if model_arch == "resnet50":
        print("Setting up timm model: resnet50.a1_in1k")
        model = timm.create_model(
            'resnet50.a1_in1k', 
            pretrained=True, 
            num_classes=num_classes
        )
        if checkpoint_path is not None:
            print(f"Loading model weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')  # timm model checkpoints are nn.Module objects
            model.load_state_dict(checkpoint.state_dict(), strict=True) 
    
    elif model_arch == "vit_large":
        print("Setting up timm model: vit_large_patch16_224.augreg_in21k_ft_in1k")
        model = timm.create_model(
            'vit_large_patch16_224.augreg_in21k_ft_in1k', 
            pretrained=True, 
            num_classes=num_classes,
            img_size=image_size,
            init_values=layerscale_init_values,
            drop_path_rate=stochastic_depth_rate,
            patch_drop_rate=patch_dropout_rate,
            drop_rate=head_dropout_rate,
            pos_drop_rate=position_embedding_dropout_rate,
            attn_drop_rate=attention_dropout_rate,
            proj_drop_rate=projection_dropout_rate,                          
        )
        
        if checkpoint_path is not None:
            print(f"Loading model weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            checkpoint_model = checkpoint["model"]
            state_dict = model.state_dict()
            
            # adapt checkpoint model to vit model
            for k in ["head.weight", "head.bias"]:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from checkpoint due to shape mismatch")
                    del checkpoint_model[k]
            
            # interpolate position embedding following DeiT
            interpolate_pos_embed(model, checkpoint_model)
            
            # load adapted checkpoint model state
            msg = model.load_state_dict(checkpoint_model, strict=False)
            
            # manually initialize head fc layer following MoCo v3
            trunc_normal_(model.head.weight, std=.02)

    else:  # model_arch == "retfound" 
        print("Setting up RetFound model")
        model = models_vit.__dict__['vit_large_patch16'](
            num_classes=num_classes,
            img_size=image_size,
            global_pool=global_pool,
            init_values=layerscale_init_values,
            drop_path_rate=stochastic_depth_rate,
            patch_drop_rate=patch_dropout_rate,
            drop_rate=head_dropout_rate,
            pos_drop_rate=position_embedding_dropout_rate,
            attn_drop_rate=attention_dropout_rate,
            proj_drop_rate=projection_dropout_rate,
        )
        
        # load mae checkpoint
        mae_checkpoint_path = checkpoint_path if checkpoint_path else (OCT_CHKPT_PATH if modality == "OCT" else CFP_CHKPT_PATH)
        print(f"Loading model weights from {mae_checkpoint_path}")
        mae_checkpoint = torch.load(mae_checkpoint_path, map_location='cpu')
        mae_checkpoint_model = mae_checkpoint["model"]
        state_dict = model.state_dict()
        
        # adapt mae model to vit model
        for k in ["head.weight", "head.bias"]:
            if k in mae_checkpoint_model and mae_checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from checkpoint due to shape mismatch")
                del mae_checkpoint_model[k]
        
        # interpolate position embedding following DeiT
        interpolate_pos_embed(model, mae_checkpoint_model)
        
        # load mae model state
        msg = model.load_state_dict(mae_checkpoint_model, strict=False)
        if int(os.environ["LOCAL_RANK"]) == 0:
            print(msg)
        if global_pool == "avg":
            assert set(msg.missing_keys) >= {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}  # >= means "is superset of"
        else:
            assert set(msg.missing_keys) >= {"head.weight", "head.bias"}  # >= means "is superset of"
       
        # manually initialize head fc layer following MoCo v3
        trunc_normal_(model.head.weight, std=.02)

        # linear probing - freeze all parameters but the head
        if training_strategy == "linear_probe":
            # hack for linear probe: revise model head with batchnorm
            model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
            # freeze all parameters but the head
            for name, param in model.named_parameters():
                param.requires_grad = False
            for name, param in model.head.named_parameters():
                param.requires_grad = True
        
        # surgical fine-tuning for input-level shift
        # freeze all parameters but embedding layer (patch_embed, pos_embed)
        if training_strategy == "surgical_finetune":
            for name, param in model.named_parameters():
                param.requires_grad = False
            for name, param in model.patch_embed.named_parameters():
                param.requires_grad = True
            model.pos_embed.requires_grad = True
        
        if training_strategy == "sft_and_lp":
            # freeze all parameters but the head and embedding layers
            for name, param in model.named_parameters():
                param.requires_grad = False
            for name, param in model.head.named_parameters():
                param.requires_grad = True
            for name, param in model.patch_embed.named_parameters():
                param.requires_grad = True
            model.pos_embed.requires_grad = True
    
    model = model.to(memory_format=torch.channels_last)  # for faster training with convolutional, pooling layers
    return model


def setup_optimizer(
        model: torch.nn.Module, 
        algo: str,
        lr: float, 
        weight_decay: float
    ):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    algo : str
        Optimization algorithm to use: adam, adamw, lars, lamb
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
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "lars": LARS,
        "lamb": Lamb,
    }
    assert algo in optimization_algorithms, f"Optimizer must be one of {list(optimization_algorithms.keys())}"
    optim_algo = optimization_algorithms[algo]
    optimizer = optim_algo(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def setup_lr_scheduler(
    optimizer: torch.optim.Optimizer, 
    total_epochs: int, 
    algo: str,
    num_warmup_epochs: int,
    warmup_init_lr: float = 0,
) -> torch.optim.lr_scheduler:
    """
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer for training
    total_epochs : int
        Total number of epochs to train for
    algo : str, optional
        Learning rate scheduler algorithm: cosine, step, poly
    warmup_epochs : int
        Number of epochs to do linear learning rate warmup
    
    Returns
    -------
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler
    """
    scheduler = None

    # timm LR schedulers
    scheduling_algorithms = {
        "cosine": CosineLRScheduler(
            optimizer = optimizer, 
            t_initial = total_epochs,  # no restarts
            cycle_limit = 1,  #  no restarts
            warmup_t = num_warmup_epochs,
            warmup_lr_init = warmup_init_lr
        ),
        "poly": PolyLRScheduler(
            optimizer = optimizer, 
            t_initial = total_epochs,  # no restarts
            cycle_limit = 1,  #  no restarts
            warmup_t = num_warmup_epochs,
            warmup_lr_init = warmup_init_lr
        ),
        "step": StepLRScheduler(
            optimizer=optimizer,
            decay_t = total_epochs // (4 * total_epochs / 50),  # 4, 5 ... initial training runs done with 50 epochs
            decay_rate = 0.5,  # 0.25, 0.75
            warmup_t = num_warmup_epochs,
            warmup_lr_init = warmup_init_lr
        ), 
    }
    scheduler = scheduling_algorithms[algo]

    # PyTorch LR schedulers
    # warmup = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=num_warmup_epochs, verbose=True)
    # scheduling_algorithms = {
    #     "cosine": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, verbose=True),
    #     "exponential": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, verbose=True)
    # }
    # scheduler = torch.optim.lr_scheduler.SequentialLR(
    #     optimizer, 
    #     [warmup, scheduling_algorithms[algo]], 
    #     milestones=[num_warmup_epochs]
    # )

    return scheduler


def setup_loss_fn(modality: str = "none", label_smoothing: float = 0.0):
    """
    Parameters
    ----------
    modality : str
        Modality: CFP, OCT, PEDS_OPTOS, none. If none, do not weight loss function.
    label_smoothing : float, optional
        Add uniformly distributed smoothing when computing the loss; range 0.0-1.0 where 0.0 means no smoothing, by default 0.0

    Returns
    -------
    loss_fn : torch.nn.Module
        Loss function for training
    """
    # referable_CFP_OCT
    # CFP training set has 548 normal and 102 referable (0.19) -> class_weights = [1.0, 5.0]
    # OCT training set has 782 normal and 132 referable (0.17) -> class_weights = [1.0, 5.0]
    
    # peds_optos_uwf
    # Optos training set has 142 abnormal and 43 normal (0.33) -> class_weights = [1.0, 3.0]

    class_weights = {
        "CFP": torch.tensor([1.0, 5.0]).cuda(),
        "OCT": torch.tensor([1.0, 5.0]).cuda(),
        "PEDS_OPTOS": torch.tensor([1.0, 3.0]).cuda(),
    }
    return torch.nn.CrossEntropyLoss(
        weight=class_weights[modality] if modality in class_weights else None,
        label_smoothing=label_smoothing,
    )  


def main(args):
    setup_ddp()
    set_seed_everywhere(args.seed)

    # Set up correct dataset paths for modality
    dataset_paths = {
            "CFP": CFP_DATASET_PATH,
            "OCT": OCT_DATASET_PATH,
            "PEDS_OPTOS": PEDS_OPTOS_DATASET_PATH,
        }
    assert args.modality in dataset_paths, "Modality must be CFP, OCT, or PEDS_OPTOS"
    dataset_path = args.dataset_path if args.dataset_path else dataset_paths[args.modality] 

    # Set up model checkpoint paths
    last_snapshot_path = args.snapshot_path + "/last.pth"
    best_snapshot_path = args.snapshot_path + "/best.pth"

    # Set up dataset and data loader objects
    train_dataloader, val_dataloader, test_dataloader = setup_data(
        dataset_path=dataset_path, 
        image_size=(args.image_height, args.image_width),
        batch_size=args.batch_size, 
        random_seed=args.seed, 
        num_augment_repeats=args.num_augment_repeats,
        resample=args.resample, 
        dataset_size=args.dataset_size, 
        augment=args.augment,
        randaugment_num_ops=args.randaugment_num_ops,
        randaugment_mag=args.randaugment_mag,
        # cutmixup=args.cutmixup,
    )
    
    # Set up model, optimizer, scheduler, loss function
    model = setup_model(
        model_arch=args.model_arch, 
        modality=args.modality, 
        training_strategy=args.training_strategy, 
        checkpoint_path=args.checkpoint_path, 
        num_classes=args.num_classes,
        image_size=(args.image_height, args.image_width),
        global_pool=args.global_pool,
        layerscale_init_values=args.layerscale_init_values,
        stochastic_depth_rate=args.stochastic_depth_rate,
        patch_dropout_rate=args.patch_dropout_rate,
        position_embedding_dropout_rate=args.position_embedding_dropout_rate,
        attention_dropout_rate=args.attention_dropout_rate,
        projection_dropout_rate=args.projection_dropout_rate,
        head_dropout_rate=args.head_dropout_rate
    )
    
    optimizer = setup_optimizer(model, algo=args.optimizer, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = setup_lr_scheduler(optimizer, args.total_epochs, args.lr_scheduler, args.warmup_epochs) if args.lr_scheduler else None
    loss_fn = setup_loss_fn(modality=args.modality if args.weight_loss_fn else "none", label_smoothing=args.label_smoothing)

    if args.sam:
        assert not args.mixed_precision, "SAM not yet supported with mixed precision training"
        assert args.grad_accum_steps == 1, "SAM not yet supported with gradient accumulation"

    # Set up W&B logging 
    gpu_id = int(os.environ["LOCAL_RANK"])
    if gpu_id == 0:  # only log on main process
        wandb.init(
            project=args.wandb_proj_name,
            config=args,
            resume=False,  # if True, continue logging from previous run if did not finish
        )
        wandb_run_number = wandb.run.name.split("-")[-1]

    # Set up trainer
    trainer = Trainer(
        model=model, 
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        grad_accum_steps=args.grad_accum_steps,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        mixed_precision=args.mixed_precision,
        grad_clip_max_norm=args.grad_clip_max_norm,
        sam=args.sam,
        save_every=args.save_every,
        last_snapshot_path=last_snapshot_path,
        best_snapshot_path=best_snapshot_path,
        test_snapshot_path=args.final_test_snapshot_path,
        evaluate_on_final_test_set=args.evaluate_on_final_test_set,
    )
    
    # Test if given a final test snapshot
    if args.final_test_snapshot_path:
        trainer.test()
    
    # Otherwise, train model
    else: 
        trainer.train(args.total_epochs)
        # If training finishes successfully, delete the last.pth model snapshot using gpu 0
        # and rename best.pth to [wandb_run_name].pth
        if gpu_id == 0:
            print(f"\nTraining completed successfully! Deleting last.pth and saving best.pth as {wandb_run_number}-{wandb.run.name}.pth\n")
            if os.path.exists(last_snapshot_path): 
                os.remove(last_snapshot_path)
            if os.path.exists(best_snapshot_path):
                os.rename(best_snapshot_path, f"{args.snapshot_path}/{wandb_run_number}-{wandb.run.name}.pth")

    # Clean up
    wandb.finish()
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Random seed
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # Task/Dataset parameters
    parser.add_argument('--modality', type=str, default="OCT", help='Must be OCT, CFP, or PEDS_OPTOS')
    parser.add_argument('--dataset_path', type=str, help='Path to root directory of PyTorch ImageFolder dataset')
    parser.add_argument('--dataset_size', type=int, help='Number of training images to train the model with.')  # CFP: 650, OCT: 914
    parser.add_argument('--wandb_proj_name', default="retfound_referable_cfp_oct", type=str, help='Name of W&B project to log to')   
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes of interest')
    parser.add_argument('--image_height', type=int, default=224, help='Input image height for model')
    parser.add_argument('--image_width', type=int, default=224, help='Input image width for model')
    
    # Model and Training Strategy
    parser.add_argument('--model_arch', type=str, default="retfound", help='Model architecture: resnet50, vit_large, retfound')
    parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint to finetune from')
    parser.add_argument('--training_strategy', type=str, default="full_finetune", help="Training strategy: full_finetune, linear_probe, surgical_finetune, sft_and_lp")  
    parser.add_argument('--global_pool', type=str, default="token", help='timm ViT global pooling: "", "avg" (for global average pooling), "token" (use class token), "map" (?)')  
 
    # Batch size (can increase with mixed precision, gradient accumulation), Number of epochs
    parser.add_argument('--mixed_precision', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size on each device')  # max batch_size with AMP {full_finetune: 64, linear_probe: 1024+}
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Number of gradient accumulation steps before performing optimizer step. Effective batch size becomes batch_size * gradient_accumulation_steps * num_gpus')
    parser.add_argument('--total_epochs', type=int, default=30, help='Total epochs to train the model')
    
    # Optimizer, Learning Rate, Learning Rate Schedule
    parser.add_argument('--optimizer', type=str, default="lars", help='Optimizer: adam, adamw, lars, lamb')
    parser.add_argument('--sam', action='store_true', help='Use Sharpness-Aware Minimization')
    parser.add_argument('--learning_rate', type=float, default=0.3, help='Learning rate')
    parser.add_argument('--lr_scheduler', type=str, help='Learning rate scheduler: cosine, poly, step')  
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Epochs to warm up LR')

    # Data Augmentation
    parser.add_argument('--augment', type=str, help='Data augmentation strategy: None, trivialaugment, randaugment, autoaugment, augmix')
    parser.add_argument('--randaugment_num_ops', type=int, default=2, help='Number of operations for RandAugment, default 2 to match PyTorch default')
    parser.add_argument('--randaugment_mag', type=int, default=9, help='Magnitude of each operation for RandAugment, default 9 to match PyTorch default') 
    parser.add_argument('--num_augment_repeats', type=int, help='Number of augmented samples per sampled input image - RepeatedAugmentationSampler samples batch_size / num_augment_repeats input images per batch to maintain constant batch size')
    parser.add_argument('--cutmixup', action='store_true', help='Use CutMix/MixUp data augmentation')  

    # Regularization
    parser.add_argument('--grad_clip_max_norm', type=float, help='Gradient clipping: clip gradient 2-norm to this value')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Add uniformly distributed smoothing when computing the loss; range 0.0-1.0 where 0.0 means no smoothing')
    parser.add_argument('--layerscale_init_values', type=float, help='Layer scale initialization values for ViT, default None, commonly 1e-6')
    parser.add_argument('--stochastic_depth_rate', type=float, default=0.0, help='Stochastic depth rate for ViT')
    parser.add_argument('--patch_dropout_rate', type=float, default=0.0, help='Patch dropout rate for ViT')
    parser.add_argument('--position_embedding_dropout_rate', type=float, default=0.0, help='Position embedding dropout rate for ViT')
    parser.add_argument('--attention_dropout_rate', type=float, default=0.0, help='Attention dropout rate for ViT')
    parser.add_argument('--projection_dropout_rate', type=float, default=0.0, help='Projection layer dropout rate for ViT')
    parser.add_argument('--head_dropout_rate', type=float, default=0.0, help='Classification head dropout rate for ViT')

    # Class imbalance
    parser.add_argument('--weight_loss_fn', action='store_true', help='Weight loss function to balance classes')
    parser.add_argument('--resample', action='store_true', help='Resample training data to balance classes')

    # Checkpointing
    parser.add_argument('--save_every', type=int, default=1, help='Save a snapshot every _ epochs')
    parser.add_argument('--snapshot_path', type=str, 
                        default="/home/dkuo/RetFound/tasks/referable_CFP_OCT/snapshots",
                        help='Path to directory for saving training snapshots (last will be saved as last.pth, best so far will be saved as best.pth)')
    
    # Evaluate on final test set
    parser.add_argument('--evaluate_on_final_test_set', action='store_true', help='Evaluate best model snapshot on final test set at the end of training')
    parser.add_argument('--final_test_snapshot_path', type=str, help='Evaluate snapshot at provided path on final test set.')

    args = parser.parse_args()
    main(args)


# With GPUs 0,1,2 visible on 1 (standalone) machine, using all available GPUs, run main_ddp_torchrun.py [args] 
# > CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=gpu main_ddp_torchrun.py [args]
