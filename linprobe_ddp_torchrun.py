# Adapted from PyTorch DistributedDataParallel tutorial:
# https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu_torchrun.py
# and from the PyTorch implementation of "Masked Autoencoders Are Scalable Vision Learners"
# https://arxiv.org/abs/2111.06377
# https://github.com/facebookresearch/mae/tree/main
#
# Run with
# > torchrun --standalone --nproc_per_node=4 lin_probe_ddp_torchrun.py [args]

import argparse
import os

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

RETFOUND_EXPECTED_IMAGE_SIZE = 224


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            loss_fn,
            save_every: int,
            last_snapshot_path: str,
            best_snapshot_path: str,
            test_snapshot_path: str = None,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
                
        self.epochs_run = 0
        self.save_every = save_every
        self.last_snapshot_path = last_snapshot_path
        if os.path.exists(last_snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(last_snapshot_path)

        self.best_snapshot_path = best_snapshot_path
        if not self.best_val_ap:
            self.best_val_ap = 0.0
        
        self.test_snapshot_path = test_snapshot_path

    def _load_snapshot(self, snapshot_path: str) -> None:
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.best_val_ap = snapshot["BEST_VAL_AP"]
        print(f"Successfully loaded snapshot from Epoch {self.epochs_run}")
    
    def _save_snapshot(self, epoch, snapshot_path):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),  # actual model is module wrapped by DDP
            "EPOCHS_RUN": epoch,
            "BEST_VAL_AP": self.best_val_ap,
        }
        torch.save(snapshot, snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}")
    

    def _train_batch(self, inputs, targets):
        self.model.train()
        self.optimizer.zero_grad()
        
        # with torch.cuda.amp.autocast():  # mixed precision
        outputs = self.model(input)
        loss = self.loss_fn(outputs, targets)
        wandb.log({"train_loss": loss})
        
        loss.backward()
        self.optimizer.step()
    
    def _validate(self, test=False):
        metrics_dict = {"avg_loss": 0.0}
        torchmetrics_dict = {
            "auroc": metrics.BinaryAUROC(thresholds=None),
            "ap": metrics.BinaryAveragePrecision(thresholds=None),
            "precision": metrics.BinaryPrecision(thresholds=None),
            "recall": metrics.BinaryRecall(thresholds=None),
            "f1": metrics.BinaryF1Score(thresholds=None),
            "accuracy": metrics.BinaryAccuracy(thresholds=None),
        }

        dataloader = self.test_dataloader if test else self.val_dataloader
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                outputs = self.model(inputs)

                for metric, fn in torchmetrics_dict.items():
                    fn(outputs, targets)
        
        metrics_dict["avg_loss"] /= len(dataloader)
        for metric, fn in torchmetrics_dict.items():
            metrics_dict[metric] = fn.compute()
            fn.reset()

        if not test:
            metrics_dict = {f"val_{metric}": value for metric, value in metrics_dict.items()}
            wandb.log(metrics_dict)

        return metrics_dict

    def _train_epoch(self, epoch):
        batch_size = len(next(iter(self.train_dataloader))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batch size {batch_size} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        
        for batch in self.train_dataloader:
            inputs, targets = batch
            inputs = inputs.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._train_batch(inputs, targets)
        
        return self._validate()

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            metrics_dict = self._train_epoch(epoch)
            self.epochs_run += 1
            if self.gpu_id == 0: 
                if epoch % self.save_every == 0:
                    self._save_snapshot(epoch, self.last_snapshot_path)
                if self.best_val_ap < metrics_dict["val_ap"]:
                    self._save_snapshot(epoch, self.best_snapshot_path)
    
    def test(self):
        self._load_snapshot(self.test_snapshot_path)
        metrics_dict = self._validate(test=True)
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


def setup_model(checkpoint_path, arch='vit_large_patch16'):
    # instantiate model
    model = getattr(models_vit, arch)()
    
    # load mae checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_model = checkpoint["model"]
    state_dict = checkpoint_model.state_dict()

    # adapt mae model to vit model
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from checkpoint due to shape mismatch")
            del checkpoint_model[k]
    
    # interpolate position embedding following DeiT
    interpolate_pos_embed(model, checkpoint_model)
    
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    assert set(msg.missing_keys) == {"head.weight", "head.bias"}

    # manually initialize head fc layer following MoCo v3
    trunc_normal_(model.head.weight, std=.02)

    # hack for linear probe: revise model head with batchnorm
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)

    # freeze all parameters but the head
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.head.named_parameters():
        param.requires_grad = True
    
    return model


def setup_optimizer(model, lr=0.03, weight_decay=0):
    optimizer = LARS(model.head.parameters(), lr, weight_decay)
    print(optimizer)
    return optimizer


def setup_loss_fn():
    return F.cross_entropy()


def main(args):
    setup_ddp()
    torch.manual_seed(0)

    # Set up training objects
    train_dataloader, val_dataloader, test_dataloader = setup_data(args.dataset_path, args.batch_size)
    model = setup_model(args.snapshot_path)
    loss_fn = setup_loss_fn()
    optimizer = setup_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    #  Set up W&B logging
    wandb.init(
        project="retfound_referable_cfp_oct",
        config=args,
    )

    # Set up trainer
    trainer = Trainer(
        model, 
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_fn,
        args.save_every,
        args.last_snapshot_path,
        args.best_snapshot_path,
        args.test_snapshot_path,
    )
    
    if args.test_snapshot_path:
        trainer.test()
        return
    
    # Train
    trainer.train(args.total_epochs)

    # Clean up
    wandb.finish()
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=1, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--last_snapshot_path', type=str, help='Path to last training snapshot')
    parser.add_argument('--best_snapshot_path', type=str, help='Path to best snapshot so far')
    parser.add_argument('--test_snapshot_path', type=str, help='Evaluate snapshot at provided path on final test set.')
    args = parser.parse_args()
    main(args)

# Run with
# > torchrun --standalone --nproc_per_node=4 lin_probe_ddp_torchrun.py [args]

"""
TODO:
- test
- W&B
- mixed precision
- hydra
"""