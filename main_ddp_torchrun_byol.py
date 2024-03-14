import argparse
import numpy as np
import os
import random
import timm
import torch
import wandb

from byol_pytorch import BYOL
from contextlib import nullcontext
from util.lars import LARS
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.optim import Lamb
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


CFP_DATASET_PATH = "/home/dkuo/RetFound/tasks/referable_CFP_OCT/dataset/CFP"
OCT_DATASET_PATH = "/home/dkuo/RetFound/tasks/referable_CFP_OCT/dataset/OCT"
PEDS_OPTOS_DATASET_PATH = "/home/dkuo/RetFound/tasks/peds_optos/dataset" 

RETFOUND_EXPECTED_IMAGE_SIZE = 224
NUM_CLASSES = 2


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            save_every: int,
            snapshot_path: str,
            grad_accum_steps: int,
            mixed_precision: bool = False,

    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])

        # training
        self.model = model.to(self.gpu_id)
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[self.gpu_id],
            find_unused_parameters=True,  # for gradient accumulation
            gradient_as_bucket_view=True,  # for faster DDP training (in theory)
            static_graph=False,  # errors when set to True
        )
        self.dataloader = data_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_accum_steps = grad_accum_steps
        
        # mixed precision
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler()

        # snapshot/checkpointing
        self.epochs_run = 0
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        if os.path.exists(self.snapshot_path):
            print(f"Loading snapshot from {self.snapshot_path}")
            self._load_snapshot(self.snapshot_path)
    
    def _load_snapshot(self, snapshot_path: str) -> None:
        """Load model and optimizer state from snapshot"""
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.module.load_state_dict(snapshot["model"])
        self.optimizer.load_state_dict(snapshot["optimizer"])
        self.epochs_run = snapshot["epoch"]
        if self.scheduler is not None:
            self.scheduler.load_state_dict(snapshot["scheduler"])
        if self.mixed_precision:
            self.scaler.load_state_dict(snapshot["scaler"])
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}.")
    
    def _save_snapshot(self, epoch: int, snapshot_path: str) -> None:
        """Save model and optimizer state to snapshot"""
        snapshot = {
            "model": self.model.module.state_dict(),  # actual model is module wrapped by DDP
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epochs_run,
        }
        if self.scheduler is not None:
            snapshot["scheduler"] = self.scheduler.state_dict()
        if self.mixed_precision:
            snapshot["scaler"] = self.scaler.state_dict()
        torch.save(snapshot, snapshot_path)
        print(f"Epoch {epoch} | Saving snapshot to {snapshot_path}.")
    
    def train(self, total_epochs: int) -> None:
        """Train model to total_epochs"""
        for epoch in range(self.epochs_run, total_epochs):
            batch_size = len(next(iter(self.dataloader))[0])
            print(f"\n[GPU{self.gpu_id}] Epoch {epoch} | Batch size {batch_size} | Steps: {len(self.dataloader)}")
            self.dataloader.sampler.set_epoch(epoch)

            self.model.train()
            self.optimizer.zero_grad()

            for index, (inputs, _) in enumerate(self.dataloader):
                take_gradient_step = ((index + 1) % self.grad_accum_steps == 0) or ((index + 1) == len(self.dataloader))
                inputs = inputs.to(self.gpu_id).to(memory_format=torch.channels_last)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16) if self.mixed_precision else nullcontext():
                    with self.model.no_sync() if not take_gradient_step else nullcontext():
                        loss = self.model(inputs) / self.grad_accum_steps
                        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batch {index} | Loss: {loss}")
                        if self.gpu_id == 0:
                            wandb.log({"loss": loss})
                        self.scaler.scale(loss).backward() if self.mixed_precision else loss.backward()

                if take_gradient_step:
                    print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batch {index} | Taking gradient step")
                    if self.mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()  # update learning rate every epoch if using LR scheduler (alternatively, update every batch)
            
            self.epochs_run += 1
            # print(f"[GPU{self.gpu_id}] Epoch {epoch} | Loss: {loss.item()}")  # TODO: print epoch loss
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch, self.snapshot_path)


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
    warmup_epochs: int = 0,
) -> torch.optim.lr_scheduler:
    """
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer for training
    total_epochs : int
        Total number of epochs to train for
    algo : str, optional
        Learning rate scheduler algorithm: cosine, exponential
    warmup_epochs : int
        Number of epochs to do linear learning rate warmup
    
    Returns
    -------
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler
    """
    scheduler = None
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=warmup_epochs, verbose=True)
    scheduling_algorithms = {
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, verbose=True),
        "exponential": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, verbose=True)
    }
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        [warmup, scheduling_algorithms[algo]], 
        milestones=[warmup_epochs]
    )
    return scheduler


def set_seed_everywhere(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # True gives faster training for fixed input size but nondeterministic
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(args):
    # Set up DDP
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    gpu_id = int(os.environ["LOCAL_RANK"])

    # Set seed
    set_seed_everywhere(args.seed)

    # Set up model
    resnet50 = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=NUM_CLASSES).to(gpu_id)
    model = BYOL(
        resnet50,
        image_size = RETFOUND_EXPECTED_IMAGE_SIZE,
        hidden_layer = 'global_pool',
        use_momentum = args.use_momentum,
    )
    # NOTE: loss is returned in forward function of BYOL class
    # NOTE: can access resnet50 as model.net
    model = model.to(memory_format=torch.channels_last)  # for faster training with convolutional, pooling layers

    # Set up optimizer and (optional) LR scheduler
    optimizer = setup_optimizer(model, args.optimizer, args.learning_rate, args.weight_decay)
    print("Optimizer: ", optimizer)
    scheduler = setup_lr_scheduler(optimizer, args.total_epochs, args.lr_scheduler, args.warmup_epochs) if args.lr_scheduler else None
    print("Scheduler: ", scheduler)

    # Set up data/dataloaders - no augmentations needed
    transform = transforms.Compose([
        transforms.Resize(RETFOUND_EXPECTED_IMAGE_SIZE, 
                            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(RETFOUND_EXPECTED_IMAGE_SIZE),  # or Resize([224, 224])? 
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    dataset = datasets.ImageFolder(args.dataset_path, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # DistributedSampler will handle shuffling
        sampler=DistributedSampler(dataset)
    )

    # Set up W&B logging for loss
    if gpu_id == 0:
        wandb.init(
            project=args.wandb_proj_name,
            config=args,
            resume=False,  # if True, continue logging from previous run if did not finish
        )

    # Train
    trainer = Trainer(
        model=model,
        data_loader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        save_every=args.save_every,
        snapshot_path=args.snapshot_path,
        grad_accum_steps=args.grad_accum_steps,
        mixed_precision=args.mixed_precision,
    )
    trainer.train(args.total_epochs)

    # If training finished successfully, save final ResNet50 model to byol_[wandb_run_name].pth and remove byol.pth
    if gpu_id == 0:
        print(f"\nTraining completed successfully! Saving final ResNet50 model as byol_{wandb.run.name}.pth\n")
        torch.save(resnet50, args.snapshot_path.replace("byol.pth", f"byol_{wandb.run.name}.pth"))
        if os.path.exists(args.snapshot_path):
            os.remove(args.snapshot_path)

    # Clean up
    wandb.finish()
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Random seed
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Task & data parameters
    parser.add_argument('--wandb_proj_name', default="retfound_referable_cfp_oct", type=str, help='Name of W&B project to log to')
    parser.add_argument('--dataset_path', type=str, 
                        default="/home/dkuo/RetFound/tasks/referable_CFP_OCT/dataset/CFP",
                        help='Path to root directory of PyTorch ImageFolder dataset')
                        # CFP: "/home/dkuo/RetFound/tasks/referable_CFP_OCT/dataset/CFP"
                        # OPTOS: "/home/dkuo/RetFound/tasks/peds_optos/dataset"
    
    # Training parameters
    parser.add_argument('--use_momentum', type=bool, default=True, help='True: BYOL, False: SimSiam')
    parser.add_argument('--total_epochs', type=int, default=10, help='Total epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size on each device')  # max batch_size with ResNet-50, AMP is 128
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Number of gradient accumulation steps before performing optimizer step. Effective batch size becomes batch_size * gradient_accumulation_steps * num_gpus')
    parser.add_argument('--mixed_precision', type=bool, default=True, help='Use automatic mixed precision')
    
    parser.add_argument('--optimizer', type=str, default="adamw", help='Optimizer: adam, adamw, lars, lamb')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--lr_scheduler', type=str, help='Learning rate scheduler: cosine, exponential, plateau')  
    parser.add_argument('--warmup_epochs', type=int, default=5, help='epochs to warm up LR')

    # Snapshot/checkpointing parameters
    parser.add_argument('--save_every', type=int, default=1, help='Save snapshot every _ epochs')
    parser.add_argument('--snapshot_path', type=str, 
                        default="/home/dkuo/RetFound/tasks/referable_CFP_OCT/snapshots/byol.pth",
                        help='Path to save training snapshots to')
    
    args = parser.parse_args()
    main(args)


# With GPUs 0,1,2 visible on 1 (standalone) machine, using all available GPUs, run main_ddp_torchrun.py [args] 
# > CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=gpu main_ddp_torchrun_byol.py [args]
