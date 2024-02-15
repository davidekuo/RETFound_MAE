import argparse
import numpy as np
import os
import random
import timm
import torch
import wandb

from byol_pytorch import BYOL
from contextlib import nullcontext
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


CFP_DATASET_PATH = "/home/dkuo/RetFound/tasks/referable_CFP_OCT/dataset/CFP"
OCT_DATASET_PATH = "/home/dkuo/RetFound/tasks/referable_CFP_OCT/dataset/OCT"
PEDS_OPTOS_DATASET_PATH = "/home/dkuo/RetFound/tasks/peds_optos/dataset" 

RETFOUND_EXPECTED_IMAGE_SIZE = 224


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
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
        )
        self.dataloader = data_loader
        self.optimizer = optimizer
        self.grad_accum_steps = grad_accum_steps
        
        # mixed precision
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler()

        # snapshot/checkpointing
        self.epochs_run = 0
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        if os.path.exists(self.snapshot_path):
            print(f"Loading snapshot from {self.snapshot_file_path}")
            self._load_snapshot(self.snapshot_file_path)
    
    def _load_snapshot(self, snapshot_path: str) -> None:
        """Load model and optimizer state from snapshot"""
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.module.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}.")
    
    def _save_snapshot(self, epoch: int, snapshot_path: str) -> None:
        """Save model and optimizer state to snapshot"""
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),  # actual model is module wrapped by DDP
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "EPOCHS_RUN": self.epochs_run,
        }
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
            
            self.epochs_run += 1
            # print(f"[GPU{self.gpu_id}] Epoch {epoch} | Loss: {loss.item()}")  # TODO: print epoch loss
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch, self.snapshot_path)


def main(args):
    # Set up DDP
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    gpu_id = int(os.environ["LOCAL_RANK"])

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # For faster training, requires fixed input size
    torch.backends.cudnn.benchmark = True  

    # Set up model
    resnet50 = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=2).to(gpu_id)
    model = BYOL(
        resnet50,
        image_size = RETFOUND_EXPECTED_IMAGE_SIZE,
        hidden_layer = 'global_pool',
        use_momentum = True,
    )
    # Note: loss is returned in forward function of BYOL class

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

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
            resume=True,
        )

    # Train
    trainer = Trainer(
        model=model,
        data_loader=dataloader,
        optimizer=optimizer,
        save_every=args.save_every,
        snapshot_path=args.snapshot_path,
        grad_accum_steps=args.grad_accum_steps,
        mixed_precision=args.mixed_precision,
    )
    trainer.train(args.total_epochs)

    # If training finished successfully, rename model snapshot to [wandb_run_name].pth
    if gpu_id == 0:
        print(f"\nTraining completed successfully! Renaming byol.pth as {wandb.run.name}.pth\n")
        if os.path.exists(args.snapshot_path):
            os.rename(args.snapshot_path, args.snapshot_path.replace("byol.pth", f"{wandb.run.name}.pth"))

    # Clean up
    wandb.finish()
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_proj_name', default="retfound_referable_cfp_oct", type=str, help='Name of W&B project to log to')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    parser.add_argument('--dataset_path', type=str, 
                        default="/home/dkuo/RetFound/tasks/referable_CFP_OCT/dataset/CFP",
                        help='Path to root directory of PyTorch ImageFolder dataset')
                        # CFP: "/home/dkuo/RetFound/tasks/referable_CFP_OCT/dataset/CFP"
                        # OPTOS: "/home/dkuo/RetFound/tasks/peds_optos/dataset"
    parser.add_argument('--snapshot_path', type=str, 
                        default="/home/dkuo/RetFound/tasks/referable_CFP_OCT/snapshots/byol.pth",
                        help='Path to save training snapshots to')
    parser.add_argument('--save_every', type=int, default=1, help='Save snapshot every _ epochs')
    
    parser.add_argument('--total_epochs', type=int, default=10, help='Total epochs to train the model')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size on each device')  # max batch_size with ResNet-50, AMP is 128
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Number of gradient accumulation steps before performing optimizer step. Effective batch size becomes batch_size * gradient_accumulation_steps * num_gpus')
    parser.add_argument('--mixed_precision', type=bool, default=True, help='Use automatic mixed precision')
    
    args = parser.parse_args()
    main(args)


# With GPUs 0,1,2 visible on 1 (standalone) machine, using all available GPUs, run main_ddp_torchrun.py [args] 
# > CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=gpu main_ddp_torchrun_byol.py [args]
