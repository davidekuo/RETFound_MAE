import argparse
import numpy as np
import os
import random
import timm
import torch
import wandb

from byol_pytorch import BYOL
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


RETFOUND_EXPECTED_IMAGE_SIZE = 224


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            save_every: int,
            snapshot_path: str,
            mixed_precision: bool = False,

    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[self.gpu_id],
            gradient_as_bucket_view=True,
            static_graph=True,  # may need to set to False
        )
        self.dataloader = data_loader
        self.optimizer = optimizer
        self.mixed_precision = mixed_precision

        # training snapshots
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
        self.model.module.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Successfully loaded snapshot from Epoch {self.epochs_run}.")
    
    def _save_snapshot(self, epoch: int, snapshot_path: str) -> None:
        """Save model and optimizer state to snapshot"""
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),  # actual model is module wrapped by DDP
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "EPOCHS_RUN": self.epochs_run,
        }
        torch.save(snapshot, snapshot_path)
        print(f"Epoch {epoch} | Saving snapshot to {snapshot_path}.")
    
    def train(self, max_epochs: int) -> None:
        """Train model to max_epochs"""
        for epoch in range(self.epochs_run, max_epochs):
            batch_size = len(next(iter(self.dataloader))[0])
            print(f"\n[GPU{self.gpu_id}] Epoch {epoch} | Batch size {batch_size} | Steps: {len(self.dataloader)}")
            self.dataloader.sampler.set_epoch(epoch)

            self.model.train()
            for inputs, targets in self.dataloader:
                inputs = inputs.to(self.gpu_id).to(memory_format=torch.channels_last)
                self.optimizer.zero_grad()
                loss = self.model(inputs)
                print(f"[GPU{self.gpu_id}] Loss: {loss}")

                if self.gpu_id == 0:
                    wandb.log({"loss": loss})

                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
            
            self.epochs_run += 1
            print(f"Epoch {epoch} | Loss: {loss.item()}")
            if epoch % self.save_every == 0:
                self._save_snapshot(epoch, self.snapshot_path)


def main(args):
    # Set up DDP
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # For faster training, requires fixed input size
    torch.backends.cudnn.benchmark = True  

    # Set up model
    resnet50 = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=2)
    model = BYOL(
        resnet50,
        image_size = RETFOUND_EXPECTED_IMAGE_SIZE,
        hidden_layer = 'avgpool',
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
    gpu_id = int(os.environ["LOCAL_RANK"])
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
        mixed_precision=args.mixed_precision,
    )
    trainer.train(args.max_epochs)

    # If training finished successfully, rename model snapshot to [wandb_run_name].pth
    if gpu_id == 0:
        if os.path.exists(args.snapshot_path):
            os.rename(args.snapshot_path, f"{wandb.run.name}.pth")

    # Clean up
    wandb.finish()
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_proj_name', default="retfound_referable_cfp_oct", type=str, help='Name of W&B project to log to')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    parser.add_argument('--dataset_path', type=str, help='Path to root directory of PyTorch ImageFolder dataset')
    parser.add_argument('--snapshot_path', type=str, help='Path to save training snapshots to')
    parser.add_argument('--save_every', type=int, default=1, help='Save snapshot every _ epochs')
    
    parser.add_argument('--mixed_precision', type=bool, default=True, help='Use automatic mixed precision')
    parser.add_argument('--total_epochs', type=int, default=30, help='Total epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size on each device')  # max batch_size with AMP {full_finetune: 64, linear_probe: 1024+}
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    
    args = parser.parse_args()
    main(args)