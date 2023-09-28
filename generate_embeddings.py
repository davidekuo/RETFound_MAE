import argparse
import torch
import pandas as pd
import models_mae
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


CFP_CHKPT_PATH = "/home/dkuo/RetFound/model_checkpoints/RETFound_cfp_weights.pth"
OCT_CHKPT_PATH = "/home/dkuo/RetFound/model_checkpoints/RETFound_oct_weights.pth"

CFP_DATASET_PATH = "/home/dkuo/RetFound/tasks/referable_CFP_OCT/dataset/CFP"
OCT_DATASET_PATH = "/home/dkuo/RetFound/tasks/referable_CFP_OCT/dataset/OCT"

RETFOUND_EXPECTED_IMAGE_SIZE = 224


# H/t RETFound_visualize.ipynb
def prepare_model(chkpt_path, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


# H/t RETFound_MAE/util/datasets.py
def build_transform():
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


def main(args):
    # Check that modality is valid
    assert args.modality in ["CFP", "OCT"], "Modality must be CFP or OCT"
    
    # Load model from checkpoint
    mae_checkpoint_path = CFP_CHKPT_PATH if args.modality == "CFP" else OCT_CHKPT_PATH
    model = prepare_model(mae_checkpoint_path)

    dataset_path = args.dataset_path if args.dataset_path else (CFP_DATASET_PATH if args.modality == "CFP" else OCT_DATASET_PATH) 
    
    # Set up datasets
    train_dataset = datasets.ImageFolder(dataset_path + "/train", transform=build_transform())
    val_dataset = datasets.ImageFolder(dataset_path + "/val", transform=build_transform())
    test_dataset = datasets.ImageFolder(dataset_path + "/test", transform=build_transform())

    # Set up dataloaders - no shuffling as we want to preserve the order of the images
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Set up DataFrames to store embeddings with columns [filename, pathology_present, cls_token]
    train_df = pd.DataFrame(train_dataset.imgs, columns=["filename", "pathology_present"])
    val_df = pd.DataFrame(val_dataset.imgs, columns=["filename", "pathology_present"])
    test_df = pd.DataFrame(test_dataset.imgs, columns=["filename", "pathology_present"])

    # Set up lists to store embeddings
    train_embeddings = []
    val_embeddings = []
    test_embeddings = []

    # Generate embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    datasets = ["train", "val", "test"]
    dataloaders = [train_dataloader, val_dataloader, test_dataloader]
    dataset_embeddings = [train_embeddings, val_embeddings, test_embeddings]
    dataframes = [train_df, val_df, test_df]

    for dataset, dataloader, embeddings, df in zip(datasets, dataloaders, dataset_embeddings, dataframes):

        for batch in dataloader:
            images, labels = batch
            images = images.to(device)
            
            with torch.no_grad():
                x, mask, ids_restore = model.forward_encoder(images, mask_ratio=0.0)   
                # shape: (batch_size, num_patches+1, 1024)
            
            # cls_token is appended in front of patch embeddings, shape: (batch_size, 1024)
            cls_tokens = x[:,0,:]
            embeddings.append(cls_tokens.to("cpu"))
        
        # Save embeddings to DataFrame
        embeddings = torch.cat(embeddings, dim=0)
        df['cls_token'] = embeddings.tolist()
        df.to_csv(f"{dataset_path}/{args.modality}_{dataset}_embeddings.csv", index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', default="CFP", type=str, help='Must be CFP or OCT')
    parser.add_argument('--dataset_path', type=str, help='Path to root directory of PyTorch ImageFolder dataset')
    parser.add_argument('--batch_size', default=64, type=int, 
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Image input size: RetFound expects 224x224')
    args = parser.parse_args()
    main(args)