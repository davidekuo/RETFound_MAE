## RETFound - A foundation model for retina imaging

### About

This is a fork of the original RETFound repository modified for personal learning and research projects. The official RETFound repository can be found [here](https://github.com/rmaphoh/RETFound_MAE), and the original Masked Autoencoders repository on which RETFound is based can be found [here](https://github.com/facebookresearch/mae).

### Code Overview

The main code can be found in `main_ddp_torchrun.py` which can be run with the following shell command to fine-tune RETFound (or ViT-Large or ResNet-50) on PyTorch ImageFolder-compatible image classification datasets with distributed data parallelism (DDP), automatic mixed precision (AMP), and W&B logging, among other features: 
```
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=gpu main_ddp_torchrun.py [args]
```

Currently unused files from the original RETFound and MAE repositories kept for reference (I am working on re-implementing MAE pretraining) have been prefixed with `rf_` and `mae_` respectively, and will be eventually removed. 

### Set up

Create and activate [conda](https://docs.conda.io/projects/miniconda/en/latest/index.html) environment:

```
conda create -n retfound python=3.7.5 -y
conda activate retfound
```

Clone this repository:
```
git clone https://github.com/davidekuo/RETFound_MAE/
cd RETFound_MAE
```

Pip install dependencies:
```
pip install -r requirements.txt
```

Download the provided RETFound pre-trained weights to a directory outside of the repository (e.g. `../model_checkpoints/`):

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Large</th>
<!-- TABLE BODY -->
<tr><td align="left">CFP</td>
<td align="center"><a href="https://drive.google.com/file/d/1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE/view?usp=sharing">download</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">OCT</td>
<td align="center"><a href="https://drive.google.com/file/d/1m6s7QYkjyjJDlpEuXm7Xp3PmjN-elfW2/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

Modify `CFP_CHKPT_PATH`, `OCT_CHKPT_PATH`, and dataset paths in lines 37-42 of `main_ddp_torchrun.py` to point to the correct paths.

Start training!
