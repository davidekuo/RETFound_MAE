# OCT Experiments
# ================
#
# Run dataset size experiments with best-performing training recipe so far (DDP, randaugment, lars, lr 0.1, batch_size 64, epochs 50)
# Increase # epochs with decreasing dataset size to keep total training time relatively constant
#
# Dataset size | # epochs
# -----------------------
# Full         | 50
# 500          | 55     (59/53 ~ 1.1)
# 250          | 100
# 100          | 200
# 50           | 400

CUDA_VISIBLE_DEVICES=0,2 torchrun --standalone --nproc_per_node=gpu main_ddp_torchrun.py --dataset_size 500 --epochs 55
CUDA_VISIBLE_DEVICES=0,2 torchrun --standalone --nproc_per_node=gpu main_ddp_torchrun.py --dataset_size 250 --epochs 100
CUDA_VISIBLE_DEVICES=0,2 torchrun --standalone --nproc_per_node=gpu main_ddp_torchrun.py --dataset_size 100 --epochs 200
CUDA_VISIBLE_DEVICES=0,2 torchrun --standalone --nproc_per_node=gpu main_ddp_torchrun.py --dataset_size 50 --epochs 400
