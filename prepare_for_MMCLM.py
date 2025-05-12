import os
import logging
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from scipy.signal import resample
import time
from torch.cuda.amp import GradScaler, autocast
from ecg_jepa import ecg_jepa
from timm.scheduler import CosineLRScheduler
from ecg_data_500Hz_ptbxl import *
import argparse
import psutil
import matplotlib.pyplot as plt

print("script starting here")


def downsample_waves(waves, new_size):
    return np.array([resample(wave, new_size, axis=1) for wave in waves])


# Argument parser
parser = argparse.ArgumentParser(description="Pretrain the JEPA model with ECG data")
parser.add_argument(
    "--mask_scale", type=float, nargs=2, default=[0.175, 0.225], help="Scale of masking"
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
parser.add_argument(
    "--mask_type", type=str, default="block", help="Type of masking"
)  # 'block' or 'random'
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--wd", type=float, default=0.05, help="Weight decay")
parser.add_argument(
    "--data_dir_shao",
    type=str,
    default="/mount/ecg/physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords/",
    help="Directory for Shaoxing data",
)
parser.add_argument(
    "--data_dir_code15",
    type=str,
    default="/mount/ecg/code15",
    help="Directory for Code15 data",
)
parser.add_argument(
    "--data_dir_ptbxl", type=str, default="", help="Directory for PTBXL data"
)
parser.add_argument(
    "--data_dir_cpsc", type=str, default="", help="Directory for CPSC data"
)
parser.add_argument(
    "--data_dir_ibk", type=str, default="", help="Directory for IBK data"
)
args = parser.parse_args()


# Access the arguments like this
mask_scale = tuple(args.mask_scale)
batch_size = args.batch_size
lr = args.lr
mask_type = args.mask_type
epochs = args.epochs
wd = args.wd
data_dir_shao = args.data_dir_shao
data_dir_code15 = args.data_dir_code15
data_dir_ptbxl = args.data_dir_ptbxl
data_dir_cpsc = args.data_dir_cpsc
data_dir_ibk = args.data_dir_ibk

# Generate timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create logs directory if it doesn't exist
save_dir = f"./weights/ecg_jepa_{timestamp}_{mask_scale}"
os.makedirs(save_dir, exist_ok=True)
log_file = os.path.join(save_dir, f"training_{timestamp}.log")

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def log_params(params_dict):
    for key, value in params_dict.items():
        logging.info(f"{key}: {value}")
        print(f"{key}: {value}")


os.makedirs(save_dir, exist_ok=True)

start_time = time.time()


# IBK
waves_ibk = waves_ibk(data_dir_ibk)
# waves_shaoxing = downsample_waves(waves_shaoxing, 5000)
print(f"IBK waves shape: {waves_ibk.shape}")
logging.info(f"IBK waves shape: {waves_ibk.shape}")
dataset_ibk = ECGDataset_pretrain(waves_ibk)

# CPSC
waves_cpsc = waves_cinc(data_dir_cpsc)
# waves_shaoxing = downsample_waves(waves_shaoxing, 5000)
print(f"CPSC waves shape: {waves_cpsc.shape}")
logging.info(f"CPSC waves shape: {waves_cpsc.shape}")
dataset_cpsc = ECGDataset_pretrain(waves_cpsc)


# Shaoxing (Ningbo + Chapman)
waves_shaoxing = waves_shao(data_dir_shao)
# waves_shaoxing = downsample_waves(waves_shaoxing, 5000)
print(f"Shao waves shape: {waves_shaoxing.shape}")
logging.info(f"Shao waves shape: {waves_shaoxing.shape}")
dataset = ECGDataset_pretrain(waves_shaoxing)

# PTBXL
waves_ptbxl = waves_ptbxl_training(data_dir_ptbxl)
print(f"PTBXL waves shape: {waves_ptbxl.shape}")
logging.info(f"PTBXL waves shape: {waves_ptbxl.shape}")
dataset_ptbxl = ECGDataset_pretrain(waves_ptbxl)

# Code15
dataset_code15 = Code15Dataset(data_dir_code15)
print(f"Code15 waves shape: ({len(dataset_code15.file_indices)}, 8, 2500)")
logging.info(f"Code15 waves shape: ({len(dataset_code15.file_indices)}, 8, 2500)")

loading_time = time.time() - start_time
print(f"Data loading time: {loading_time:.2f}s")
### AChtung!!!!hier werden nur 8 leads betrachtet!!!!!
dataset = ConcatDataset([dataset, dataset_ibk, dataset_code15, waves_ptbxl, waves_cpsc])
