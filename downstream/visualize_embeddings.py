import yaml
import argparse
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import logging  # Import the logging module
from datetime import datetime  # Import the datetime module
import torch

import seaborn as sns
import matplotlib.pyplot as plt

# Get the current working directory
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from timm.scheduler import CosineLRScheduler
from ecg_data_500Hz import ECGDataset
from torch.utils.data import DataLoader
from models import load_encoder
from linear_probe_utils import (
    features_dataloader,
    train_multilabel,
    train_multiclass,
    LinearClassifier,
    SimpleLinearRegression,
)
import torch.optim as optim
import torch.nn as nn

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from ecg_data_500Hz_ptbxl_ours import *


def parse():
    parser = argparse.ArgumentParser("ECG downstream training")

    # parser.add_argument('--model_name',
    #                     default="mvt_larger_larger",
    #                     type=str,
    #                     help='resume from checkpoint')

    parser.add_argument(
        "--ckpt_dir",
        default="../weights/multiblock_epoch100.pth",
        type=str,
        metavar="PATH",
        help="pretrained encoder checkpoint",
    )

    parser.add_argument(
        "--output_dir",
        default="./output/finetuning",
        type=str,
        metavar="PATH",
        help="output directory",
    )

    parser.add_argument("--dataset", default="ptbxl", type=str, help="dataset name")

    parser.add_argument(
        "--data_dir",
        default="/mount/ecg/ptb-xl-1.0.3/",
        type=str,
        help="dataset directory",
    )

    parser.add_argument(
        "--task", default="multiclass", type=str, help="downstream task"
    )

    parser.add_argument(
        "--pathology", default="mvo", type=str, help="medical task to be solved"
    )

    parser.add_argument(
        "--data_percentage",
        default=1.0,
        type=float,
        help="data percentage (from 0 to 1) to use in few-shot learning",
    )

    parser.add_argument(
        "--data_mvo",
        default="",  # "/mount/ecg/cpsc_2018/"
        type=str,
        help="dataset mvo directory",
    )

    # Use parse_known_args instead of parse_args
    args, unknown = parser.parse_known_args()

    with open(
        os.path.realpath(f"../configs/downstream/finetuning/fine_tuning_ejepa.yaml"),
        "r",
    ) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in vars(args).items():
        if v:
            config[k] = v

    return config


def main(config):
    os.makedirs(config["output_dir"], exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create log filename with current time
    ckpt_name = os.path.splitext(os.path.basename(config["ckpt_dir"]))[0]
    log_filename = os.path.join(
        config["output_dir"],
        f"log_{ckpt_name}_{config['task']}_{config['dataset']}_{current_time}.txt",
    )

    # Configure logging
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Log the config dictionary
    logging.info("Configuration:")
    logging.info(yaml.dump(config, default_flow_style=False))

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    data_path = config["data_mvo"]
    logging.info(f"Loading {config['dataset']} dataset...")
    print(f"Loading {config['dataset']} dataset...")
    waves_train = torch.load(data_path + "/ecgs_train.pt")
    waves_val = torch.load(data_path + "/ecgs_val.pt")
    waves_test = torch.load(data_path + "/ecgs_test.pt")
    is_test = torch.load(os.path.join(data_path, "IS_size_test.pt"), weights_only=False)
    mvo_test = torch.load(os.path.join(data_path, "mvo_size_test.pt"), weights_only=False)
    ids_test = torch.load(os.path.join(data_path, "IDs_test.pt"), weights_only=False)
    is_val = torch.load(os.path.join(data_path,  "IS_size_val.pt"), weights_only=False)
    mvo_val = torch.load(os.path.join(data_path,  "mvo_size_val.pt"), weights_only=False)
    ids_val = torch.load(os.path.join(data_path,  "IDs_val.pt"), weights_only=False)
    is_train = torch.load(os.path.join(data_path,  "IS_size_train.pt"), weights_only=False)
    mvo_train = torch.load(os.path.join(data_path,  "mvo_size_train.pt"), weights_only=False)
    ids_train = torch.load(os.path.join(data_path,  "IDs_train.pt"), weights_only=False)

    if config["task"] == "multilabel" or config["task"] == "multiclass":
        labels_train = torch.load(data_path + "/mvo_train.pt")
        labels_val = torch.load(data_path + "/mvo_val.pt")
        labels_test = torch.load(data_path + "/mvo_test.pt")
    print(ids_train[-5:-1], flush = True)
    print(len(ids_train), len(mvo_train),len(labels_train), flush=True)
    print(mvo_train[-5:-1], flush = True)
    print(labels_train[-5:-1],flush = True)
    waves_train = np.concatenate((waves_train[:, :2, :], waves_train[:, 6:, :]), axis=1)
    waves_val = np.concatenate((waves_val[:, :2, :], waves_val[:, 6:, :]), axis=1)
    waves_test = np.concatenate((waves_test[:, :2, :], waves_test[:, 6:, :]), axis=1)

    print(labels_train[:4])
    # waves_train, waves_test, labels_train, labels_test = waves_from_config(config,reduced_lead=True)

    if config["task"] == "multilabel":
        _, n_labels = labels_train.shape
    elif config["task"] == "multiclass":
        n_labels = len(np.unique(labels_train))
    else:
        """ for the regression case, the output should be one number"""
        n_labels = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f"Loading encoder from {config['ckpt_dir']}...")
    print(f"Loading encoder from {config['ckpt_dir']}...")
    encoder, embed_dim = load_encoder(ckpt_dir=config["ckpt_dir"])
    encoder = encoder.to(device)

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    waves = np.concatenate((waves_train, waves_test, waves_val), 0)
    labels = np.concatenate((labels_train, labels_test, labels_val), 0)
    ids = np.concatenate((ids_train, ids_test, ids_val), 0)
    is_size = np.concatenate((is_train, is_test, is_val), 0)
    mvo_size = np.concatenate((mvo_train, mvo_test, mvo_val), 0)

    print("shapes ", waves.shape, labels.shape)
    dataset_with_labels = ECGDataset(waves, labels)

    # PTBXL
    waves_train, waves_test, labels_train, labels_test = waves_ptbxl(
        "/scratch/nadja/PTBXL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
    )
    print(f"PTBXL waves shape: {waves_train.shape}")
    logging.info(f"PTBXL waves shape: {waves_train.shape}")
    dataset_ptbxl = ECGDataset(np.concatenate((waves_train, waves_test),0), np.concatenate((labels_train, labels_test),0))

    all_labels = []
    all_labels_ptbxl = []
    all_features = []
    # Create a dataloader local
    dataloader_with_labels = torch.utils.data.DataLoader(
        dataset_with_labels, batch_size=128, shuffle=False, num_workers=2
    )

    # Create a dataloader ptbxl
    dataloader_with_labels_ptbxl = torch.utils.data.DataLoader(
        dataset_ptbxl, batch_size=128, shuffle=False, num_workers=2
    )
    with torch.no_grad():
        for wave, target in dataloader_with_labels:
            print(wave.shape, target.shape)
            repr = encoder.representation(wave.to(device))  # (bs, 8, 2500) -> (bs, dim)
            print("rep ", repr.shape)
            all_features.append(repr.cpu())
            all_labels.append(target)

    lab_ours = torch.cat(all_labels)
    features_ours = torch.cat(all_features)
    all_ids = ids
    all_is = is_size
    all_mvo = mvo_size
    
    LAB_ours = np.argmax(lab_ours, axis=1)
    EMB_ours = features_ours
    # Choose a method for dimensionality reduction
    method = "tsne"  # Options: "pca", "tsne", "umap"

    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=10, random_state=42)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)

    # Reduce dimensions
    EMB_2D = reducer.fit_transform(EMB_ours.detach().cpu())

    # Plot
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("pastel", 2)  # 7-class pastel color palette
    sns.scatterplot(
        x=EMB_2D[:, 0],
        y=EMB_2D[:, 1],
        hue=LAB_ours,
        palette=palette,
        alpha=0.8,
        edgecolor="w",  # white edge for better separation
        s=100,  # increase point size for better visibility
    )

    # Title and Labels
    plt.title(f"2D Visualization using {method.upper()}", fontsize=18, weight="bold")
    plt.xlabel("Dim 1", fontsize=14)
    plt.ylabel("Dim 2", fontsize=14)

    # Customize legend
    plt.legend(title="Class", loc="best", fontsize=12, title_fontsize=14)

    # Add grid for better readability
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)

    # Save the figure
    plt.savefig(
        "/home/nadja/MVO_Project_multilabel/embeddings/embedding_space_tsne_ours_"
        + config["pathology"]
        + ".png",
        dpi=300,  # High resolution for clarity
    )

    with torch.no_grad():
        for wave, target in dataloader_with_labels_ptbxl:
            print(wave.shape, target.shape, flush = True)
            repr = encoder.representation(wave.to(device))  # (bs, 8, 2500) -> (bs, dim)
            print("rep ", repr.shape)
            all_features.append(repr.cpu())
            all_labels_ptbxl.append(target)

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    all_labels_ptbxl = torch.cat(all_labels_ptbxl)

    #### save the embeddings
    torch.save(
        all_features,
        f"/home/nadja/MVO_Project_multilabel/embeddings/embeddings_ours_{ckpt_name}"
        + config["pathology"]
        + ".pt",
    )
    torch.save(
        all_labels,
        f"/home/nadja/MVO_Project_multilabel/embeddings/labels_ours_{ckpt_name}"
        + config["pathology"]
        + ".pt",
    )
    print('ids ', all_ids, flush = True)
    torch.save(
        all_ids,
        f"/home/nadja/MVO_Project_multilabel/embeddings/ids_ours_{ckpt_name}"
        + config["pathology"]
        + ".pt",
    )
    torch.save(
        all_is,
        f"/home/nadja/MVO_Project_multilabel/embeddings/is_ours_{ckpt_name}"
        + config["pathology"]
        + ".pt",
    )
    torch.save(
        all_mvo,
        f"/home/nadja/MVO_Project_multilabel/embeddings/mvo_ours_{ckpt_name}"
        + config["pathology"]
        + ".pt",
    )
    torch.save(
        all_labels_ptbxl,
        f"/home/nadja/MVO_Project_multilabel/embeddings/labels_ptbxl"
        + config["pathology"]
        + ".pt",
    )

    print("waves mean: ", wave[0][0].mean())
    print(f"Representation shape: {repr.shape}")
    LAB = torch.zeros((all_features.shape[0], 8))
    LAB[: len(dataset_with_labels), :3] = all_labels[: len(dataset_with_labels)]
    LAB[len(dataset_with_labels) :, 3:] = all_labels_ptbxl[:]

    LAB = np.argmax(LAB, axis=1)
    print(LAB)
    EMB = all_features
    print("emb " + str(EMB.shape))
    # Choose a method for dimensionality reduction
    method = "tsne"  # Options: "pca", "tsne", "umap"

    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=10, random_state=42)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)

    # Reduce dimensions
    EMB_2D = reducer.fit_transform(EMB.detach().cpu())

    # Plot
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("pastel", 7)  # 7-class pastel color palette
    sns.scatterplot(
        x=EMB_2D[:, 0],
        y=EMB_2D[:, 1],
        hue=LAB,
        palette=palette,
        alpha=0.8,
        edgecolor="w",  # white edge for better separation
        s=100,  # increase point size for better visibility
    )

    # Title and Labels
    plt.title(f"2D Visualization using {method.upper()}", fontsize=18, weight="bold")
    plt.xlabel("Dim 1", fontsize=14)
    plt.ylabel("Dim 2", fontsize=14)

    # Customize legend
    plt.legend(title="Class", loc="best", fontsize=12, title_fontsize=14)

    # Add grid for better readability
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)

    # Save the figure
    plt.savefig(
        "/home/nadja/MVO_Project_multilabel/embeddings/embedding_space_tsne_"
        + config["pathology"]
        + ".png",
        dpi=300,  # High resolution for clarity
    )


if __name__ == "__main__":
    config = parse()

    main(config)
