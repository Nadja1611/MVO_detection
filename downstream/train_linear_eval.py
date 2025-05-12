import yaml
import argparse
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import logging  # Import the logging module
from datetime import datetime  # Import the datetime module
from pathlib import Path
import re
from torch.utils.tensorboard import SummaryWriter

# Get the current working directory
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from ecg_data_500Hz import *

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
from boxplots import boxplot
import re


def parse():
    parser = argparse.ArgumentParser("ECG downstream training")

    # parser.add_argument('--model_name',
    #                     default="ejepa_random",
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
        default="./output/linear_eval",
        type=str,
        metavar="PATH",
        help="output directory",
    )

    parser.add_argument(
        "--data_mvo",
        default="",  # "/mount/ecg/cpsc_2018/"
        type=str,
        help="dataset mvo directory",
    )
    parser.add_argument(
        "--dropout",
        default=0.0,  # "/mount/ecg/cpsc_2018/"
        type=float,
        help=0.0,
    )

    parser.add_argument("--dataset", default="ptbxl", type=str, help="dataset name")

    parser.add_argument(
        "--data_dir",
        default="/mount/ecg/ptb-xl-1.0.3/",  # "/mount/ecg/cpsc_2018/"
        type=str,
        help="dataset directory",
    )

    parser.add_argument(
        "--task", default="multilabel", type=str, help="downstream task"
    )

    parser.add_argument(
        "--data_percentage",
        default=1.0,
        type=float,
        help="data percentage (from 0 to 1) to use in few-shot learning",
    )

    # Use parse_known_args instead of parse_args
    args, unknown = parser.parse_known_args()

    with open(
        os.path.realpath(f"../configs/downstream/linear_eval/linear_eval_ejepa.yaml"),
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



        # Initialize TensorBoard writer
    log_dir = "/home/nadja/MVO_Project_multilabel/TB_JEPA/"
    hparams = {
        'data_percentage': config['data_percentage']
        }
    data_name = os.path.basename(config['data_mvo']) 
    print('name', data_name, flush = True)   
    run_name = f"Linear_Probing_ckpt_{ckpt_name}_data_{data_name}_data_percentage_{hparams['data_percentage']}"
    log_dir = Path("/home/nadja/MVO_Project_multilabel/TB_JEPA_LinProbing") / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    #dir_new = os.path.join(log_dir, "resnet")
    #os.makedirs(dir_new, exist_ok=True)
    result_dir = os.makedirs("/home/nadja/MVO_Project_multilabel/predictions_JEPA_linprobing/", exist_ok = True)
    result_subdir = Path("/home/nadja/MVO_Project_multilabel/predictions_JEPA_linprobing/") / run_name
    os.makedirs(str(result_subdir), exist_ok = True)


    writer.flush()



    # waves_train, waves_test, labels_train, labels_test = waves_from_config(config)
    list_of_folders = os.listdir(data_path)

    list_of_folders = [f for f in os.listdir(data_path) if f != ".DS_Store"]
    fold = 0
    All_probs, All_targets = [], []
    fold = 0
    is_all = None
    mvo_all = None
    ids_all = None
    is_all = None
    Fold_mean_f1, Fold_mean_auc, MEAN_AUC, MEAN_F1 = [], [], [], []
    for files in sorted(
        list_of_folders, key=lambda x: int(re.search(r"\d+", x).group())
    ):
        fold += 1
        waves_train = torch.load(os.path.join(data_path, files, "ecgs_train.pt"))
        waves_val = torch.load(os.path.join(data_path, files, "ecgs_val.pt"))
        waves_test = torch.load(os.path.join(data_path, files, "ecgs_test.pt"))
        labels_train = torch.load(os.path.join(data_path, files, "mvo_train.pt"))
        labels_val = torch.load(os.path.join(data_path, files, "mvo_val.pt"))
        labels_test = torch.load(os.path.join(data_path, files, "mvo_test.pt"))
        is_test = torch.load(os.path.join(data_path, files, "IS_size_test.pt"), weights_only=False)
        is_test = is_test.numpy() if isinstance(is_test, torch.Tensor) else is_test


        if is_all is None:
            is_all = is_test
        else:
            is_all = np.concatenate((is_all, is_test), axis=0)   

        mvo_test = torch.load(os.path.join(data_path, files, "mvo_size_test.pt"), weights_only=False)
        if mvo_all is None:
            mvo_all = mvo_test
        else:
            mvo_all = np.concatenate((mvo_all, mvo_test), axis=0)  

        ids_test = torch.load(os.path.join(data_path, files, "IDs_test.pt"), weights_only=False)
        if ids_all is None:
            ids_all = ids_test
        else:
            ids_all = np.concatenate((ids_all, ids_test), axis=0)  
        print(labels_test[:50], flush = True)
        print(ids_all[:50], flush = True)
        print(mvo_all[:50], flush = True)
        SEED = 42  # Or any fixed number
        np.random.seed(SEED)
        # Assume args.data_percentage exists and is a float in (0, 1]
        if config['data_percentage'] < 1.0:
            train_size = int(len(waves_train) * config['data_percentage'])
            val_size = int(len(waves_val) * config['data_percentage'])

            indices_train = np.random.permutation(len(waves_train))[:train_size]
            indices_val = np.random.permutation(len(waves_val))[:val_size]

            waves_train = waves_train[indices_train]
            labels_train = labels_train[indices_train]

            waves_val = waves_val[indices_val]
            labels_val = labels_val[indices_val]
        # volumes = torch.load(os.path.join(data_path, "mvo_vol_CNN_val.pt"))

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

        data_percentage = config["data_percentage"]
        n_trial = 1 if data_percentage == 1 else 3

        AUCs, F1s = [], []
        logging.info(f"Start training...")
        print(f"Start training...")
        for n in range(n_trial):
            num_samples = len(waves_train)


            num_workers = config["dataloader"]["num_workers"]



            
        waves_train = np.concatenate(
            (waves_train[:, :2, :], waves_train[:, 6:, :]), axis=1
        )
        waves_test = np.concatenate(
            (waves_test[:, :2, :], waves_test[:, 6:, :]), axis=1
        )

        print('shape after subsampling, ', waves_train.shape, flush = True)
            
        if config["task"] == "multilabel":
            _, n_labels = labels_train.shape
        else:
            n_labels = len(np.unique(labels_train))

        device = "cuda" if torch.cuda.is_available() else "cpu"

        logging.info(f"Loading encoder from {config['ckpt_dir']}...")
        print(f"Loading encoder from {config['ckpt_dir']}...")
        encoder, embed_dim = load_encoder(ckpt_dir=config["ckpt_dir"])
        encoder = encoder.to(device)

        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

        data_percentage = config["data_percentage"]
        n_trial = 1 if data_percentage == 1 else 3

        logging.info(f"Start training...")
        print(f"Start training...")
        for n in range(n_trial):
            num_samples = len(waves_train)
            if data_percentage < 1.0:
                num_desired_samples = round(num_samples * data_percentage)
                selected_indices = np.random.choice(
                    num_samples, num_desired_samples, replace=False
                )
                waves_train_selected = waves_train[selected_indices]
                labels_train_selected = labels_train[selected_indices]
            else:
                waves_train_selected = waves_train
                labels_train_selected = labels_train

            num_workers = config["dataloader"]["num_workers"]
            train_dataset = ECGDataset(waves_train_selected, labels_train_selected)
            test_dataset = ECGDataset(waves_test, labels_test)
            train_loader = DataLoader(
                train_dataset, batch_size=32, shuffle=True, num_workers=num_workers
            )
            test_loader = DataLoader(
                test_dataset, batch_size=32, shuffle=False, num_workers=num_workers
            )

            bs = config["dataloader"]["batch_size"]
            train_loader_linear = features_dataloader(
                encoder, train_loader, batch_size=bs, shuffle=True, device=device
            )
            test_loader_linear = features_dataloader(
                encoder, test_loader, batch_size=bs, shuffle=False, device=device
            )

            num_epochs = config["train"]["epochs"]
            lr = config["train"]["lr"]

            criterion = (
                nn.BCEWithLogitsLoss()
                if config["task"] == "multilabel"
                else nn.CrossEntropyLoss()
            )
            linear_model = LinearClassifier(embed_dim, n_labels).to(device)
            optimizer = optim.AdamW(linear_model.parameters(), lr=lr)
            iterations_per_epoch = len(train_loader_linear)
            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_epochs * iterations_per_epoch,
                cycle_mul=1,
                lr_min=lr * 0.1,
                cycle_decay=0.1,
                warmup_lr_init=lr * 0.1,
                warmup_t=10,
                cycle_limit=1,
                t_in_epochs=True,
            )

            if config["task"] == "multilabel":
                auc, f1, y_probs, y_true = train_multilabel(
                    num_epochs,
                    linear_model,
                    optimizer,
                    criterion,
                    scheduler,
                    train_loader_linear,
                    test_loader_linear,
                    device,
                    print_every=True,
                )
            else:
                auc, f1 = train_multiclass(
                    num_epochs,
                    linear_model,
                    criterion,
                    optimizer,
                    train_loader_linear,
                    test_loader_linear,
                    device,
                    scheduler=scheduler,
                    print_every=True,
                    amp=False,
                )
            All_probs.append(y_probs)
            All_targets.append(y_true)
            AUCs.append(auc)
            F1s.append(f1)
            logging.info(f"Trial {n + 1}: AUC: {auc:.3f}, F1: {f1:.3f}")
            print(f"Trial {n + 1}: AUC: {auc:.3f}, F1: {f1:.3f}")

        mean_auc = np.mean(AUCs)
        std_auc = np.std(AUCs)
        mean_f1 = np.mean(F1s)
        std_f1 = np.std(F1s)
        logging.info(
            f"Mean AUC: {mean_auc:.3f} +- {std_auc:.3f}, Mean F1: {mean_f1:.3f} +- {std_f1:.3f}"
        )
        print(
            f"Mean AUC: {mean_auc:.3f} +- {std_auc:.3f}, Mean F1: {mean_f1:.3f} +- {std_f1:.3f}"
        )

        Fold_mean_auc.append(mean_auc)
        Fold_mean_f1.append(mean_f1)
        MEAN_AUC.append(np.mean(Fold_mean_auc))
        MEAN_F1.append(np.mean(Fold_mean_f1))
        writer.add_scalar("mean_AUC_test", np.mean(MEAN_AUC), fold - 1)
        writer.add_scalar("mean_F1_test", np.mean(MEAN_F1), fold - 1)
    print(MEAN_AUC, flush = True)
    print("length of the list is ", len(MEAN_F1), flush=True)
    print("mean AUC: ", np.mean(MEAN_AUC), flush=True)
    print("mean F1: ", np.mean(MEAN_F1), flush=True)
    ### save the results into results folder corresponding to experiment
    torch.save({
        'probs':   [torch.tensor(p) for p in All_probs],
        'targets':  [torch.tensor(p) for p in All_targets],
        'ids': ids_all,
        'mvo_size': mvo_all,
        'is_size': is_all
    }, os.path.join(str(result_subdir), "results.pt") )     



    writer.flush()

    # Log epoch metrics

if __name__ == "__main__":
    config = parse()

    main(config)
