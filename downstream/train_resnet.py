import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import logging  # Import the logging module
from datetime import datetime  # Import the datetime module

# Get the current working directory
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from ecg_data_500Hz import ECGDataset
from torch.utils.data import DataLoader
from resnet152 import *
from linear_probe_utils import FinetuningClassifier
import torch.nn as nn
from augmentation import *

import util.misc as misc
from engine_training import evaluate, train_one_epoch, plot_confusion_for_class
from util.losses import build_loss_fn
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config
from util.perf_metrics import build_metric_fn, is_best_metric
from pathlib import Path
import re
from torch.utils.tensorboard import SummaryWriter



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
    parser.add_argument("--batch_size", type=float, default=128,
                        help="batch size")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout probability (default: 0.5)")
    parser.add_argument("--epochs_resnet", type=float, default=50,
                        help="number of epochs")
    parser.add_argument("--blr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--model_config", type=float, default=50,
                        help="model config")                        
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
        "--metrics_dir",
        default="/home/nadja/ECG_JEPA_Git/metrics/baseline",
        type=str,
        help="metrics directory",
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
        os.path.realpath(
            f"../configs/downstream/finetuning_scratch/fine_tuning_ejepa.yaml"
        ),
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
        f"Model_{config['model_config']}_log_{ckpt_name}_{config['task']}_{config['dataset']}_{current_time}.txt",
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

    pathology = config["pathology"]

    task = config["task"]
    config["metric"]["task"] = task
    # Initialize TensorBoard writer
    log_dir = "/home/nadja/MVO_Project_multilabel/TB_Resnet/"
    hparams = {
            'lr': config['blr'],
            'dropout': config['dropout'],
            'batch_size': config['batch_size'],
            'data_percentage': config['data_percentage']
        }

    data_name = os.path.basename(config['data_mvo'])    


    run_name = f"Model_{config['model_config']}_weights_{config['ckpt_dir']}_lr_{hparams['lr']}_dropout_{hparams['dropout']}_batchsize_{hparams['batch_size']}_data_{data_name}_data_percentage_{hparams['data_percentage']}"
    log_dir = Path("/home/nadja/MVO_Project_multilabel/TB_Resnet") / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    #dir_new = os.path.join(log_dir, "resnet")
    #os.makedirs(dir_new, exist_ok=True)

    writer.flush()
    # define data augmentation
    aug = {
        "rand_augment": {
            "use": True,
            "kwargs": {
                "op_names": [
                    "shift",
                    "cutout",
                    "drop",
                    "flip",
                    "erase",
                    "sine",
                    "partial_sine",
                    "partial_white_noise",
                ],
                "level": 10,
                "num_layers": 2,
                "prob": 0.5,
            },
        },
        "train_transforms": [
            {"highpass_filter": {"fs": 250, "cutoff": 0.67}},
            {"lowpass_filter": {"fs": 250, "cutoff": 40}},
        ],
        "eval_transforms": [
            {"highpass_filter": {"fs": 250, "cutoff": 0.67}},
            {"lowpass_filter": {"fs": 250, "cutoff": 40}},
        ],
    }

    # # st_mem model requires shorter input length
    # if config['model_name'] == 'st_mem':
    #     aug['train_transforms'].append({'random_crop': {'crop_length': 2250}})
    #     aug['eval_transforms'].append({'random_crop': {'crop_length': 2250}})

    train_transforms = get_transforms_from_config(aug["train_transforms"])
    randaug_config = aug.get("rand_augment", {})
    use_randaug = randaug_config.get("use", False)
    if use_randaug:
        randaug_kwargs = randaug_config.get("kwargs", {})
        train_transforms.append(get_rand_augment_from_config(randaug_kwargs))

    test_transforms = get_transforms_from_config(aug["eval_transforms"])

    train_transforms = Compose(train_transforms + [ToTensor()])
    test_transforms = Compose(test_transforms + [ToTensor()])

    # load dataset
    logging.info(f"Loading {config['dataset']} dataset...")
    print(f"Loading {config['dataset']} dataset...")
    ## specify data for 5 fold crossvalidation
    data_path = config["data_mvo"]
    # waves_train, waves_test, labels_train, labels_test = waves_from_config(config)
    acc_all, auc_all, f1_all, f1_ours_all, auc_ours_all = [], [], [], [], []
    list_of_folders = os.listdir(data_path)
    list_of_folders = [f for f in os.listdir(data_path) if f != ".DS_Store"]
    ACC, AUC = [], []
    fold = 0
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
        ### take only eight leads
        waves_train = np.concatenate(
            (waves_train[:, :2, :], waves_train[:, 6:, :]), axis=1
        )
        waves_val = np.concatenate((waves_val[:, :2, :], waves_val[:, 6:, :]), axis=1)
        waves_test = np.concatenate(
            (waves_test[:, :2, :], waves_test[:, 6:, :]), axis=1
        )

        #     print("trainig data, ", labels_train.shape, flush=True)
        if task == "multilabel":
            _, n_labels = labels_train.shape
            config["metric"]["num_labels"] = n_labels
            #       print(f"Number of labels: {n_labels}")
            n = n_labels
        else:
            n_classes = len(np.unique(labels_train))
            config["metric"]["num_classes"] = n_classes
            #    print(f"Number of classes: {n_classes}")
            n = n_classes

        train_dataset = ECGDataset(waves_train, labels_train, train_transforms)
        val_dataset = ECGDataset(waves_val, labels_val, test_transforms)
        test_dataset = ECGDataset(waves_test, labels_test, test_transforms)

        data_loader_train = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
        )
        data_loader_val = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
        )
        data_loader_test = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        #logging.info(f"Loading encoder from {config['ckpt_dir']}...")
        # print(f"Loading encoder from {config['ckpt_dir']}...")
        dropout = config['dropout']
        if config['model_config'] == 152:
            model = ResNet152_1D(num_classes=3, input_channels=8, dropout=dropout).to(device)
        if config['model_config'] == 50:
            model = ResNet50_1D(num_classes=3, input_channels=8, dropout=dropout).to(device)
        print('ckpt', config['ckpt_dir'], flush = True)
        checkpoint = torch.load(config['ckpt_dir'], map_location="cpu")
        state_dict = checkpoint["encoder"]

        # Remove the mismatched keys (fc layer)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}

        # Load partial weights
        model.load_state_dict(state_dict, strict=False)
        lr = config["blr"] 
        # print("learning rate ", str(lr), flush=True)
       # lr = config['lr']
        print("learning rate ", str(lr), flush=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=config["train"]["weight_decay"],
        )

        loss_scaler = NativeScaler()
        criterion = (
            nn.BCEWithLogitsLoss()
            if config["task"] == "multilabel"
            else nn.CrossEntropyLoss()
        )
        output_act = (
            nn.Sigmoid() if config["task"] == "multilabel" else nn.Softmax(dim=-1)
        )
        best_loss = float("inf")

        metric_fn, best_metrics = build_metric_fn(config["metric"])
        metric_fn.to(device)

        import time

        # Start training
        start_time = time.time()
        use_amp = True

        output_dir = config["output_dir"]
        log_writer = None
        ACC, AUC, F1 = [], [], []
        training_loss, validation_loss = [], []

        best_val_loss = float("inf")
        best_model = None
        for epoch in range(config["epochs_resnet"]):
            train_stats = train_one_epoch(
                model,
                criterion,
                data_loader_train,
                optimizer,
                device,
                epoch,
                loss_scaler,
                log_writer,
                config["train"],
                use_amp=use_amp,
            )

            (
                valid_stats,
                metrics,
                fn,
                fp,
                tp,
                y_true,
                y_pred,
                y_probs,
                precision,
                recall,
                f1_scores,
                accuracy,
                fn_distractors,
            ) = evaluate(
                model,
                criterion,
                data_loader_val,
                device,
                metric_fn,
                output_act,
                use_amp=use_amp,
            )

            curr_loss = valid_stats["loss"]
            # Log to TensorBoard
            #  print(f"Logging loss for epoch {epoch} for file {files}", flush=True)

            writer.add_scalar(f"Loss_train_{files}", train_stats["loss"], epoch)
            writer.add_scalar(f"Loss_valid_{files}", valid_stats["loss"], epoch)
            writer.add_scalar(f"AUC_test_{files}", metrics["MultilabelAUROC"], epoch)
            writer.add_scalar(
                f"F1-score_{files}/test", metrics["MultilabelF1Score"], epoch
            )



            writer.flush()
            if curr_loss < best_val_loss:
                print("better val loss found")
                best_val_loss = valid_stats["loss"]
                best_model = model.state_dict()

                # Log epoch metrics
            logging.info(f"Epoch: {epoch}")
            logging.info(f"Training Loss: {train_stats['loss']:.4f}")
            logging.info(f"Validation Loss: {curr_loss:.4f}")
            training_loss.append(train_stats["loss"])
            validation_loss.append(curr_loss)
            #  print(f"Epoch: {epoch}")
            # print(f"Training Loss: {train_stats['loss']:.4f}")
            # print(f"Validation Loss: {curr_loss:.4f}")
            for metric_name, metric_class in metric_fn.items():
                curr_metric = metrics[metric_name]
                logging.info(f"{metric_name}: {curr_metric:.3f}")
                print(f"{metric_name}: {curr_metric:.3f}")
                if is_best_metric(metric_class, best_metrics[metric_name], curr_metric):
                    best_metrics[metric_name] = curr_metric
                logging.info(f"Best {metric_name}: {best_metrics[metric_name]:.3f}")
                print(f"Best {metric_name}: {best_metrics[metric_name]:.3f}")

                """ save metrics into lists"""
                for item in metrics:
                    print(item, flush=True)
                ACC.append(metrics["MultilabelAccuracy"])
                AUC.append(metrics["MultilabelAUROC"])
                F1.append(metrics["MultilabelF1Score"])

                model.to("cpu")
                torch.save(
                    {
                        "encoder": model.state_dict(),
                        "epoch": epoch,
                    },
                    f"{config['output_dir']}/train{epoch + 1}.pth",
                )
                model.to("cuda")
                print(
                    "========================================================================================"
                )

            # Extract filename without path
            ckpt_filename = Path(config["ckpt_dir"]).stem  # Gets 'epoch10'

            # Extract the number from the filename
            match = re.search(r"(\d+)$", ckpt_filename)  # Finds trailing digits
            ckpt_number = (
                match.group(1) if match else ckpt_filename
            )  # Use the number if found

            # Save metrics

        if best_model is not None:
            model.load_state_dict(best_model)
            metric_fn_test, best_metrics_test = build_metric_fn(config["metric"])
            metric_fn_test.to(device)
            (
                test_stats,
                test_metrics,
                fn,
                fp,
                tp,
                y_true,
                y_pred,
                y_probs,
                precision,
                recall,
                f1_scores,
                accuracy,
                fn_distractors,
            ) = evaluate(
                model,
                criterion,
                data_loader_test,
                device,
                metric_fn_test,
                output_act,
                use_amp=True,
            )

            labels = ["MVO+IMH-", "MVO+IMH+", "MVO-IMH-"]
            x = np.arange(len(labels))  # positions for each class label
            width = 0.25  # width for each bar

            fig, ax = plt.subplots()

            # Position each bar with unique offsets:
            rects1 = ax.bar(x - width, fp, width, label="False Positives")
            rects2 = ax.bar(x, fn, width, label="False Negatives")
            rects3 = ax.bar(x + width, tp, width, label="True Positives")

            ax.set_ylabel("Count")
            ax.set_title("Error Counts per Class")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45)
            ax.legend()

            plt.tight_layout()

            plt.savefig(
                f"{config['metrics_dir']}/metrics2_fold_{fold}_{ckpt_number}.png"
            )
            plt.close()
            # Plot confusion matrices for each class

            for metric_name, metric_class in metric_fn_test.items():
                curr_metric = test_metrics[metric_name]
                logging.info(f"{metric_name}: {curr_metric:.3f}")
                print(f"Test {metric_name}: {curr_metric:.3f}")
                if is_best_metric(metric_class, best_metrics[metric_name], curr_metric):
                    best_metrics_test[metric_name] = curr_metric
                logging.info(
                    f"Best Test {metric_name}: {best_metrics_test[metric_name]:.3f}"
                )
                print(f"Best Test {metric_name}: {best_metrics_test[metric_name]:.3f}")
            # print calculated numbers
            print("===================== metrics testing ============================")

            for metric_name, metric_class in metric_fn_test.items():
                curr_metric = test_metrics[metric_name]
                logging.info(f"{metric_name}: {curr_metric:.3f}")
                print(f"Test {metric_name}: {curr_metric:.3f}")
                if is_best_metric(metric_class, best_metrics[metric_name], curr_metric):
                    best_metrics_test[metric_name] = curr_metric
                logging.info(
                    f"Best Test {metric_name}: {best_metrics_test[metric_name]:.3f}"
                )
                print(f"Best Test {metric_name}: {best_metrics_test[metric_name]:.3f}")
            # print calculated numbers
            print("===================== metrics testing ============================")

            acc_all.append(test_metrics["MultilabelAccuracy"])
            auc_all.append(test_metrics["MultilabelAUROC"])
            f1_all.append(test_metrics["MultilabelF1Score"])

            print("the length of the acc list is, ", len(auc_all))
            print("Mean test ACC:", np.mean(acc_all))
            print("Mean test AUC:", np.mean(auc_all))
            print("Mean test F1:", np.mean(f1_all))

            f"outputs saved to: {config['metrics_dir']}/metrics_{ckpt_number}.npz"

            # make barplots
            plt.figure(figsize=(8, 6))
            labels = ["F1 Ours", "AUC Ours", "Accuracy"]
            means = [np.mean(f1_all), np.mean(auc_all), np.mean(acc_all)]
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

            bars = plt.bar(labels, means, color=colors, edgecolor="black")

            # Add value labels on top
            for bar in bars:
                yval = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval + 0.01,
                    f"{yval:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )

            # Aesthetic improvements
            plt.title("Mean Test Metrics", fontsize=16)
            plt.ylim(0, 1.1)
            plt.ylabel("Score", fontsize=14)
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{config['metrics_dir']}/metrics_{ckpt_number}.png")

            np.savez_compressed(
                f"{config['metrics_dir']}/metrics_{ckpt_number}.npz",
                acc=np.mean(acc_all),
                f1=np.mean(f1_all),
                auc=np.mean(auc_all),
            )

            writer.add_scalar("mean_ACC_test", np.mean(acc_all), fold - 1)
            writer.add_scalar("mean_AUC_test", np.mean(auc_all), fold - 1)
            writer.add_scalar("mean_F1_test", np.mean(f1_all), fold - 1)

            writer.flush()

            # Log epoch metrics
            logging.info(f"Epoch: {epoch}")
            logging.info(f"Training Loss: {train_stats['loss']:.4f}")
            logging.info(f"Validation Loss: {curr_loss:.4f}")
            training_loss.append(train_stats["loss"])
            validation_loss.append(curr_loss)

            """ save metrics into lists"""


            model.to("cpu")
            torch.save(
                {
                    "encoder": model.state_dict(),
                    "epoch": epoch,
                },
                f"{config['output_dir']}/train{epoch + 1}.pth",
            )
            model.to("cuda")
            print(
                "========================================================================================"
            )

        # Extract filename without path
        
        ckpt_filename = Path(config["ckpt_dir"]).stem  # Gets 'epoch10'

        # Extract the number from the filename
        match = re.search(r"(\d+)$", ckpt_filename)  # Finds trailing digits
        ckpt_number = (
            match.group(1) if match else ckpt_filename
        )  # Use the number if found
        print("ckpt number,", ckpt_number, flush=True)

        # Save metrics

    writer.close()


if __name__ == "__main__":
    config = parse()

    # pretrained_ckpt_dir = {
    #     'ejepa_random': f"../weights/random_epoch100.pth",
    #     'ejepa_multiblock': f"../weights/multiblock_epoch100.pth",
    #     # 'cmsc': "../weights/shao+code15/CMSC/epoch300.pth",
    #     # 'cpc': "../weights/shao+code15/cpc/base_epoch100.pth",
    #     # 'simclr': "../weights/shao+code15/SimCLR/epoch300.pth",
    #     # 'st_mem': "../weights/shao+code15/st_mem/st_mem_vit_base.pth",
    # }

    # # pretrained_ckpt_dir['mvt_larger_larger'] = f"../weights/shao+code15/block_masking/jepa_v4_20240720_215455_(0.175, 0.225)/epoch{100}.pth"

    # config['ckpt_dir'] = pretrained_ckpt_dir[config['model_name']]

    main(config)
