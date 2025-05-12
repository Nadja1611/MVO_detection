# Original work Copyright (c) Meta Platforms, Inc. and affiliates. <https://github.com/facebookresearch/mae>
# Modified work Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
from sklearn.metrics import roc_auc_score
from collections import defaultdict

import math
import sys
from typing import Dict, Iterable, Optional, Tuple
from boxplots import boxplot

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from gradcam import *

import torch
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score
import numpy as np

from sklearn.metrics import confusion_matrix


def custom_loss_with_prior(y_pred, y_true, lambda_prior=0.1):
    """
    BCE loss + logic-based regularization enforcing:
    - B can only occur if A occurs (i.e. B → A)
    - P(B | A) ≈ 0.6

    Args:
        y_pred: Tensor of shape (batch_size, num_classes) with probabilities
        y_true: Ground truth tensor of same shape
        A_index: Index of class A (parent class)
        B_index: Index of class B (dependent class)
        lambda_prior: Weight for the regularization term
    """
    y_prob = torch.sigmoid(y_pred)  # Convert logits to probabilities

    # Get probabilities for A and B
    p_A = y_prob[:, 0]
    p_B = y_prob[:, 1]

    # Enforce: B can only occur if A occurs => penalize p_B when p_A is low
    rule1 = (p_B * (1 - p_A)) ** 2  # Soft logic: B → A

    # Enforce: P(B | A) ≈ 0.6 when A is active
    # So if p_A is high, we want p_B to be ≈ 0.6
    rule2 = (p_A * p_B - 0.6 * p_A) ** 2

    # Combine prior losses
    # prior_loss = torch.mean( rule2)

    rule3 = (0.6 - (p_B / (p_A + 0.000001))) ** 2
    prior_loss = torch.mean(rule3)
    total_loss = lambda_prior * prior_loss
    return total_loss


def plot_confusion_for_class(y_true, y_pred, class_index, class_name, path):
    # Convert multilabel indicator to binary vectors for the specified class index
    y_true_binary = y_true[:, class_index]
    y_pred_binary = y_pred[:, class_index]

    cm = confusion_matrix(y_true_binary, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {class_name}")
    plt.savefig(path)
    plt.close()


def compute_f1_precision_recall(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1, recall, precision


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    config: Optional[dict] = None,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 50

    accum_iter = config["accum_iter"]
    max_norm = config["max_norm"]

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler


        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(samples)
            #  heatmap = compute_grad_cam(model.encoder_blocks.blocks[-1].attn)
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                targets = targets.to(dtype=outputs.dtype)
            # targets = torch.argmax(targets, dim=1)  # if targets are one-hot

            loss = criterion(
                outputs, targets
            )  # + custom_loss_with_prior(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((epoch + data_iter_step / len(data_loader)) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    metric_fn: torchmetrics.Metric,
    output_act: torch.nn.Module,
    target_dtype: torch.dtype = torch.long,
    use_amp: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"
    all_targets = []
    all_predictions = []
    all_predictions_probs, all_targets_probs = [], []
    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if samples.ndim == 4:  # batch_size, n_drops, n_channels, n_frames
                logits_list = []
                for i in range(samples.size(1)):
                    logits = model(samples[:, i])
                    logits_list.append(logits)
                logits_list = torch.stack(logits_list, dim=1)
                outputs_list = output_act(logits_list)
                logits = logits_list.mean(dim=1)
                outputs = outputs_list.mean(dim=1)
            else:
                logits = model(samples)
                outputs = output_act(logits)
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                targets = targets.to(dtype=outputs.dtype)
            # targets = torch.argmax(targets, dim=1)  # if targets are one-hot

            loss = criterion(logits, targets)

        outputs = misc.concat_all_gather(outputs)
        targets = misc.concat_all_gather(targets).to(dtype=target_dtype)
        metric_fn.update(outputs, targets)
        metric_logger.meters["loss"].update(loss.item(), n=samples.size(0))

        # Save all targets and predictions for confusion matrix and accuracy calculation
        all_targets.append(targets.cpu().numpy())
        threshold = 0.4

        # Convert continuous scores to binary labels using thresholding
        binary_labels = (outputs.cpu().numpy() >= threshold).astype(int)
        all_predictions.append(binary_labels)
        all_targets_probs.append(targets.cpu().numpy())
        all_predictions_probs.append(outputs.cpu().numpy())

    #     # Concatenate all targets and predictions

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    all_targets_probs = np.concatenate(all_targets_probs)
    all_predictions_probs = np.concatenate(all_predictions_probs)
    all_predictions_probs = all_predictions_probs[:, :]

    num_classes = all_targets.shape[1]
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    tp = np.zeros(num_classes)
    tn = np.zeros(num_classes)

    # For false negatives: track what else was predicted instead
    fn_distractors = [defaultdict(int) for _ in range(num_classes)]

    # Compute confusion matrix components for each class
    for i in range(num_classes):
        tp[i] = np.logical_and(all_predictions[:, i] == 1, all_targets[:, i] == 1).sum()
        fp[i] = np.logical_and(all_predictions[:, i] == 1, all_targets[:, i] == 0).sum()
        fn[i] = np.logical_and(all_predictions[:, i] == 0, all_targets[:, i] == 1).sum()
        tn[i] = np.logical_and(all_predictions[:, i] == 0, all_targets[:, i] == 0).sum()

    # Compute precision, recall, F1 score, and accuracy for each class
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    accuracy = np.divide(
        tp + tn,
        tp + tn + fp + fn,
        out=np.zeros_like(tp),
        where=(tp + tn + fp + fn) != 0,
    )

    # Analyze false negatives
    for sample_idx in range(all_targets.shape[0]):
        target = all_targets[sample_idx]
        pred = all_predictions[sample_idx]

        for class_idx in range(num_classes):
            if target[class_idx] == 1 and pred[class_idx] == 0:
                # This is a false negative for class_idx
                distractors = np.where(pred == 1)[0]  # other classes predicted
                for d in distractors:
                    if d != class_idx:
                        fn_distractors[class_idx][d] += 1

    print(all_targets_probs, flush=True)
    print("preds prob ", all_predictions_probs, flush=True)
    print("preds, ", all_predictions, flush=True)
    auc = roc_auc_score(all_targets_probs, all_predictions_probs)
    print("targets, ", all_targets, flush=True)

    metric_logger.synchronize_between_processes()
    valid_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    metrics = metric_fn.compute()
    if isinstance(metrics, dict):  # MetricCollection
        metrics = {k: v.item() for k, v in metrics.items()}
    else:
        metrics = {metric_fn.__class__.__name__: metrics.item()}
    metric_str = "  ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
    metric_str = f"{metric_str} loss: {metric_logger.loss.global_avg:.3f}"
    # print(f"* {metric_str}")
    metric_fn.reset()
    return (
        valid_stats,
        metrics,
        fp,
        fn,
        tp,
        all_targets,
        all_predictions,
        all_predictions_probs,
        precision,
        recall,
        f1,
        accuracy,
        fn_distractors,
    )
