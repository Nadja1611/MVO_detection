import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, roc_auc_score
from scipy.special import expit, softmax
from tqdm import tqdm


# Precompute the features from the encoder and store them
def precompute_features(encoder, loader, device):
    encoder.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        print("Precomputing features...")
        for wave, label in tqdm(loader):
            bs, _, _ = wave.shape
            wave = wave.to(device)
            feature = encoder.representation(wave)  # (bs,c*50,384)
            all_features.append(feature.cpu())
            all_labels.append(label)

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)

    return all_features, all_labels


def features_dataloader(encoder, loader, batch_size=32, shuffle=True, device="cpu"):
    features, labels = precompute_features(encoder, loader, device=device)
    dataset = torch.utils.data.TensorDataset(features, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
    )

    return dataloader


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, apply_bn=False):
        super(LinearClassifier, self).__init__()
        self.apply_bn = apply_bn
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=1e-6)
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        if self.apply_bn:
            x = self.bn(x)

        x = self.fc(x)
        return x


#class FinetuningClassifier(nn.Module):
#    def __init__(self, encoder, encoder_dim, num_labels, device="cpu", apply_bn=False):
#        super(FinetuningClassifier, self).__init__()
#        self.encoder = encoder
 #       self.encoder_dim = encoder_dim
 #       # self.bn = nn.BatchNorm1d(encoder_dim, affine=False, eps=1e-6) # this outputs nan value in mixed precision
 #       self.fc = LinearClassifier(encoder_dim, num_labels, apply_bn=apply_bn)

  #  def forward(self, x):
   #     bs, _, _ = x.shape
    #    x = self.encoder.representation(x)
    #    # x = self.bn(x)
    #    x = self.fc(x)
    #    return x

class FinetuningClassifier(nn.Module):
    def __init__(self, encoder, encoder_dim, num_labels, device="cpu", apply_bn=False, dropout_rate=0.5):
        """
        Args:
            encoder (nn.Module): Pretrained model encoder (e.g., a CNN, Transformer, etc.)
            encoder_dim (int): Dimension of the encoder output (i.e., the feature size).
            num_labels (int): Number of output labels for the classification task.
            device (str): Device to use, default is "cpu".
            apply_bn (bool): Whether to apply batch normalization or not.
            dropout_rate (float): Dropout rate to apply after encoder output, default is 0.5.
        """
        super(FinetuningClassifier, self).__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.fc = LinearClassifier(encoder_dim, num_labels, apply_bn=apply_bn)
        
        # Dropout layer with specified rate
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout rate specified during initialization
        
    def forward(self, x):
        """
        Forward pass of the model. It passes the input through the encoder and then through the classifier.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output logits for classification.
        """
        bs, _, _ = x.shape
        
        # Get the encoder's feature representation
        x = self.encoder.representation(x)
        
        # Apply dropout after the encoder output
        x = self.dropout(x)
        
        # Classifier output
        x = self.fc(x)
        
        return x


class SimpleLinearRegression(nn.Module):
    def __init__(self, input_dim=384):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.linear(x)
        return x.view(-1)


def train_multilabel(
    num_epochs,
    linear_model,
    optimizer,
    criterion,
    scheduler,
    train_loader_linear,
    test_loader_linear,
    device,
    print_every=True,
):
    iterations_per_epoch = len(train_loader_linear)
    max_auc = 0.0

    for epoch in range(num_epochs):
        linear_model.train()

        for minibatch, (batch_features, batch_labels) in enumerate(train_loader_linear):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = linear_model(batch_features)
            loss = criterion(outputs, batch_labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch * iterations_per_epoch + minibatch)

        all_labels = []
        all_outputs = []

        with torch.no_grad():
            linear_model.eval()
            for batch_features, batch_labels in test_loader_linear:
                batch_features = batch_features.to(device)
                outputs = linear_model(batch_features)
                all_labels.append(batch_labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        all_labels = np.vstack(all_labels)
        all_outputs = np.vstack(all_outputs)
        all_probs = expit(all_outputs)

        auc_scores = [
            roc_auc_score(all_labels[:, i], all_outputs[:, i])
            if np.unique(all_labels[:, i]).size > 1
            else float("nan")
            for i in range(all_labels.shape[1])
        ]
        avg_auc = np.nanmean(auc_scores)

        # Optional: Print per-class AUC scores
        for idx, auc in enumerate(auc_scores):
            print(f"Class {idx}: AUC = {auc:.4f}", flush=True)

        if avg_auc > max_auc:
            max_auc = avg_auc

        # Compute F1 score
        predicted_labels = (all_probs >= 0.5).astype(int)

        # F1 score for each class (1 vs rest)
        per_class_f1 = f1_score(all_labels, predicted_labels, average=None)

        macro_f1 = f1_score(all_labels, predicted_labels, average="micro")

        if print_every:
            print(
                f"Epoch({epoch}) AUC: {avg_auc:.3f}({max_auc:.3f}), Macro F1: {macro_f1:.3f}, Per class F1: {per_class_f1.mean():.3f}"
            )

    return avg_auc, macro_f1, all_probs, all_labels


def train_multiclass(
    num_epochs,
    model,
    criterion,
    optimizer,
    train_loader_linear,
    test_loader_linear,
    device,
    scheduler=None,
    print_every=False,
    amp=False,
):
    iterations_per_epoch = len(train_loader_linear)
    max_auc = 0.0
    macro_f1 = 0.0

    if amp:
        scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()

        for minibatch, (batch_features, batch_labels) in enumerate(train_loader_linear):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()

            # Mixed precision training
            if amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)

                scaler.scale(loss).backward()  # Scale the loss and backpropagate
                scaler.step(optimizer)  # Step the optimizer
                scaler.update()  # Update the scaler

            else:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step(epoch * iterations_per_epoch + minibatch)

        all_labels = []
        all_outputs = []

        with torch.no_grad():
            model.eval()
            for minibatch, (batch_features, batch_labels) in enumerate(
                test_loader_linear
            ):
                batch_features = batch_features.to(device)

                if amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_features)
                else:
                    outputs = model(batch_features)

                all_labels.append(batch_labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        all_labels = np.concatenate(all_labels)
        all_outputs = np.vstack(all_outputs)
        print(all_labels[:10], flush=True)
        print(all_outputs[:10], flush=True)

        # when using amp, all_outputs should be changed to float32
        if amp:
            all_outputs = np.float32(all_outputs)

        all_probs = softmax(all_outputs, axis=1)
        print(all_probs[:10], flush=True)

        # Compute ROC AUC score
        print("labels, ", all_labels.shape, flush=True)
        print("probs, ", all_probs.shape, flush=True)

        avg_auc = roc_auc_score(all_labels, all_probs[:, 1])
        if avg_auc > max_auc:
            max_auc = avg_auc

        # Compute F1 score
        predicted_labels = np.argmax(all_outputs, axis=1)
        macro_f1 = f1_score(all_labels, predicted_labels, average="binary")
        print("GT", all_labels, flush=True)
        print("pred, ", predicted_labels, flush=True)
        tp = np.sum(predicted_labels * all_labels)
        fn = np.sum((1 - predicted_labels) * all_labels)
        fp = np.sum(predicted_labels * (1 - all_labels))
        tn = np.sum((1 - predicted_labels) * (1 - all_labels))
        if print_every:
            print(
                f"Epoch({epoch}) AUC: {avg_auc:.3f}({max_auc:.3f}), Binary F1: {macro_f1:.3f}, FP: {fp:.3f}, TP: {tp:.3f}, TN: {tn:.3f}, FN: {fn:.3f}"
            )

    print(
        f"Epoch({epoch}) AUC: {avg_auc:.3f}({max_auc:.3f}), Macro F1: {macro_f1:.3f}, FP: {fp:.3f}, TP: {tp:.3f}, TN: {tn:.3f}, FN: {fn:.3f}"
    )
    return avg_auc, macro_f1


# from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import accuracy_score, confusion_matrix

# def train_multiclass(num_epochs, model, criterion, optimizer, train_loader_linear, test_loader_linear, device, scheduler=None, print_every=False, amp=False):
#     iterations_per_epoch = len(train_loader_linear)
#     max_auc = 0.0
#     macro_f1 = 0.0

#     label_map = {5: 'BBB', 11: 'PVC', 13: 'Paced', 4: 'AVB4', 0: 'AFIB', 14: 'SA',
#                  6: 'BBB_AFIB', 12: 'PVC2', 3: 'AVB2', 1: 'AFL', 8: 'NSR', 15: 'VT',
#                  2: 'AVB1', 9: 'PAC', 10: 'PAC2', 7: 'NQT'}

#     if amp:
#         scaler = GradScaler()

#     for epoch in range(num_epochs):
#         model.train()

#         for minibatch, (batch_features, batch_labels) in enumerate(train_loader_linear):
#             batch_features = batch_features.to(device)
#             batch_labels = batch_labels.to(device)
#             optimizer.zero_grad()

#             # Mixed precision training
#             if amp:
#                 with torch.cuda.amp.autocast():
#                     outputs = model(batch_features)
#                     loss = criterion(outputs, batch_labels)

#                 scaler.scale(loss).backward()  # Scale the loss and backpropagate
#                 scaler.step(optimizer)  # Step the optimizer
#                 scaler.update()  # Update the scaler

#             else:
#                 outputs = model(batch_features)
#                 loss = criterion(outputs, batch_labels)
#                 loss.backward()
#                 optimizer.step()

#             if scheduler is not None:
#                 scheduler.step(epoch * iterations_per_epoch + minibatch)

#         all_labels = []
#         all_outputs = []

#         with torch.no_grad():
#             model.eval()
#             for minibatch, (batch_features, batch_labels) in enumerate(test_loader_linear):
#                 batch_features = batch_features.to(device)

#                 if amp:
#                     with torch.cuda.amp.autocast():
#                         outputs = model(batch_features)
#                 else:
#                     outputs = model(batch_features)

#                 all_labels.append(batch_labels.cpu().numpy())
#                 all_outputs.append(outputs.cpu().numpy())

#         all_labels = np.concatenate(all_labels)
#         all_outputs = np.vstack(all_outputs)

#         # when using amp, all_outputs should be changed to float32
#         if amp:
#             all_outputs = np.float32(all_outputs)

#         all_probs = softmax(all_outputs, axis=1)

#         # Compute ROC AUC score
#         avg_auc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovo')
#         if avg_auc > max_auc:
#             max_auc = avg_auc

#         # Compute F1 score
#         predicted_labels = np.argmax(all_outputs, axis=1)
#         macro_f1 = f1_score(all_labels, predicted_labels, average='macro')

#         # Compute Accuracy
#         accuracy = accuracy_score(all_labels, predicted_labels)

#         # Compute Confusion Matrix
#         conf_matrix = confusion_matrix(all_labels, predicted_labels)

#         if print_every:
#             print(f'Epoch({epoch}) AUC: {avg_auc:.3f}({max_auc:.3f}), Macro F1: {macro_f1:.3f}, Accuracy: {accuracy:.3f}')
#             print(f'Confusion Matrix:\n{conf_matrix}')

#         # Plot and save the confusion matrix
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
#                     xticklabels=[label_map[i] for i in range(len(label_map))],
#                     yticklabels=[label_map[i] for i in range(len(label_map))])
#         plt.title(f'Confusion Matrix - Epoch {epoch}')
#         plt.ylabel('True label')
#         plt.xlabel('Predicted label')
#         plt.savefig(f'./confusion_matrix_epoch_{epoch}.png')
#         plt.close()

#     # Print final metrics after training
#     print(f'Final Epoch({epoch}) AUC: {avg_auc:.3f}({max_auc:.3f}), Macro F1: {macro_f1:.3f}, Accuracy: {accuracy:.3f}')
#     print(f'Confusion Matrix:\n{conf_matrix}')

#     return avg_auc, macro_f1
