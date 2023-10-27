from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
import torch.nn.functional as F
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torchvision.models as models
import torch
import torchmetrics
import numpy as np
import sys
from torchvision.models import resnet50, ResNet50_Weights
from torch.autograd import Variable
from torchvision.ops.focal_loss import sigmoid_focal_loss

class BalancedAccuracy(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("true_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        tp, tn, fp, fn = self.calculate_metrics(preds, target)
        self.true_positives += tp
        self.true_negatives += tn
        self.false_positives += fp
        self.false_negatives += fn

    def calculate_metrics(self, preds, target):
        tp = torch.sum((preds == 1) & (target == 1))
        tn = torch.sum((preds == 0) & (target == 0))
        fp = torch.sum((preds == 1) & (target == 0))
        fn = torch.sum((preds == 0) & (target == 1))
        return tp, tn, fp, fn

    def compute(self):
        balanced_acc = (self.true_positives / (self.true_positives + self.false_negatives) +
                        self.true_negatives / (self.true_negatives + self.false_positives)) / 2
        return balanced_acc

# define the LightningModule
class LitResNet(pl.LightningModule):
    def __init__(self, alpha=0.75, gamma=2.0, in_channels=3):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(2048, 2)

        if in_channels == 1:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        for param in self.model.parameters():
            param.requires_grad = True

        self.train_acc = BalancedAccuracy()
        self.valid_acc = BalancedAccuracy()
        self.test_acc = BalancedAccuracy()
        self.alpha = alpha
        self.gamma = gamma
        self.name = "resnet"

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y, _ = batch
        logits = self.model(x)
        # training metrics
        preds = torch.argmax(logits, dim=1)
        y_reshaped = torch.zeros(x.size(0), 2)
        y_reshaped[range(y_reshaped.shape[0]), y] = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y_reshaped = y_reshaped.to(device)
        loss = sigmoid_focal_loss(logits, y_reshaped, alpha=self.alpha, gamma=self.gamma, reduction="mean")
        self.train_acc(preds, y)
        self.log("train_loss", loss, batch_size=x.size(0), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, batch_size=x.size(0), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation_loop
        x, y, _ = batch
        logits = self.model(x)
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        y_reshaped = torch.zeros(x.size(0), 2)
        y_reshaped[range(y_reshaped.shape[0]), y] = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y_reshaped = y_reshaped.to(device)
        loss = sigmoid_focal_loss(logits, y_reshaped, alpha=self.alpha, gamma=self.gamma, reduction="mean")
        self.valid_acc(preds, y)
        self.log("val_loss", loss, batch_size=x.size(0), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.valid_acc, batch_size=x.size(0), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y, _ = batch
        logits = self.model(x)
        preds = torch.argmax(logits, dim=1)
        y_reshaped = torch.zeros(x.size(0), 2)
        y_reshaped[range(y_reshaped.shape[0]), y] = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y_reshaped = y_reshaped.to(device)
        loss = sigmoid_focal_loss(logits, y_reshaped, alpha=self.alpha, gamma=self.gamma, reduction="mean")
        self.test_acc(preds, y)
        self.log("test_loss", loss, batch_size=x.size(0), on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, batch_size=x.size(0), on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def forward(self, x):
        # used by pl for inference
        return self.model(x)
    
    def predict_step(self, batch, batch_idx):
        # used by pl for inference
        x, y, cat_ids = batch
        logits = self.model(x)
        preds = torch.argmax(logits, dim=1)
        # flatten preds, y and cat_ids
        preds = preds.flatten()
        preds = preds.cpu().numpy()
        y = y.flatten()
        y = y.cpu().numpy()
        cat_ids = list(cat_ids)

        return y, preds, cat_ids