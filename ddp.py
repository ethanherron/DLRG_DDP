import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
import time


# Global variables
PATH_DATASETS = '/work/mech-ai-scratch/DLRG_Datasets/CIFAR10'
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)


def main(args):
    
    
    # Global variables
    PATH_DATASETS = '/work/mech-ai-scratch/DLRG_Datasets/CIFAR10'
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = int(os.cpu_count() / 2)
    # Prep dataset
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    cifar10_dm = CIFAR10DataModule(
        data_dir=PATH_DATASETS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    # create our model - we are just going to edit the torchvision resnet model for 32x32 cifar10 image sizes
    def create_model():
        model = torchvision.models.resnet18(pretrained=False, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        return model

    class LitResnet(LightningModule):
        def __init__(self, lr=0.05):
            super().__init__()

            self.save_hyperparameters()
            self.model = create_model()

        def forward(self, x):
            out = self.model(x)
            return F.log_softmax(out, dim=1)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
            self.log("train_loss", loss)
            return loss

        def evaluate(self, batch, stage=None):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = accuracy(preds, y)

            if stage:
                self.log(f"{stage}_loss", loss, prog_bar=True)
                self.log(f"{stage}_acc", acc, prog_bar=True)

        def validation_step(self, batch, batch_idx):
            self.evaluate(batch, "val")

        def test_step(self, batch, batch_idx):
            self.evaluate(batch, "test")

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
            return {"optimizer": optimizer}



    model = LitResnet(lr=0.05)

    trainer = Trainer(
        max_epochs=50,
        accelerator="gpu",
        strategy='ddp', # Data Distributed Parallel Strategy! Pytorch lightning takes care of this for you!!!
        devices=args.num_devices,  
        logger=CSVLogger(save_dir="logs/"),
    )


    start_time = time.time()
    trainer.fit(model, cifar10_dm)
    end_time = time.time()
    
    print(f'{args.num_devices} GPUs with batch sizes of {args.batch_size} each took ', (end_time - start_time), ' seconds to train.')
    
    
    
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Train CIFAR 10 classification model with DDP!')
        parser.add_argument('-b','--batch_size', default=512, type=int,
                            help='Batch size')
        parser.add_argument('-g','--num_devices', default=4, type=int,
                            help='num gpus')
        hparams = parser.parse_args()
        main(hparams)