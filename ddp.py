import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
import time





def main(args):
    
    print(' ')
    print(' prepping dataset...')
    # Global variables
    # Prep dataset
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]
    )

    train_dataset = CIFAR10('./data', train=True, transform=train_transforms, download=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=True, drop_last=True, 
                                               num_workers=2, 
                                               pin_memory=True)

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

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
            return {"optimizer": optimizer}


    print(' ')
    print(' creating model...')
    model = LitResnet(lr=0.05)

    print(' ')
    print(' training model...')
    trainer = Trainer(
        max_epochs=3,
        accelerator='gpu',
        strategy='ddp', # Data Distributed Parallel Strategy! Pytorch lightning takes care of this for you!!!
        devices=args.num_devices
    )

    start_time = time.time()
    trainer.fit(model, train_loader)
    end_time = time.time()
    
    print(f'{args.num_devices} GPUs with batch sizes of {args.batch_size} each took ', (end_time - start_time), ' seconds to train.')
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CIFAR 10 classification model with DDP!')
    parser.add_argument('-b','--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('-g','--num_devices', default=1, type=int,
                        help='num gpus')
    hparams = parser.parse_args()
    main(hparams)
