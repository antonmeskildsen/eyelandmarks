from typing import Type

from torch import utils, optim, nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from eyelandmarks.data import *

class EyeDataModule(pl.LightningDataModule):
    def __init__(self, path, dataset_type: Type[EyeDataset], transformed_width=128, crop_size=256, num_workers=4):
        super().__init__()
        self.path = path
        self.num_workers = num_workers
        self.transformed_width = transformed_width
        self.crop_size = crop_size
        self.dataset_type = dataset_type

    def prepare_data(self):
        self.data = self.dataset_type(
            self.path, self.transformed_width, self.crop_size, random_crop=True
        )

    def setup(self, stage):
        gen = torch.Generator().manual_seed(42)

        train_len = int(len(self.data) * 0.85)
        val_len = len(self.data) - train_len

        self.train_set, self.val_set = utils.data.random_split(
            self.data, (train_len, val_len), generator=gen
        )

    def train_dataloader(self):
        return utils.data.DataLoader(self.train_set, 64, num_workers=self.num_workers)

    def val_dataloader(self):
        return utils.data.DataLoader(self.val_set, 64, num_workers=self.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()



class LandmarkNet(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        mse = torchmetrics.MeanSquaredError()
        self.train_mse = mse.clone()
        self.val_mse = mse.clone()

        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters())
    
    def training_step(self, train_batch, batch_id):
        x, y = train_batch
        x = x.float()
        y = y.float()
        y_mark = self(x)

        loss = F.mse_loss(y_mark, y)
        preds = y_mark.detach()
        targets = y.detach()

        distance_error = torch.sqrt(torch.sum(((preds[:, :2]-targets[:, :2])*128)**2, dim=1))*((targets[:, 2]+1)/2)

        return {
            'loss': loss,
            'preds': preds,
            'targets': targets,
            'distance': distance_error
        }

    def training_step_end(self, outputs):
        self.train_mse(outputs['preds'], outputs['targets'])
        self.log('train/mse_step', self.train_mse)
        self.log('train/distance_error', outputs['distance'].mean())
        self.log('train/distance_error_median', outputs['distance'].median())
    
    def training_epoch_end(self, outputs):
        self.log('train/mse_epoch', self.train_mse.compute())
    
    def validation_step(self, val_batch, batch_id):
        res = self.training_step(val_batch, batch_id)
        return res
    
    def validation_step_end(self, outputs):
        self.val_mse(outputs['preds'], outputs['targets'])
        self.log('val/mse_step', self.train_mse)
        self.log('val/distance_error', outputs['distance'].mean())
        self.log('val/distance_error_median', outputs['distance'].median())
    
    def validation_epoch_end(self, outputs):
        self.log('val/mse_epoch', self.val_mse.compute())