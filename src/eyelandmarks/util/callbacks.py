import pytorch_lightning as pl
import wandb

import numpy as np
import cv2 as cv

def scale(val, width):
    return (val+0.5)*width

class PupilCenterCallback(pl.Callback):
    """Logs the input and output images of a module.
    
    Images are stacked into a mosaic, with output on the top
    and input on the bottom."""
    
    def __init__(self, log_interval=20):
        super().__init__()
        self.log_interval = log_interval
          
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % self.log_interval == 0:
            x, _ = batch
            x = np.array(x.cpu().detach().numpy()*255, dtype=np.uint8)

            size = x.shape[2]

            imgs = []
            for img, y_mark, y in zip(x, outputs['preds'], outputs['targets']):
                img = cv.cvtColor(img[0], cv.COLOR_GRAY2BGR)


                img = cv.drawMarker(img, (int(scale(y_mark[0], size)), int(scale(y_mark[1], size))), (0, 0, 255), cv.MARKER_CROSS, 10, 2)
                img = cv.drawMarker(img, (int(scale(y[0], size)), int(scale(y[1], size))), (255, 0, 0), cv.MARKER_TILTED_CROSS, 10, 2)
                imgs.append(wandb.Image(img, caption=f'Sample'))
            
            caption = "Sample outputs"
            trainer.logger.experiment.log({
                "val/examples": imgs,
                "global_step": trainer.global_step
                })


class PupilLandmarksCallback(pl.Callback):
    """Logs the input and output images of a module.
    
    Images are stacked into a mosaic, with output on the top
    and input on the bottom."""
    
    def __init__(self, log_interval=20):
        super().__init__()
        self.log_interval = log_interval
          
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % self.log_interval == 0:
            x, _ = batch
            x = np.array(x.cpu().detach().numpy()*255, dtype=np.uint8)

            size = x.shape[2]

            imgs = []
            for img, y_mark, y in zip(x, outputs['preds'], outputs['targets']):
                img = cv.cvtColor(img[0], cv.COLOR_GRAY2BGR)

                for n in range(8): # 8 landmarks
                    img = cv.drawMarker(img, (int(scale(y_mark[n*2+0], size)), int(scale(y_mark[n*2+1], size))), (0, 0, 255), cv.MARKER_CROSS, 3, 1)
                    img = cv.drawMarker(img, (int(scale(y[n*2+0], size)), int(scale(y[n*2+1], size))), (255, 0, 0), cv.MARKER_TILTED_CROSS, 3, 1)
                imgs.append(wandb.Image(img, caption=f'Sample'))
            
            caption = "Sample outputs"
            trainer.logger.experiment.log({
                "val/examples": imgs,
                "global_step": trainer.global_step
                })


class AllLandmarksCallback(pl.Callback):
    """Logs the input and output images of a module.
    
    Images are stacked into a mosaic, with output on the top
    and input on the bottom."""
    
    def __init__(self, log_interval=20):
        super().__init__()
        self.log_interval = log_interval
          
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % self.log_interval == 0:
            x, _ = batch
            x = np.array(x.cpu().detach().numpy()*255, dtype=np.uint8)

            size = x.shape[2]

            imgs = []
            for img, y_mark, y in zip(x, outputs['preds'], outputs['targets']):
                img = cv.cvtColor(img[0], cv.COLOR_GRAY2BGR)

                for n in range(50): # 50 landmarks
                    img = cv.drawMarker(img, (int(scale(y_mark[n*2+0], size)), int(scale(y_mark[n*2+1], size))), (0, 0, 255), cv.MARKER_CROSS, 3, 1)
                    img = cv.drawMarker(img, (int(scale(y[n*2+0], size)), int(scale(y[n*2+1], size))), (255, 0, 0), cv.MARKER_TILTED_CROSS, 3, 1)
                imgs.append(wandb.Image(img, caption=f'Sample'))
            
            caption = "Sample outputs"
            trainer.logger.experiment.log({
                "val/examples": imgs,
                "global_step": trainer.global_step
                })