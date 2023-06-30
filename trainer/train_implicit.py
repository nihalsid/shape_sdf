from pathlib import Path

import pytorch_lightning as pl
import torch

import numpy as np
import marching_cubes as mc
from torchmetrics import MeanAbsoluteError

from dataset.point_sdf_pair import PointSDFPairDataset
from model.implicit import SimpleImplicitDecoder


class ImplicitTrainingModule(pl.LightningModule):

    def __init__(self):
        super(ImplicitTrainingModule, self).__init__()
        self.train_dataset, self.val_dataset = PointSDFPairDataset(), PointSDFPairDataset()
        self.model = SimpleImplicitDecoder()
        self.mean_error = MeanAbsoluteError()
        # for visualization of mesh, precompute the grid
        self.eval_resolution = 192
        x_range = y_range = z_range = np.linspace(-0.50, 0.50, self.eval_resolution).astype(np.float32)
        grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()
        stacked = torch.from_numpy(np.hstack((grid_x[:, np.newaxis], grid_y[:, np.newaxis], grid_z[:, np.newaxis]))).to(self.device)
        self.stacked_split = torch.split(stacked, 32 ** 3, dim=0)

    def forward(self, batch):
        return self.model(batch['points'])

    @staticmethod
    def loss(prediction, target):
        return torch.abs(prediction - target)

    def training_step(self, batch, batch_index):
        predicted_sdf = self.forward(batch)
        loss = self.loss(predicted_sdf, batch['sdf']).mean()
        return {'loss': loss}

    def validation_step(self, batch, batch_index):
        predicted_sdf = self.forward(batch)
        self.mean_error(predicted_sdf, batch['sdf'])

    def on_validation_epoch_end(self):
        print(f'\nError at step {self.global_step}: {self.mean_error.compute()}')
        self.mean_error.reset()
        self.visualize_prediction()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.model.parameters()), lr=0.0005)
        return [optimizer]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=2048, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    def visualize_prediction(self):
        sdf_values = []
        for points in self.stacked_split:
            with torch.no_grad():
                sdf = self.model(points.to(self.device))
            sdf_values.append(sdf.detach().cpu())
        sdf_values = torch.cat(sdf_values, dim=0).reshape((self.eval_resolution, self.eval_resolution, self.eval_resolution)).numpy()
        vertices, triangles = mc.marching_cubes(sdf_values, 0)
        vertices = vertices / self.eval_resolution - 0.5
        mc.export_obj(vertices, triangles, "outputs/mesh.obj")


def main():
    from datetime import datetime
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    # configuration
    max_epoch = 350
    save_epoch = 10
    check_val_every_n_epoch = 10
    experiment = f"{datetime.now().strftime('%d%m%H%M')}"

    checkpoint_callback = ModelCheckpoint(dirpath=(Path("runs") / experiment), filename='_ckpt_{epoch}', save_top_k=-1, verbose=False, every_n_epochs=save_epoch)
    model = ImplicitTrainingModule()
    trainer = Trainer(devices=[0], num_sanity_val_steps=0, val_check_interval=1.0, check_val_every_n_epoch=check_val_every_n_epoch, max_epochs=max_epoch, callbacks=[checkpoint_callback], benchmark=True)
    trainer.fit(model)


if __name__ == '__main__':
    main()
