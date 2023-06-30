from torch.utils.data.dataset import Dataset
import numpy as np


class PointSDFPairDataset(Dataset):
    """
    A very simple dataset that returns (point, sdf) pair for the shape stored in data/hollow_knight.npz
    """
    def __init__(self):
        sample = np.load("data/hollow_knight.npz")
        self.points = sample["points"]
        self.sdf = sample["sdf"]
        self.len = 500000

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
        returns point sdf pair for network supervision
        """
        index = index % self.points.shape[0]
        return {
            'points': self.points[index].astype(np.float32),
            'sdf': self.sdf[index].astype(np.float32)
        }
