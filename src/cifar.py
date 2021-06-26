# Copyright 2021 Alvaro Bartolome, alvarobartt @ GitHub
# See LICENSE for details.

from torch.nn.modules import transformer
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule


class SimpsonsTransforms(T.Compose):
    def __init__(self, phase):
        self.phase = phase
        self.transforms = {
            'train': [
                T.Resize((32, 32)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ],
            'val': [
                T.Resize((32, 32)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ],
            'test': [
                T.Resize((32, 32)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        }
        
        super().__init__(self.transforms[self.phase])


class SimpsonsImageFolder(ImageFolder):
    def __init__(self, root, phase):
        super().__init__(root=f"{root}/{phase}")
        self.phase = phase
        self.transform = SimpsonsTransforms(phase=phase)


class SimpsonsDataModule(LightningDataModule):
    def __init__(self, dataset_path, batch_size):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
    
    def train_dataloader(self):
        self.train_image_folder = SimpsonsImageFolder(root=self.dataset_path, phase='train')

        return DataLoader(dataset=self.train_image_folder,
                          batch_size=self.batch_size, 
                          num_workers=12, shuffle=True)
    
    def val_dataloader(self):
        self.val_image_folder = SimpsonsImageFolder(root=self.dataset_path, phase='val')

        return DataLoader(dataset=self.val_image_folder,
                          batch_size=self.batch_size, 
                          num_workers=12)
    
    def test_dataloader(self):
        self.test_image_folder = SimpsonsImageFolder(root=self.dataset_path, phase='test')
        
        return DataLoader(dataset=self.test_image_folder,
                          batch_size=self.batch_size,
                          num_workers=12)
