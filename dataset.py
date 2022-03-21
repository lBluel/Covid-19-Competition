import os
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision.transforms import Compose, Resize, RandomCrop, RandomRotation, Normalize, ToTensor

class CovidDataset(Dataset):
    def __init__(self, root_dir, phase = 'Train', transform=None):
        self.root_dir = os.path.join(root_dir, phase)
        self.transform = transform
        self.image_list = None
        self.phase = phase

        self.label_data = None
        if phase == 'Train':
            self.label_data = pd.read_csv(os.path.join(root_dir, "{}.csv".format(phase)), header=None)
            self.image_list = self.label_data.iloc[:, 0].to_list()
        else:
            self.image_list = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(image_path).convert("RGB")

        sample = dict({"image": image, "image_path": image_path})
        if self.phase == 'Train':
            percentage = self.label_data.iloc[idx, 1]
            subject = self.label_data.iloc[idx, 2]
            sample = dict({
                "image": image, 
                "image_path": image_path,
                "percentage": percentage,
                "subject": subject
            })

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample

def get_dataset(root_dir, phase, val_split = None):
    if phase == 'Val':
        return CovidDataset(
            root_dir, 
            phase=phase, 
            transform=Compose([
                Resize((256, 256)),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        )
    
    train_val_dataset = CovidDataset(
            root_dir,
            phase=phase,
            transform=Compose([
                Resize((256, 256)),
                RandomCrop((224, 224)),
                RandomRotation(degrees=(-10, 10)),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        )
    
    if val_split is None:
        return train_val_dataset
    
    val_split_length = int(len(train_val_dataset) * val_split)
    splits = [len(train_val_dataset) - val_split_length, val_split_length]
    train_dataset, val_dataset = random_split(train_val_dataset, splits)
    return train_dataset, val_dataset