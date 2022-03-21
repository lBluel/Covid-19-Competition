import os
import torch
import pandas as pd

from torch.optim import Adam
from torch.nn import SmoothL1Loss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.densenet import Densenet
from dataset import get_dataset
from train import Trainer

if __name__ == '__main__':
    name = "densenet"
    root_dir = "/home/sb4539/Covid-19-Competition"
    if not os.path.exists(os.path.join(root_dir, name)):
        os.makedirs(os.path.join(root_dir, name))

    batch_size = {'train': 16, 'valid': 16}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Densenet().to(device)
    
    train_datset, val_dataset = get_dataset(root_dir, phase='Train', val_split=0.1)
    dataloader = {
        'train': DataLoader(train_datset, batch_size=batch_size['train'], shuffle=True, num_workers=0),
        'valid': DataLoader(val_dataset, batch_size=batch_size['valid'], shuffle=True, num_workers=0)
    }

    criterion = SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_datset) // batch_size['train'])

    trainer = Trainer(criterion, optimizer, scheduler, dataloader, root_dir, batch_size)
    model, _, _ = trainer.train_model(name, model, epochs=30)

    model.eval()
    test_dataset = get_dataset(root_dir, phase='Val')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    results = []
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(test_loader):
            image = sample_batched['image'].float().to(device)
            img_name = sample_batched['image_path']
            output = model(image).squeeze().float()
            results.append([img_name[0].split("/")[-1], output.cpu().data.numpy()])
    df = pd.DataFrame(results, columns=['image_name', 'output']).sort_values('image_name').reset_index(drop=True)
    df.to_csv(os.path.join(root_dir, name, "predictions.csv"), index=False, header=False)