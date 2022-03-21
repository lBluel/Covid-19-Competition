import os
import torch
import numpy as np

class Trainer:
    def __init__(self, criterion, optimizer, scheduler, data_loader, root_dir, batch_size, patience = 8):
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_loader = data_loader
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.patience = patience

    def train_model(self, name, model, epochs=30):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        min_validation_loss, patience = np.inf, 0
        train_loss, val_loss = [], []
        for epoch in range(epochs):
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                    batch_step_size = len(self.data_loader[phase].dataset) / self.batch_size[phase]
                else:
                    model.eval()
                    batch_step_size = len(self.data_loader[phase].dataset) / self.batch_size[phase]
                
                log_loss = []
                for batch_idx, sample in enumerate(self.data_loader[phase]):
                    images = sample["image"].to(device)
                    labels = sample['percentage'].to(device)
                    outputs = model(images).squeeze().float()
                    loss = self.criterion(outputs, labels)
                    
                    if phase == 'train':
                        with torch.set_grad_enabled(True):
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            self.scheduler.step()

                    log_loss.append(loss.item())
                    if batch_idx % 25 == 0:
                        print("Epoch {} : {} ({:04d}/{:04d}) Loss = {:.4f}".format(epoch, phase, batch_idx, int(batch_step_size), loss.item()))
                
                if phase == 'train':    
                    train_loss.append(np.mean(log_loss))
                else:
                    val_loss.append(np.mean(log_loss))

            if val_loss[-1] < min_validation_loss:
                patience = 0
                print("Validation loss decreased. Saving model as checkpoint...")
                if not os.path.isdir(os.path.join(self.root_dir, name)):
                    os.makedirs(os.path.join(self.root_dir, name))

                min_validation_loss = val_loss[-1]
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': val_loss[-1]
                }, os.path.join(self.root_dir, name, "checkpoint.tar".format(epoch)))
            else:
                if patience > self.patience:
                    print("Validation loss hasn't decreased for {} epochs. Early Stopping the training...".format(patience))
                    break
                patience += 1
            np.save(os.path.join(self.root_dir, name, "val-loss-epoch-{}.npy".format(epoch)), val_loss)
            np.save(os.path.join(self.root_dir, name, "train-loss-epoch-{}.npy".format(epoch)), train_loss)
        return model, train_loss, val_loss