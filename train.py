import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import Dataset
from model import build_unet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Load dataset """
    train_x = sorted(glob("/home/fteam5/strain/data/train/image/*jpg"))
    train_y = sorted(glob("/home/fteam5/strain/data/train/mask/*jpg"))

    valid_x = sorted(glob("/home/fteam5/strain/data/valid/image/*jpg"))
    valid_y = sorted(glob("/home/fteam5/strain/data/valid/mask/*jpg"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 256
    W = 256
    size = (H, W)
    batch_size = 16
    num_epochs = 35
    lr = 1e-4
    checkpoint_path = "/home/fteam5/strain/checkpoint/check.pth"

    """ Dataset and loader """
    train_dataset = Dataset(train_x, train_y)
    valid_dataset = Dataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = build_unet()
    model = model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_valid_loss = float("inf")

    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()

        epoch_train_loss = 0.0

        # Wrap train_loader with tqdm
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as t:
            for x, y in t:
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)

                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()

                # Update tqdm description with current loss
                t.set_postfix(train_loss=epoch_train_loss/(len(t)))

        # Calculate average training loss for the epoch
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation loss
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {epoch_train_loss:.3f}\n'
        data_str += f'\tValidation Loss: {valid_loss:.3f}\n'
        print(data_str)

        # Save train and validation losses
        valid_losses.append(valid_loss)

        # Step scheduler
        scheduler.step(valid_loss)

    # Plotting and saving the loss curve
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('/home/fteam5/strain/checkpoint/loss.jpg')
    plt.show()
