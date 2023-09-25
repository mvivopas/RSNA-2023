import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import nibabel as nib

from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler

import sys
sys.path.append("dataloaders")
from dataloader import RSNA_AbdominalInjuryDataset

sys.path.append("models")
from DenseNet3D import DenseNet121, DenseNet169, DenseNet201, DenseNet264
from ResNet3D import ResNet50, ResNet101, ResNet152

ARGS_PATH = 'arguments.json'
ORGANS = ['liver', 'spleen', 'kidney_right', 'kidney_left', 'small_bowel']

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize your DataLoader here, just as you did before
    with open(ARGS_PATH) as f:
        args = json.load(f)
    
    data_dir: str = args['data_niftis_path']
    csv_dir: str = args['csv_path']

    ## TODO: Adapt this part of the code to work with different organs called by the function
    full_dataset = RSNA_AbdominalInjuryDataset(data_dir, csv_dir, "bowel")

    # Initialize k-Fold cross-validation
    k_folds = 3
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(full_dataset)):
    
        # Print
        print('\n--------------------------------')
        print(f'FOLD {fold}')
        print('--------------------------------\n')
    
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = DataLoader(
                          full_dataset, 
                          batch_size=2, sampler=train_subsampler)
        testloader = DataLoader(
                          full_dataset,
                          batch_size=1, sampler=test_subsampler)

        # Initialize TensorBoard writer
        writer = SummaryWriter('runs/RSNA_experiment') 

        # Initialize model, loss, optimizer
        #model = DesneNet121().to(device)
        model = ResNet50(num_classes=2, channels=1).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(100):
            epoch_loss = 0.0
            for i, (inputs, labels) in enumerate(tqdm(trainloader)): 
                inputs, labels = inputs.float().to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                epoch_loss += loss.item()

                writer.add_scalar('training loss', epoch_loss / (i + 1), epoch * len(trainloader) + i)

                if i % 4 == 1:  
                    loss.backward()
                    optimizer.step()

            print(f"\nTraining Epoch {epoch + 1} loss: {epoch_loss / (i+1):.3f}")

            # Validation loop
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(testloader):
                    inputs, labels = inputs.float().to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                val_accuracy = 100 * correct / total
                writer.add_scalar('Validation Loss', val_loss / (i + 1), epoch)
                writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

            print(f"Validation Epoch {epoch + 1} loss: {val_loss / (i + 1):.3f}, accuracy: {val_accuracy:.2f}%\n")

        writer.close()