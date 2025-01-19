import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import copy
import argparse
import os
import logging
import sys
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#rom torch_snippets import Report
#from torch_snippets import *

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion):
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(test_loader)
    total_acc = running_corrects.double() / len(test_loader)
    


def train(model, train_loader, validation_loader, criterion, optimizer):
    epochs=5
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    #log = Report(epochs)

    for epoch in range(epochs):

        # Shuffle the data for distributed training
        train_loader.sampler.set_epoch(epoch)

        for phase in ['train', 'valid']:
            if phase=='train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = validation_loader

            # Reset running loss and corrects
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            logger.info(f"{phase.capitalize()} Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase=='valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter+=1

        if loss_counter==1:
            logger.info('Early stopping')
            break

    return model
    
def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    return model

def create_data_loaders(data, batch_size, rank, world_size):
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, num_replicas=world_size, rank=rank)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, sampler=test_sampler)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_data, num_replicas=world_size, rank=rank)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, sampler=validation_sampler) 
    
    return train_data_loader, test_data_loader, validation_data_loader

if __name__=='__main__':
    """
    Pre-requisites:
    1. Setup the dataset in all the nodes 
        1. Clone https://github.com/samlexrod/udacity-education.git
        2. Change directory to /projects/operationalizing 
        3. Run `source dogImagesSetup.sh`
            * Creates a folder called 'dogImages' in the same directory as this script in all the nodes.
            * Downloads the dog dataset from https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
            * Unzips the dataset in the 'dogImages' folder
            * Removes the zip file
    4. Setup the environment variables for distributed training by adding them to /etc/environment
        * Edit the /etc/environment file on the master node
        ```
        vim /etc/environment
        ```
        * Add the following environment variables to the file on insert mode
        ```
        MASTER_ADDR=<master-private-ip>
        MASTER_PORT=8888
        WORLD_SIZE=2
        RANK=0
        ```
        * Save and exit the file by pressing `Esc` and typing `:wq`
        * Repeat the same steps for the slave nodes, ensuring the `RANK` is set to the appropriate value
    5. Ensure all slave nodes can communicate with the master node.
        * Add inbound rule to the security group of the master node to allow traffic from the slave nodes.
        * Use the `All traffic` rule to allow traffic from the slave nodes, for simplicity. Only use necessary rules in production.
        * You will need to ping the master node from the slave nodes to ensure communication is working.
        ```bash
        ping -c 5 <master-private-ip>
        ```
        * Once the ping is successful, you can proceed to the next step.
    """

    batch_size=2
    learning_rate=1e-4

    # Initialize distributed process group
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Set up loaders
    train_loader, test_loader, validation_loader=create_data_loaders('dogImages',batch_size, rank, world_size)
    model=net().to(rank)

    # Use DistributedDataParallel to make model distributed
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.module.fc.parameters(), lr=learning_rate)

    logger.info("Starting Model Training")
    model=train(model, train_loader, validation_loader, criterion, optimizer)

    if rank == 0:
        torch.save(model.state_dict(), 'TrainedModels/model.pth')
        logger.info('saved')
