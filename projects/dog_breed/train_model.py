#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
# import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
# from torchvision.models import ResNet18_Weights
import boto3
from PIL import Image
import argparse
import json
import os
from tqdm import tqdm
import shutil
from time import sleep
from io import BytesIO

from smdebug import modes
from smdebug.pytorch import get_hook

from PIL import ImageFile
Image.LOAD_TRUNCATED_IMAGES = True

s3 = boto3.client('s3', verify=True)

def test(model, test_loader, criterion, hook):
    '''
    This function tests the model on the test data and prints the accuracy.

    Parameters:
    model: The model to test
    test_loader: The data loader for the test data
    criterion: The loss criterion to use
    hook: The hook to use for profiling
    '''
    model.eval()
    if hook:
        hook.set_mode(modes.EVAL)

    test_loss = 0
    correct = 0
    with torch.no_grad(): # this disables gradient computation which is not needed for testing
        for data, target in test_loader: # iterate over the test data
            output = model(data) # get the model's prediction            
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # get the number of correct predictions

    test_loss /= len(test_loader.dataset) # calculate the average loss
    accuracy = 100. * correct / len(test_loader.dataset)  # Compute accuracy

    if hook:
        # Convert accuracy to a torch.Tensor
        accuracy_tensor = torch.tensor(accuracy, dtype=torch.float32)
        hook.record_tensor_value("accuracy", accuracy_tensor)  # Log accuracy as a tensor

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        

def train(model, train_loader, criterion, optimizer, epoch, hook):
    '''
    This function trains the model on the training data for each epoch.

    Parameters:
    model: The model to train
    train_loader: The data loader for the training data
    criterion: The loss criterion to use
    optimizer: The optimizer to use
    epoch: The current epoch number
    '''
    model.train()
    if hook:
        hook.set_mode(modes.TRAIN)

    for batch_idx, (data, target) in tqdm(enumerate(train_loader)): # iterate over the training data
        optimizer.zero_grad() # zero the gradients for this batch to avoid accumulation of gradients from previous batches
        output = model(data) # get the model's prediction
        loss = criterion(output, target)  # use the criterion to calculate the loss
        loss.backward() # backpropagate the loss because we want to minimize it
        optimizer.step() # update the model's weights based on the gradients calculated during backpropagation
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
def net(num_classes, model_type="resnet18", freeze_layers=True):
    if model_type == "resnet18":
        model = models.resnet18(pretrained=True)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        for param in model.fc.parameters():  # Enable gradients for classifier
            param.requires_grad = True
    elif model_type == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=True)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model

class CustomDataset(Dataset):
    def __init__(self, metadata, transform=None, processing_type='train'):
        self.metadata = [item for item in metadata if item['processing_type'] == processing_type]
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_meta = self.metadata[idx]
        local_image_path = image_meta['image_path']
        label_numeric = int(image_meta['label_numeric'])

        image = Image.open(local_image_path)

        # Apply additional transformations if needed
        if self.transform:
            image_tensor = self.transform(image)

        return image_tensor, label_numeric


def create_data_loaders(metadata, batch_size, shuffle=True, num_workers=2):
    '''
    Creates data loaders for training and testing.

    Parameters:
    data: The preprocessed data and labels
    batch_size: The size of each mini-batch
    shuffle: Whether to shuffle the data (default is True)
    num_workers: The number of subprocesses to use for data loading (default is 2)

    Returns:
    train_loader: DataLoader for the training data
    test_loader: DataLoader for the test data
    '''

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Create separate datasets for training and testing
    print("-> Creating custom data loaders...")
    train_dataset = CustomDataset(metadata, transform=transform, processing_type='train')
    test_dataset = CustomDataset(metadata, transform=transform, processing_type='test')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def create_validate_manifest():
    container_data_path = '/opt/ml/input/data'
    s3_output_path = os.environ.get('SM_OUTPUT_DATA_DIR')

    valid_metadata = []
    failed_metadata = []

    for root, dirs, files in os.walk(container_data_path):
        for file in files:
            image_path = os.path.join(root, file)
            path_split = root.split('/')

            if not file.endswith('.jpg'):
                failed_metadata.append(
                    {
                        'image_path': os.path.join(root, file),
                        'error_msg': 'Invalid file format',
                        'error_type': 'InvalidFileFormat'

                    }
                )
                continue # Skip the file if it's not a JPEG image
                
            processing_type = path_split[-2]
            label_numeric = path_split[-1].split(".")[0]
            label_description = path_split[-1].split(".")[1]
                
            # Open the image file with PIL
            image = Image.open(image_path)

            # Check if the image is in RGB format
            if image.mode != 'RGB':
                failed_metadata.append(
                    {
                        'image_path': os.path.join(root, file),
                        'error_msg': 'Invalid image mode',
                        'error_type': 'InvalidImageMode'
                    }
                )
                continue # Skip the file if it's not in RGB format

            # Check if the image is corrupted
            try:
                image.verify()
                image = Image.open(image_path)

                # Convert the image to RGB format to ensure compatibility with the model
                image = image.convert('RGB')

                valid_metadata.append(
                    {
                        'image_path': os.path.join(root, file),
                        'processing_type': processing_type,
                        'label_numeric': int(label_numeric)-1, # Subtract 1 to make the labels 0-indexed
                        'label_description': label_description
                    }
                )

            except Exception as e:

                failed_metadata.append(
                    {
                        'image_path': os.path.join(root, file),
                        'error_msg': str(e),
                        'error_type': 'CorruptedFile'
                    }
                )
                continue # Skip the file if it's corrupted  

    # Save the valid and failed metadata to JSON files
    valid_manifest_path = f'{s3_output_path}/metadata/valid_metadata.json'
    failed_manifest_path = f'{s3_output_path}/metadata/failed_metadata.json'

    # Create the directories if they don't exist
    os.makedirs(f'{s3_output_path}/metadata', exist_ok=True)

    with open(valid_manifest_path, 'w') as f:
        json.dump(valid_metadata, f)

    with open(failed_manifest_path, 'w') as f:
        json.dump(failed_metadata, f)

    return valid_metadata 

def save_torchscript_model(model, model_dir):
    # Set the model to evaluation mode
    model.eval()
    
    # Generate a dummy input that matches the input size of your model
    dummy_input = torch.randn(1, 3, 224, 224)  # Adjust based on your input shape

    try:
        # Convert the model to TorchScript using `torch.jit.trace`
        traced_model = torch.jit.trace(model, dummy_input)
    
        # Save the TorchScript model
        torch.jit.save(traced_model, f"{model_dir}/model.pth")
        print(f"-> TorchScript model saved to {model_dir}/model.pth")
    
        # Test the saved TorchScript model by loading it
        loaded_model = torch.jit.load(f"{model_dir}/model.pth")
        print("-> TorchScript model loaded successfully for verification.")

        # Run a forward pass using dummy input to validate
        loaded_model(dummy_input)
        print("-> TorchScript model verification successful: Forward pass completed.")

    except Exception as e:
        print(f"-> Error saving or testing TorchScript model: {e}")
        raise



def main(args):
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    print(f"-> Selected model type: {args.model_type}")
    model = net(args.num_classes, model_type=args.model_type, freeze_layers=True)


    # Initialize the Debuger/Profiler hook
    try:
        hook = get_hook(create_if_not_exists=True)
    except:
        hook = None
    print("*"*60)
    if hook:
        print("-> USING DEBUGER/PROFILER...")
        hook.register_hook(model)
        print("Registered collections:", hook.get_collections())
    else:
        print("-> USING LOCAL RUN...")
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion_options = {
        "cross_entropy": nn.CrossEntropyLoss()
        # Add more loss functions as needed
    }
    print(f"-> Using {args.criterion} loss criterion...")
    loss_criterion = loss_criterion_options[args.criterion]
    if hook:
        hook.register_loss(loss_criterion)

    optimizer_options = {
        "Adadelta": optim.Adadelta(model.parameters(), lr=args.lr),
        "Adam": optim.Adam(model.parameters(), lr=args.lr),
        "SGD": optim.SGD(model.parameters(), lr=args.lr)
    }
    print(f"-> Using {args.optimizer} optimizer...")
    optimizer = optimizer_options[args.optimizer]
    
    # Validate the data prior to training
    valid_metadata = create_validate_manifest()

    # Load the data
    train_loader, test_loader = create_data_loaders(valid_metadata, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
    
    print("*"*60)
    print("-> Starting model training...")
    for epoch in range(1, args.epochs + 1):
        '''
        TODO: Call the train function to start training your model
        Remember that you will need to set up a way to get training data from S3
        '''
        train(model, train_loader, loss_criterion, optimizer, epoch, hook=hook)
    
        '''
        TODO: Test the model to see its accuracy
        '''
        test(model, test_loader, loss_criterion, hook=hook)
    
    '''
    TODO: Save the trained model
    '''
    # Save the trained model
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    save_torchscript_model(model, model_dir)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Train dog breed classifier')
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument('--model-type', type=str, default='resnet18', help='model type to use (resnet18 or vgg16)')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='batch size for testing')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adadelta', help='optimizer to use')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='loss criterion to use')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle the training data')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--path', type=str, default='model.pth', help='path to save the trained model')
    parser.add_argument('--num-classes', type=int, default=133, help='number of classes in the dataset')   
    
    args=parser.parse_args()

    main(args)
