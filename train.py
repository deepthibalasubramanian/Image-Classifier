import argparse
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

def build_and_train_model(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    print(f"Selected model architecture: {arch}")
    print(f"Learning rate: {learning_rate}, Hidden units: {hidden_units}, Epochs: {epochs}")
    # Define data directories
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32),
    }

    # Load pre-trained model architecture
    if arch == 'VGG':
        model = models.vgg16(pretrained=True)
        num_features = model.classifier[0].in_features
    elif arch == 'DenseNet':
        model = models.densenet121(pretrained=True)
        num_features = model.classifier.in_features
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier
    classifier = nn.Sequential(
        nn.Linear(num_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    # Use GPU if available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Train the classifier
    steps = 0
    running_loss = 0
    print_every = 20

    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss, accuracy = validate(model, dataloaders['valid'], criterion, device)

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss:.3f}.. "
                      f"Validation accuracy: {accuracy:.3f}")

                running_loss = 0
                model.train()

    # Save the checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {
        'arch': arch,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'model_state_dict': model.state_dict()
    }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, save_path)

    print(f"Model trained and saved to {save_path}")

def validate(model, dataloader, criterion, device='cpu'):
    model.eval()
    accuracy = 0
    validation_loss = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model.forward(inputs)
            validation_loss += criterion(outputs, labels).item()

            ps = torch.exp(outputs)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

    return validation_loss / len(dataloader), accuracy / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description='Train a deep learning model on a dataset')
    parser.add_argument('data_dir', type=str, help='Path to the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the model checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg16 or densenet121)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of units in the hidden layer of the classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')

    args = parser.parse_args()

    build_and_train_model(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)

if __name__ == '__main__':
    main()
