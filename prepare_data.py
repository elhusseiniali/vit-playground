# Import libraries
import torch
import torchvision
import torchvision.transforms as transforms
import os

def prepare_CIFAR10_data(batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=train_transform)
    if train_sample_size is not None:
        # Randomly sample a subset of the training set
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)



    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=True, transform=test_transform)
    if test_sample_size is not None:
        # Randomly sample a subset of the test set
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


def prepare_MNIST_data(batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=3),  # Convert to RGB by replicating channels
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((28, 28), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                            download=True, transform=train_transform)
    if train_sample_size is not None:
        # Randomly sample a subset of the training set
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)



    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.Normalize((0.5,), (0.5,))])

    testset = torchvision.datasets.MNIST(root='../data', train=False,
                                        download=True, transform=test_transform)
    if test_sample_size is not None:
        # Randomly sample a subset of the test set
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    
    return trainloader, testloader, classes


def prepare_Places365_data(batch_size=4, num_workers=2, train_sample_size=100000, test_sample_size=25000):
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
        # transforms.Resize((256, 256)),
        transforms.Resize((64, 64)), 
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    trainset = torchvision.datasets.Places365(root='./data',
                                               split='train-standard',
                                               # split='val',
                                               small= True, transform = train_transform, download= True)
    
    if train_sample_size is not None:
        # Randomly sample a subset of the training set
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)



    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        # transforms.Resize((256, 256)),
        transforms.Resize((64, 64)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    testset = torchvision.datasets.Places365(root='./data', split='val', 
                                                  small= True, transform = test_transform, download= True)
    if test_sample_size is not None:
        # Randomly sample a subset of the test set
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    from places365classes import places365_classes
    classes = places365_classes
    
    return trainloader, testloader, classes



def prepare_ImageNet200_data(batch_size=4, num_workers=2):

    from tiny_img import download_tinyImg200
    if not os.path.exists('./tiny-imagenet-200/'):
        download_tinyImg200('.')

    train_transform = transforms.Compose(
    [transforms.ToTensor(),
    # transforms.Resize((224, 224)),
    transforms.Resize((64, 64)), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
    transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.ImageFolder('tiny-imagenet-200/train', transform=train_transform)

    test_transform = transforms.Compose(
    [transforms.ToTensor(),
    # transforms.Resize((224, 224)),
    transforms.Resize((64, 64)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_dataset = torchvision.datasets.ImageFolder('tiny-imagenet-200/val', transform=test_transform)
    
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [80000, 20000])
    # test_dataset, val_dataset = torch.utils.data.random_split(val_dataset, [10000, 10000])

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
    

    classes = list(range(0, 200))
    
    return trainloader, testloader, classes
