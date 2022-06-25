import os
import sys
import tempfile
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    print(f'rank:{rank}, world_size:{world_size}')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)  # , num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)  # , num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()

    # imshow(torchvision.utils.make_grid(images))
    # print labels
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    return trainset, trainloader, testset, testloader, classes


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")

    setup(rank, world_size)

    trainset, trainloader, testset, testloader, classes = generate_data()

    net = Net().to(rank)

    ddp_model = DDP(net, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            labels = labels.to(rank)

            outputs = ddp_model(inputs.to(rank))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Should cleanup be here or below?
            # cleanup()

            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                break

    # Should cleanup be here or above?
    cleanup()

    print('Finished Training')


def demo_checkpoint(rank, world_size):
    print(f"Running checkpoint DDP example on rank {rank}.")
    setup(rank, world_size)

    trainset, trainloader, testset, testloader, classes = generate_data()

    net = Net().to(rank)

    ddp_model = DDP(net, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001, momentum=0.9)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

        # Use a barrier() to make sure that process 1 loads the model after process
        # 0 saves it.
        dist.barrier()
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        ddp_model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location=map_location))

    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloader):

            inputs, labels = data

            optimizer.zero_grad()

            labels = labels.to(rank)

            outputs = ddp_model(inputs.to(rank))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Should cleanup be here or below?
            # cleanup()

            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                break

        # Should the model be saved before or after cleanup?
        #
        # Saving the model after each epoch
        if rank == 0:
            # All processes should see same parameters as they all start from same
            # random parameters and gradients are synchronized in backward passes.
            # Therefore, saving it in one process is sufficient.
            torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

            # Should cleanup be here or above?
            # cleanup()

    # Now that we are done with training, do we now cleanup the processes again?
    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()

    print('Finished Training')

    # PATH = './cifar_net.pth'
    # torch.save(net.state_dict(), PATH)

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    # Reloading the model to the rank and to DDP to since we have already
    # executed "cleanup" on the model
    # create local model
    net = Net().to(rank)

    ddp_model = DDP(net, device_ids=[rank])

    # Loading our model since we have redefined our
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    outputs = ddp_model(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(4)))

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data

            labels = labels.to(rank)

            outputs = ddp_model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():

        for data in testloader:
            images, labels = data

            labels = labels.to(rank)

            outputs = ddp_model(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):

                labels = labels.to(rank)
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


class NetMP(nn.Module):
    def __init__(self, dev0, dev1, dev2, dev3):
        super(NetMP, self).__init__()
        # super().__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.dev2 = dev2
        self.dev3 = dev3
        self.conv1 = nn.Conv2d(3, 6, 5).to(dev0)
        self.pool = nn.MaxPool2d(2, 2).to(dev1)
        self.conv2 = nn.Conv2d(6, 16, 5).to(dev2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120).to(dev3)
        self.fc2 = nn.Linear(120, 84).to(dev1)
        self.fc3 = nn.Linear(84, 10).to(dev2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def demo_model_parallel(rank, world_size):
    # For the sake of reducing redundncy let's implement demo_model_parallel as a
    # version of demo_basic, not demo_checkpoint
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    dev0 = ((rank * 4) % 4)
    dev1 = ((rank * 4 + 1) % 4)
    dev2 = ((rank * 4 + 2) % 4)
    dev3 = ((rank * 4 + 3) % 4)

    trainset, trainloader, testset, testloader, classes = generate_data()

    net = NetMP(dev0, dev1, dev2, dev3)

    ddp_model = DDP(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloader):

            inputs, labels = data

            optimizer.zero_grad()

            labels = labels.to(dev1)

            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Should cleanup be here or below?
            # cleanup()

            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                break

            # Should cleanup be here or above?
            cleanup()

    print('Finished Training')


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    # run_demo(demo_basic, world_size)
    mp.spawn(demo_basic, args=(world_size,), nprocs=world_size, join=True)

    # run_demo(demo_checkpoint, world_size)
    # run_demo(demo_model_parallel, world_size)
