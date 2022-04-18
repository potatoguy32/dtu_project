import torch
import numpy as np
import sys

sys.path.append("src/data")
import data
sys.path.pop()


def process_data():
    trainset, testset = data.mnist()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False)
    train_images = torch.tensor(np.empty(shape=(0, 28, 28)))
    train_labels = torch.tensor(np.empty(shape=(0, )))
    for images, labels in trainloader:
        train_images = torch.cat((train_images, images.view((500, 28, 28))), 0)
        train_labels = torch.cat((train_labels, labels))
    
    with open("data/processed/train.npz", "wb") as f:
        np.savez(f, images=train_images, labels=train_labels)
        
    test_images = torch.tensor(np.empty(shape=(0, 28, 28)))
    test_labels = torch.tensor(np.empty(shape=(0, )))
    for images, labels in testloader:
        test_images = torch.cat((test_images, images.view((500, 28, 28))), 0)
        test_labels = torch.cat((test_labels, labels))
    
    with open("data/processed/test.npz", "wb") as f:
        np.savez(f, images=test_images, labels=test_labels)
    
    return 0

if __name__ == '__main__':
    process_data()