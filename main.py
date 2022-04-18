import argparse
import sys
import io

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim

from src.models.model import MyAwesomeModel
import src.visualizations.helper as helper
from src.data.data import mnist


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.001)
        # add any additional argument that you want
        parser.add_argument('--optimizer', default='adam')
        parser.add_argument('--criterion', default='nll')
        parser.add_argument('--epochs', default=5)
        parser.add_argument('--print', default=50)
        args = parser.parse_args(sys.argv[2:])
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        lr = float(args.lr)
        setup = {
            'optimizer': {
                'adam': optim.Adam(model.parameters(), lr=lr),
                'sgd': optim.SGD(model.parameters(), lr=lr)
            },
            'criterion': {
                'nll': nn.NLLLoss(),
                'mse': nn.MSELoss(),
                'entropy': nn.CrossEntropyLoss()
            }
        }

        trainset, testset = mnist()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
        optimizer = setup['optimizer'][args.optimizer]
        criterion = setup['criterion'][args.criterion]
        epochs = int(args.epochs)
        steps = 0
        print_every = args.print
        running_loss = 0
        training_losses = []
        test_losses = []
        test_accuracy = 0
        print("Training day and night...")
        for e in range(epochs):
            # Model in training mode, dropout is on
            model.train()
            for images, labels in trainloader:
                steps += 1
                # Flatten images into a 784 long vector
                images.resize_(images.shape[0], 784)
                images = images.type(torch.FloatTensor)
                optimizer.zero_grad()
                output = model.forward(images)
                labels = labels.type(torch.LongTensor)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if steps % print_every == 0:
                    # Model in inference mode, dropout is off
                    model.eval()
                    # Turn off gradients for validation, will speed up inference
                    with torch.no_grad():
                        accuracy = 0
                        test_loss = 0
                        for images, labels in testloader:
                            images = images.resize_(images.shape[0], 784)
                            images = images.type(torch.FloatTensor)
                            output = model.forward(images)
                            test_loss += criterion(output, labels).item()
                            ## Calculating the accuracy
                            # Model's output is log-softmax, take exponential to get the probabilities
                            ps = torch.exp(output)
                            # Class with highest probability is our predicted class, compare with true label
                            equality = (labels.data == ps.max(1)[1])
                            # Accuracy is number of correct predictions divided by all predictions, just take the mean
                            accuracy += equality.type_as(torch.FloatTensor()).mean()

                    training_losses.append(running_loss/print_every)
                    test_losses.append(test_loss/len(testloader))
                    test_accuracy = accuracy/len(testloader)
                    print("Epoch: {}/{} ".format(e+1, epochs),
                            "Training Loss: {:.3f} ".format(running_loss/print_every),
                            "Test Loss: {:.3f} ".format(test_loss/len(testloader)),
                            "Test Accuracy: {:.3f}".format(test_accuracy))

                    running_loss = 0
                    # Make sure dropout and grads are on for training
                    model.train()
                if test_accuracy >= 0.85:
                    break

        plt.plot(training_losses, label="train loss")
        plt.plot(test_losses, label="test loss")
        plt.legend()
        plt.show()
        torch.save(model.state_dict(), "models/MyAwesomeModelCheckpoint.pth")
        return

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default='models/MyAwesomeModelCheckpoint.pth')
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))
        model.eval()
        _, testset = mnist()
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
        accuracy = 0
        for images, labels in testloader:
            images = images.resize_(images.shape[0], 784)
            images = images.type(torch.FloatTensor)
            accuracy += (torch.exp(model.forward(images)).max(1)[1] == labels.data).type_as(torch.FloatTensor()).mean()

        print("Test set accuracy: {:.3f}".format(accuracy/len(testloader)))
        print("Here's an example: ")
        images_iterator = iter(testloader)
        image, _ = next(images_iterator)
        helper.view_classify(img=image[0, :], ps=torch.exp(model(image[0, :])))
        plt.show()
        return

if __name__ == '__main__':
    TrainOREvaluate()
