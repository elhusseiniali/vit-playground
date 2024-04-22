import torch
from tqdm import tqdm
import time

from utils import save_experiment, save_checkpoint


class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, config, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.config = config
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies, train_times, inference_times = (
            [],
            [],
            [],
            [],
            [],
        )

        # Train the model
        for i in range(epochs):
            print(f"Starting Epoch {i + 1} of {epochs}.")
            train_loss, train_time = self.step(trainloader)
            accuracy, test_loss, inference_time = self.evaluate(testloader)
            train_losses.append(train_loss)
            train_times.append(train_time)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            inference_times.append(inference_time)

            print(
                f"Epoch: {i+1}, Train loss: {train_loss:.4f}, "
                f"Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
            )

            if (
                save_model_every_n_epochs > 0
                and (i + 1) % save_model_every_n_epochs == 0
                and i + 1 != epochs
            ):
                print("\tSave checkpoint at epoch", i + 1)
                save_checkpoint(self.exp_name, self.model, i + 1)

        # Save the experiment

        save_experiment(
            self.exp_name,
            self.config,
            self.model,
            train_losses,
            test_losses,
            accuracies,
            train_times,
            inference_times,
        )

    def step(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        t0 = time.time()
        for batch in tqdm(trainloader, unit="batch", total=len(trainloader)):
            # Move the batch to the device
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss
            loss = self.loss_fn(self.model(images)[0], labels)
            # Backpropagate the loss
            loss.backward()
            # Update the model's parameters
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        t1 = time.time()
        train_time = t1 - t0
        return total_loss / len(trainloader.dataset), train_time

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        t0 = time.time()
        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                # Get predictions
                logits, _ = self.model(images)

                # Calculate the loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        t1 = time.time()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        inference_time = t1 - t0
        return accuracy, avg_loss, inference_time
