import torch
from network import BayesianCNN, CNN
from datasets import DatasetLoader
import torch.nn as nn
from mc_loss import monte_carlo
# Define hyperparameters
n_epochs = 30
batch_size = 64
lr = 0.001
n_classes = 2
hidden_dim = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset_loader = DatasetLoader(root_dir="datasets/train",
                               batch_size=batch_size)
train_loader = dataset_loader.get_dataloader()

# Define model

model = BayesianCNN(n_classes=n_classes, hidden_dim=hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Train model

for epoch in range(n_epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    running_loss = 0.0
    running_corrects = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs + torch.randn_like(inputs) * 0.5
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        mean, var = model(inputs)

        loss = criterion(mean, labels)
        exp_var = torch.exp(var)
        mc_loss = monte_carlo(mean, exp_var, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(mean, 1)
        running_corrects += torch.sum(preds == labels.data)
        print(f"Batch {i + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

# Save model
torch.save(model.state_dict(), "b_model.pth")
print("Done!")
