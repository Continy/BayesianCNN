from network import BayesianCNN, CNN
import torch
from torchvision.models import resnet34
from datasets import DatasetLoader
from matplotlib import pyplot as plt
from torch.distributions import Normal

batch_size = 1
bayesian_model_path = "b_model.pth"
res_model_path = "model.pth"
n_classes = 2
hidden_dim = 512
bayesian_model = BayesianCNN(n_classes=n_classes, hidden_dim=hidden_dim)
res_model = CNN(n_classes=n_classes, hidden_dim=hidden_dim)

bayesian_model.load_state_dict(torch.load(bayesian_model_path))
res_model.load_state_dict(torch.load(res_model_path))
dataset_loader = DatasetLoader(root_dir="datasets/test", batch_size=batch_size)

test_loader = dataset_loader.get_dataloader()
res_model.eval()
bayesian_model.eval()
acc = []
b_acc = []
for i, (inputs, labels) in enumerate(test_loader):
    inputs = inputs + torch.randn_like(inputs) * 1
    outputs = res_model(inputs)
    mean, var = bayesian_model(inputs)
    _, preds = torch.max(outputs, 1)
    _, b_preds = torch.max(mean, 1)
    accuracy = torch.sum(preds == labels.data)
    b_accuracy = torch.sum(b_preds == labels.data)
    print(f"Batch {i + 1}/{len(test_loader)}")
    acc.append(accuracy / len(labels))
    b_acc.append(b_accuracy / len(labels))
print(f"Resnet Accuracy: {sum(acc) / len(acc)}")
print(f"Bayesian Accuracy: {sum(b_acc) / len(b_acc)}")
plt.scatter(range(len(acc)), acc)
plt.scatter(range(len(b_acc)), b_acc)
plt.show()
