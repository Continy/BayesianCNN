import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# Define custom dataset class
class CatsAndDogsDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.cat_imgs = [
            os.path.join(os.path.join(root_dir, "cats"), f)
            for f in os.listdir(os.path.join(root_dir, "cats"))
        ]
        self.dog_imgs = [
            os.path.join(os.path.join(root_dir, "dogs"), f)
            for f in os.listdir(os.path.join(root_dir, "dogs"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.cat_imgs) + len(self.dog_imgs)

    def __getitem__(self, index):
        if index < len(self.cat_imgs):
            img_path = self.cat_imgs[index]
            label = 0
        else:
            img_path = self.dog_imgs[index - len(self.cat_imgs)]
            label = 1

        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label


class DatasetLoader:

    def __init__(self, root_dir, batch_size=32, shuffle=True):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def get_dataloader(self):
        dataset = CatsAndDogsDataset(root_dir=self.root_dir,
                                     transform=self.transform)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle)
        return dataloader


if __name__ == '__main__':
    # Define dataset loader
    dataset_loader = DatasetLoader(root_dir='datasets/train')
    dataloader = dataset_loader.get_dataloader()

    # Iterate over data
    for i, data in enumerate(dataloader):
        inputs, labels = data
        print(inputs.shape, labels.shape)
