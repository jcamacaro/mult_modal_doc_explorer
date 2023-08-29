import os
from torch.utils.data import Dataset
import torchvision
import torch

class ImageEncoder:

    def __init__(self,
            images_dir,
                 ):

        self.images_dir = images_dir
        self.data_loader = None
        self.model = None

    def load_images(self):
        # '/home/jaimec/Documents/Torch/learn_pytorch/images/TDA'
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.ImageFolder(root=self.images_dir,
                                                   transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000)
        self.data_loader = dataloader


    def encode_images(self):
        model = torchvision.models.resnet18(weights='DEFAULT')
        model.fc = torch.nn.Identity()
        model.eval()
        for data in self.data_loader:
            embed = model(data[0]).detach().numpy()
        return embed

    def get_imageso(self):
        return next(iter(self.data_loader))[0].numpy()

    def encode_image(self, imgs_tensor):
        model = torchvision.models.resnet18(weights='DEFAULT')
        model.fc = torch.nn.Identity()
        model.eval()
        embed = model(imgs_tensor).detach().numpy()
        return embed

    def image_encoder_pair(self):
        model = torchvision.models.resnet18(weights='DEFAULT')
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model
        for data in self.data_loader:
            embed = model(data[0]).detach().numpy()

        return [next(iter(self.data_loader))[0].numpy(), embed]
