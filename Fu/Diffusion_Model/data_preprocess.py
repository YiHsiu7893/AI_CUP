import os
import torch
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_folder, target_folder, img_transform=None, label_transform=None, mode="train"):
        self.data_folder = data_folder
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.data_files = sorted(os.listdir(data_folder))
        self.target_folder = target_folder
        self.target_files = sorted(os.listdir(target_folder))
        self.mode = mode

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_name = self.data_files[idx]
        data_path = os.path.join(self.data_folder, data_name)
        target_name = self.target_files[idx]
        target_path = os.path.join(self.target_folder, target_name)

        # Load input data image
        data = Image.open(data_path).convert("RGB")

        # Assuming target images have the same filename but in a different folder
        target = Image.open(target_path).convert("RGB")

        if self.mode == "train" and self.img_transform and self.label_transform:
            data = self.label_transform(data)
            target = self.img_transform(target)

        return data, target

class LabelDataset(Dataset):
    # This dataset is used for inference. It only loads the data and not the target
    def __init__(self, data_folder, label_transform=None):
        self.data_folder = data_folder
        self.img_transform = label_transform
        self.data_files = sorted(os.listdir(data_folder))

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data_name = self.data_files[idx]
        data_path = os.path.join(self.data_folder, data_name)

        # Load input data image
        data = Image.open(data_path).convert("RGB")

        if self.img_transform:
            data = self.img_transform(data)

        return data

# Example usage
# Assuming you have your data and targets already loaded
data = "../dataset/train/label_img"
targets = "../dataset/train/img"

# Define the transformation
img_transform = transforms.Compose(
    [
        transforms.Resize((171,96)),  # Resize the smallest edge to 128
        transforms.Lambda(lambda img: pad_to_desired(img)),  # Pad the image if needed
        # transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        # transforms.RandomRotation(10),  # Randomly rotate the image by up to 10 degrees
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
    ]
)

label_transform = transforms.Compose(
    [
        transforms.Resize((171,96)),  # Resize the smallest edge to 128
        transforms.Lambda(lambda img: pad_to_desired(img)),  # Pad the image if needed
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ]
)

def pad_to_desired(img):
    desired_size = 96
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
    return transforms.functional.pad(img, padding)

def get_data_loader(data=data, targets=targets, img_transform=img_transform, label_transform=label_transform, mode="train"):
    dataset = CustomDataset(data, targets, img_transform, label_transform, mode)
    # print the shape of the first image
    print(dataset[0][0].shape)
    # plot the first image and its target
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(dataset[0][0].permute(1, 2, 0))
    ax[0].set_title("Data")
    ax[1].imshow(dataset[0][1].permute(1, 2, 0))
    ax[1].set_title("Target")
    plt.show()
    return DataLoader(dataset, batch_size=32, shuffle=True)



def get_data(args):
    # train_dataset = datasets.ImageFolder(os.path.join(args['dataset_path'], args['train_folder']), transform=img_transform)
    train_dataset = CustomDataset(data, targets, img_transform, label_transform, "train")
    # only get part of the dataset
    train_dataset = torch.utils.data.Subset(train_dataset, indices=range(0, 3000))
    # val_dataset = datasets.ImageFolder(os.path.join(args.dataset_path, args.val_folder), transform=label_transform)
    
    # if args.slice_size>1:
    #     train_dataset = torch.utils.data.Subset(train_dataset, indices=range(0, len(train_dataset), args.slice_size))
    #     val_dataset = torch.utils.data.Subset(val_dataset, indices=range(0, len(val_dataset), args.slice_size))

    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])
    # val_dataset = DataLoader(val_dataset, batch_size=2*args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_dataloader