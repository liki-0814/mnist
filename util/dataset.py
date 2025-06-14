from torch.utils.data import Dataset
import torch

class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 将图像转换为PyTorch张量
        image = torch.tensor(image, dtype=torch.float32)

        # 标准化图像 (MNIST标准化参数)
        image = (image - 0.1307) / 0.3081

        # 添加通道维度 (灰度图像)
        image = image.unsqueeze(0)  # 从 [28, 28] 变为 [1, 28, 28]

        return image, label