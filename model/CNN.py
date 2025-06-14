import torch
from torch import nn

class MnistCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MnistCNN, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # 第二层卷积
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # 全连接层
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)


    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 展平特征层
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def train_model(self, train_loader, optimizer, criterion, device, epoch):
        self.train()
        total_loss = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += output.argmax(dim=1).eq(target).sum().item()

        avg_loss = total_loss / len(train_loader)
        acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch}: Train loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        return acc

    def evaluate_model(self, test_loader, criterion, device):
        self.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                test_loss += criterion(output, target).item()
                correct += output.argmax(dim=1).eq(target).sum().item()

        avg_loss = test_loss / len(test_loader)
        acc = correct / len(test_loader.dataset)
        print(f"Eval loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        return acc

