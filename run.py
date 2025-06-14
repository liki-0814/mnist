from os.path import join

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from util import *
from model import *


def load_data():
    # Set file paths based on added MNIST Datasets
    input_path = './dataset'
    training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (x_train_temp, y_train_temp), (x_test_temp, y_test_temp) = mnist_dataloader.load_data()
    return (x_train_temp, y_train_temp), (x_test_temp, y_test_temp)



if __name__ == '__main__':
    # load data
    (x_train, y_train), (x_test, y_test) = load_data()
    train_data = MNISTDataset(x_train, y_train)
    test_data = MNISTDataset(x_test, y_test)

    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # 初始化模型
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MnistCNN().to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)

    # 训练过程
    for epoch in range(1, 11):  # 训练 10 个 epoch
        train_acc = model.train_model(train_loader, optimizer, criterion, device, epoch)
        val_acc = model.evaluate_model(test_loader, criterion, device)
        scheduler.step(val_acc)

    # 保存模型
    torch.save(model.state_dict(), "./offline_model/cnn_mnist.pth")
    print("模型已保存为 mnist_cnn.pth")







