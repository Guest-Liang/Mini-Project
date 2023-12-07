import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
from torch import nn
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载GENKI数据集
def load_genki_data():
    images = []
    labels = []
    poses = []

    with open('./Mini Project/genki4k/labels.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            img_path = f'./Mini Project/genki4k/files/file{i+1:04}.jpg'
            img = cv2.imread(img_path)
            img = cv2.resize(img, (256, 256))
            images.append(img)

            # 解析标签信息
            label, yaw, pitch, roll = map(float, line.strip().split())
            labels.append(int(label))
            poses.append([yaw, pitch, roll])

    images = np.array(images).transpose((0, 3, 1, 2))  # 转换为(C, H, W)格式
    # images = np.array(images)
    labels = np.array(labels)
    poses = np.array(poses)

    return images, labels, poses


images, labels, poses = load_genki_data()
torch.manual_seed(42)
train_images, test_images = random_split(images, [3200,800])
train_labels, test_labels = random_split(labels, [3200,800])

# 转换为 Tensor
train_images = torch.stack([torch.from_numpy(item[0]) for item in train_images])
train_labels = torch.tensor([item for item in train_labels])
test_images = torch.stack([torch.from_numpy(item[0]) for item in test_images])
test_labels = torch.tensor([item for item in test_labels])
# 创建DataLoader
train_data = TensorDataset(train_images, train_labels)
test_data = TensorDataset(test_images, test_labels)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # C1层 MNIST数据集的图像大小是28x28，填充2
        self.pool1 = nn.AvgPool2d(2, stride=2)  # S2层
        self.conv2 = nn.Conv2d(6, 16, 5)  # C3层
        self.pool2 = nn.AvgPool2d(2, stride=2)  # S4层
        self.fc1 = nn.Linear(16*5*5, 120)  # C5层
        self.fc2 = nn.Linear(120, 84)  # F6层
        self.fc3 = nn.Linear(84, 10)  # 输出层

    def forward(self, x):
        x = F.relu(self.conv1(x))  # C1层
        x = self.pool1(x)  # S2层
        x = F.relu(self.conv2(x))  # C3层
        x = self.pool2(x)  # S4层
        x = x.view(-1, 16*5*5)  # 展平
        x = F.relu(self.fc1(x))  # C5层
        x = F.relu(self.fc2(x))  # F6层
        x = self.fc3(x)  # 输出层
        return x

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU())
        self.d = nn.Sequential(nn.Linear(32*28*28, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.d(x)
        return x

# # Create an instance of the model
# model = MyModel()
# model = model.to(device)

# Create an instance of the model
model= LeNet5()
model = model.to(device)

loss_object = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

losses = []

def train(epoch, device):
    running_loss = 0.0  # 这整个epoch的loss清零
    running_total = 0
    running_correct = 0
    for batch_idx, (data, target) in enumerate(train_data, 0):
        inputs, target = data.to(device), target.to(device)
        inputs = inputs.float()
        target = target.long()
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = loss_object(outputs, target)

        loss.backward()
        optimizer.step()

        # 把运行中的loss累加起来，为了下面300次一除
        running_loss += loss.item()
        # 把运行中的准确率acc算出来
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 300 == 299:  # 不想要每一次都出loss，浪费时间，选择每300次出一个平均损失,和准确率
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0  # 这小批300的loss清零
            running_total = 0
            running_correct = 0  # 这小批300的acc清零
    avg_loss = running_loss / running_total
    losses.append(avg_loss) # 画loss图用

def test(epoch,device):
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data, target in test_data:
            images, labels = data.to(device), target.to(device)
            images = images.float()
            labels = labels.long()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            total += labels.size(0)  # 张量之间的比较运算
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch+1, EPOCHS, 100 * acc))  # 求测试的准确率，正确数/总数
    return acc


EPOCHS = 10


for epoch in range(EPOCHS):
    model.train()
    train(epoch, device)
    
model.eval()
test(epoch, device)