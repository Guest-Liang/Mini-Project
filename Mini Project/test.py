import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cv2
import numpy as np
import datetime
import os, copy, glob
from sklearn.model_selection import train_test_split

# 调试参数
isLoadModel = False # 是否加载模型


# 设置GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 5

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

# 构建笑脸分类模型cnn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 64 * 64, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Lenet5
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = None  # 将全连接层初始化为None
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        if self.fc1 is None:  # 如果全连接层还没有被创建
            self.fc1 = nn.Linear(x.size(1), 120).to(x.device)  # 根据x的大小创建全连接层
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# 加载数据集
images, labels, poses = load_genki_data()
# array=np.random.randint(4000,size=5)
# for i in array:
#     np.random.seed(i)
#     cv2.imshow(f'{i}:Lables:{labels[i]}',images[i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# 划分训练集和测试集
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# 转换为Tensor
train_images = torch.from_numpy(train_images).float()
train_labels = torch.from_numpy(train_labels).long()
test_images = torch.from_numpy(test_images).float()
test_labels = torch.from_numpy(test_labels).long()

# 创建DataLoader
train_data = TensorDataset(train_images, train_labels)
test_data = TensorDataset(test_images, test_labels)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# model = Net().to(device) # CNN
model = LeNet5().to(device) # Lenet5
if isLoadModel:
    # 加载模型
    model_files = glob.glob('./Mini Project/model_*.pt') # 获取所有模型文件
    latest_model_file = max(model_files, key=os.path.getctime) # 找到最新的模型文件
    model.load_state_dict(torch.load(latest_model_file)) # 加载最新的模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)


# 初始化最高精度为0
best_acc = 0.0
# 初始化最好的模型
best_model = None

for epoch in range(epochs):
    running_loss = 0.0
    accuracies = []
    for i, data in enumerate(train_loader, 0):
        model.train()
        correct = 0
        total = 0
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracies.append(correct / total * 100)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f'\rBatch {i+1}, Loss: {loss.item():.4f}, Accuracy: {correct / total * 100:.4f}%',end='')
    print(f'\nEpoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}, Average Accuracy: {sum(accuracies) / len(accuracies):.4f}%')
    
    # 在测试集上计算精度
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # 计算精度
    acc = correct / total
    print(f'Test Accuracy: {acc * 100:.4f}%')
    
    # 如果精度更高，保存模型
    if acc > best_acc:
        best_acc = acc
        best_model = copy.deepcopy(model.state_dict())

# 保存最好的模型
save_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
torch.save(best_model, 'model_{save_time}.pt')
print(f'Test Accuracy: {correct / total * 100:.4f}%')