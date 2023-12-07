import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from PIL import Image
import datetime



# 参数-------------------------------------
epochs=15
isSaveModel = False
# 参数-------------------------------------



# 1. 设置GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2. 定义VGGNet模型
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 3. 加载genki4k数据集
# 请根据实际情况修改数据集路径
dataset_path = './Mini Project/genki4k/'
labels = []
with open(os.path.join(dataset_path, 'labels.txt'), 'r') as file:
    lines = file.readlines()
    for line in lines:
        labels.append(int(line.split()[0]))  # 只取第一个标签

# 4. 对数据集进行预处理
class Genki4kDataset(Dataset):
    def __init__(self, img_dir, labels, transform=None):
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, 'files', f'file{idx+1:04}.jpg')  # 图片名称格式为file0001.jpg, file0002.jpg, ...
        image = Image.open(img_name)
        image = image.resize((128, 128))
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 将所有图像转换为灰度图像
    transforms.Resize((128, 128)),  # 将所有图像调整为128x128
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# 5. 划分训练集、验证集和测试集
train_data, val_test_data, train_labels, val_test_labels = train_test_split(list(range(len(labels))), labels, test_size=0.4)
val_data, test_data, val_labels, test_labels = train_test_split(val_test_data, val_test_labels, test_size=0.5)

train_dataset = Genki4kDataset(dataset_path, train_labels, transform)
val_dataset = Genki4kDataset(dataset_path, val_labels, transform)
test_dataset = Genki4kDataset(dataset_path, test_labels, transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 6. 训练模型
model = VGGNet()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):  # loop over the dataset multiple times
    model.train()
    print("Epoch: ", epoch+1, "start training...")
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # 计算精度
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        print(f'\rEpoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}, Accuracy: {acc}',end='')

    # 在验证集上验证
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_acc += accuracy_score(labels.cpu(), preds.cpu())
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    print(f'\nEpoch {epoch+1}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')

# # 7. 评估模型
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in test_loader:
#         images, labels = data
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# 7. 评估模型
# 在测试集上测试
model.eval()
test_loss = 0
test_acc = 0
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        test_acc += accuracy_score(labels.cpu(), preds.cpu())
test_loss /= len(test_loader)
test_acc /= len(test_loader)
print(f'\nTest Loss: {test_loss}, Test Accuracy: {test_acc}')

# 8. 保存模型
if isSaveModel:
    save_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # 保存模型参数
    torch.save(model.state_dict(), f'./model_parameters_{save_time}.pt')
    print("Model parameters saved.")
    # 保存整个模型
    torch.save(model, f'./model_{save_time}.pt')
    print("Model saved.")