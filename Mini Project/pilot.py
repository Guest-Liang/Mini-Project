import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import os
from PIL import Image
import datetime



# 参数-------------------------------------
epochs = 100
isSaveModel = True
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
        self.regression = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        c = self.classifier(x)
        r = self.regression(x)
        # print(c.shape,type(c))  # 打印 c 的形状、类型
        # print(r.shape,type(r))  # 打印 r 的形状、类型
        return c, r

# 预训练的resnet18模型
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # 移除原始的全连接层

        # 二分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 2),
        )

        # 3D姿态预测器
        self.regression = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        x = self.resnet(x)
        return self.classifier(x), self.regression(x)



# 3. 加载genki4k数据集
# 请根据实际情况修改数据集路径
dataset_path = './Mini Project/genki4k/'
labels = []
with open(os.path.join(dataset_path, 'labels.txt'), 'r') as file:
    lines = file.readlines()
    for line in lines:
        items = line.split()
        label = [int(items[0])] + [float(x) for x in items[1:]]  # 第一个是整数，后面三个是浮点数
        labels.append(label)

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
        # image = image.resize((128, 128))
        image = image.resize((224, 224))
        label_smile = self.labels[idx][0]
        label_pose = torch.tensor([self.labels[idx][1], self.labels[idx][2], self.labels[idx][3]])

        if self.transform:
            image = self.transform(image)

        return image, label_smile, label_pose

transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),  # 将所有图像转换为灰度图像
    # # transforms.Resize((128, 128)),  # 将所有图像调整为128x128
    # transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))

    transforms.Lambda(lambda image: image.convert('RGB')),  # 将所有图像转换为RGB图像
    transforms.Resize((224, 224)),  # 将所有图像调整为224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对每个通道进行归一化  
])


# 5. 划分训练集、验证集和测试集
train_data, val_test_data, train_labels, val_test_labels = train_test_split(list(range(len(labels))), labels, test_size=0.4)
val_data, test_data, val_labels, test_labels = train_test_split(val_test_data, val_test_labels, test_size=0.5)

# train_dataset = Genki4kDataset(dataset_path, train_labels, transform)
# val_dataset = Genki4kDataset(dataset_path, val_labels, transform)
# test_dataset = Genki4kDataset(dataset_path, test_labels, transform)
train_dataset = Genki4kDataset(dataset_path, [labels[i] for i in train_data], transform)
val_dataset = Genki4kDataset(dataset_path, [labels[i] for i in val_data], transform)
test_dataset = Genki4kDataset(dataset_path, [labels[i] for i in test_data], transform)

train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=60, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=60, shuffle=True)

# 6. 训练模型
# model = VGGNet()
model = ResNet18()
model = model.to(device)
cla_loss = nn.CrossEntropyLoss()
reg_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):  # loop over the dataset multiple times
    model.train()
    print("Epoch:", epoch+1, "start training...")
    for i, data in enumerate(train_loader, 0):
        inputs, labels_smile, labels_pose = data[0].to(device), data[1].to(device), data[2].to(device)
        optimizer.zero_grad()
        outputs_smile, outputs_pose = model(inputs)
        loss_cla = cla_loss(outputs_smile, labels_smile.long())
        loss_reg = reg_loss(outputs_pose, labels_pose.float())
        Loss = 1.95*loss_cla + 0.05*loss_reg
        Loss.backward()
        optimizer.step()
        
        # 计算精度
        preds_smile = torch.argmax(outputs_smile, dim=1)
        acc_smile = accuracy_score(labels_smile.cpu(), preds_smile.cpu())
        mse_pose = mean_squared_error(labels_pose.cpu().numpy(), outputs_pose.detach().cpu().numpy())
        print(f'\rEpoch {epoch+1}, Batch {i+1}, Loss(smile): {loss_cla.item():.4f}, Loss(pose): {loss_reg.item():.4f}, Accuracy(smile): {acc_smile:.4f}, MSE(pose): {mse_pose:.4f}',end='')

    # 在验证集上验证
    model.eval()
    val_loss_cla = 0
    val_loss_reg = 0
    val_acc_smile = 0
    val_mse_pose = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels_smile, labels_pose = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs_smile, outputs_pose = model(inputs)
            loss_cla = cla_loss(outputs_smile, labels_smile.long())
            loss_reg = reg_loss(outputs_pose, labels_pose.float())
            val_loss_cla += loss_cla.item()
            val_loss_reg += loss_reg.item()
            preds_smile = torch.argmax(outputs_smile, dim=1)
            val_acc_smile += accuracy_score(labels_smile.cpu(), preds_smile.cpu())
            val_mse_pose += mean_squared_error(labels_pose.cpu().numpy(), outputs_pose.detach().cpu().numpy())
    val_loss_cla /= len(val_loader)
    val_loss_reg /= len(val_loader)
    val_acc_smile /= len(val_loader)
    val_mse_pose /= len(val_loader)
    print(f'\nEpoch {epoch+1}, Validation Loss(smile): {val_loss_cla:.4f}, Validation Loss(pose): {val_loss_reg:.4f}, Validation Accuracy(smile): {val_acc_smile:.4f}, Validation MSE(pose): {val_mse_pose:.4f}')

# 7. 评估模型
# 在测试集上测试
model.eval()
test_loss_cla = 0
test_loss_reg = 0
test_acc_smile = 0
test_mse_pose = 0
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, labels_smile, labels_pose = data[0].to(device), data[1].to(device), data[2].to(device)
        outputs_smile, outputs_pose = model(inputs)
        loss_cla = cla_loss(outputs_smile, labels_smile.long())
        loss_reg = reg_loss(outputs_pose, labels_pose.float())
        test_loss_cla += loss_cla.item()
        test_loss_reg += loss_reg.item()
        preds_smile = torch.argmax(outputs_smile, dim=1)
        test_acc_smile += accuracy_score(labels_smile.cpu(), preds_smile.cpu())
        test_mse_pose += mean_squared_error(labels_pose.cpu().numpy(), outputs_pose.detach().cpu().numpy())
test_loss_cla /= len(test_loader)
test_loss_reg /= len(test_loader)
test_acc_smile /= len(test_loader)
test_mse_pose /= len(test_loader)
print(f'\nTest Loss(smile): {test_loss_cla:.4f}, Test Loss(pose): {test_loss_reg:.4f}, Test Accuracy(smile): {test_acc_smile:.4f}, Test MSE(pose): {test_mse_pose:.4f}')

# 8. 保存模型
if isSaveModel:
    save_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # 保存模型参数
    torch.save(model.state_dict(), f'./model_parameters_{save_time}.pt')
    print("Model parameters saved.")
    # 保存整个模型
    torch.save(model, f'./model_{save_time}.pt')
    print("Model saved.")