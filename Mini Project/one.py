import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, mean_squared_error
import os
from PIL import Image
import datetime


# 参数-------------------------------------
epochs = 10
isSaveModel = True
# 参数-------------------------------------


# 1. 设置GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2. 定义预训练的模型
# resnet18
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 2),
        )
        
    def forward(self, x):
        x = self.resnet(x)
        return self.classifier(x)

# Alexnet
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        # 冻结预训练模型的所有层
        for param in self.alexnet.parameters():
            param.requires_grad = False
        num_ftrs = self.alexnet.classifier[6].in_features
        self.alexnet.classifier[6] = nn.Linear(num_ftrs, 2)  # 替换全连接层

    def forward(self, x):
        x = self.alexnet(x)
        return x

# VGG11
class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.vgg = torchvision.models.vgg11(pretrained=True)
        # 冻结预训练模型的所有层
        # for param in self.vgg.parameters():
        #     param.requires_grad = False
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(num_ftrs, 2)  # 替换全连接层

    def forward(self, x):
        x = self.vgg(x)
        return x

# VGG16
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        # 冻结预训练模型的所有层
        for param in self.vgg.parameters():
            param.requires_grad = False
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(num_ftrs, 2)  # 替换全连接层

    def forward(self, x):
        x = self.vgg(x)
        return x



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
        label_smile = self.labels[idx][0]
        label_pose = torch.tensor([self.labels[idx][1], self.labels[idx][2], self.labels[idx][3]])

        if self.transform:
            image = self.transform(image)

        return image, label_smile, label_pose

transform = transforms.Compose([
    transforms.Lambda(lambda image: image.convert('RGB')),  # 将所有图像转换为RGB图像
    transforms.Resize((224, 224)),  # 将所有图像调整为224x224
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对每个通道进行归一化  
])


# 5. 划分训练集、验证集和测试集
dataset = Genki4kDataset(dataset_path, labels, transform)
total_size = len(dataset)  # 数据集大小
train_size = int(0.8 * total_size)  # 训练集
val_size = int(0.1 * total_size)    # 验证集
test_size = total_size - train_size - val_size  # 剩余部分为测试集
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# 6. 训练模型
# model = VGGNet().to(device)
model = ResNet18().to(device)
# model = AlexNet().to(device)
# model = VGG11().to(device)
# model = VGG16().to(device)
cla_loss = nn.CrossEntropyLoss()

# 使用二元交叉熵损失
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) 

for epoch in range(epochs):  # loop over the dataset multiple times
    model.train()
    print("Epoch:", epoch+1, "start training...")
    for i, data in enumerate(train_loader, 0):
        inputs, labels_smile = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs_smile = model(inputs)
        loss_cla = cla_loss(outputs_smile, labels_smile.long())
        # loss_cla = criterion(outputs_smile, labels_smile.float().view(-1, 1)) # 单输出问题
        loss_cla.backward()
        optimizer.step()
        
        # 计算精度
        preds_smile = torch.argmax(outputs_smile, dim=1)
        # acc_smile = accuracy_score(labels_smile.cpu(), preds_smile.cpu())
        acc_smile = torch.eq(preds_smile, labels_smile).sum().item() / labels_smile.size(0)
        print(f'\rEpoch {epoch+1}, Batch {i+1}, Loss(smile): {loss_cla.item():.4f}, Accuracy(smile): {acc_smile:.4f}',end='')
    
    scheduler.step()
    # 在验证集上验证
    model.eval()
    val_loss_cla = 0
    val_acc_smile = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels_smile = data[0].to(device), data[1].to(device)
            outputs_smile = model(inputs)
            loss_cla = cla_loss(outputs_smile, labels_smile.long())
            # loss_cla = criterion(outputs_smile, labels_smile.float().view(-1, 1)) # 单输出问题
            val_loss_cla += loss_cla.item()
            preds_smile = torch.argmax(outputs_smile, dim=1)
            val_acc_smile += accuracy_score(labels_smile.cpu(), preds_smile.cpu())
    val_loss_cla /= len(val_loader)
    val_acc_smile /= len(val_loader)
    print(f'\nEpoch {epoch+1}, Validation Loss(smile): {val_loss_cla:.4f}, Validation Accuracy(smile): {val_acc_smile:.4f}')

# 7. 评估模型
# 在测试集上测试
model.eval()
test_loss_cla = 0
test_acc_smile = 0
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, labels_smile= data[0].to(device), data[1].to(device)
        outputs_smile = model(inputs)
        loss_cla = cla_loss(outputs_smile, labels_smile.long())
        # loss_cla = criterion(outputs_smile, labels_smile.float().view(-1, 1)) # 单输出问题
        test_loss_cla += loss_cla.item()
        preds_smile = torch.argmax(outputs_smile, dim=1)
        test_acc_smile += accuracy_score(labels_smile.cpu(), preds_smile.cpu())
test_loss_cla /= len(test_loader)
test_acc_smile /= len(test_loader)
print(f'\nTest Loss(smile): {test_loss_cla:.4f}, Test Accuracy(smile): {test_acc_smile:.4f}')

# 8. 保存模型
if isSaveModel:
    save_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # 保存模型参数
    torch.save(model.state_dict(), f'./model__cla_parameters_{save_time}.pt')
    print("Model parameters saved.")
    # 保存整个模型
    torch.save(model, f'./model_cla_{save_time}.pt')
    print("Model saved.")