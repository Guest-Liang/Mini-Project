import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, mean_squared_error
import os
from PIL import Image
import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# 参数-------------------------------------
epochs = 10
isSaveModel = False
# 参数-------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class AlexNet2(nn.Module):
    def __init__(self):
        super(AlexNet2, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        # 冻结预训练模型的所有层
        for param in self.alexnet.parameters():
            param.requires_grad = False
        num_ftrs = self.alexnet.classifier[6].in_features
        # 替换全连接层
        self.alexnet.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.alexnet(x)
        return x


dataset_path = './Mini Project/genki4k/'
labels = []
with open(os.path.join(dataset_path, 'label-sex.txt'), 'r') as file:
    lines = file.readlines()
    for line in lines:
        items = line.split()
        label = [int(items[0])]  # 第一个是整数，1是男
        labels.append(label)

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

        if self.transform:
            image = self.transform(image)

        return image, label_smile

transform = transforms.Compose([
    transforms.Lambda(lambda image: image.convert('RGB')),  # 将所有图像转换为RGB图像
    transforms.Resize((224, 224)),  # 将所有图像调整为224x224
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对每个通道进行归一化  
])

dataset = Genki4kDataset(dataset_path, labels, transform)
total_size = len(dataset)  # 数据集大小
train_size = int(0.8 * total_size)  # 训练集
val_size = int(0.1 * total_size)    # 验证集
test_size = total_size - train_size - val_size  # 剩余部分为测试集
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)



def TrainFunc2(model):
    cla_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) 

    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        print("Epoch", epoch+1, "start training...")
        for i, data in enumerate(train_loader, 0):
            inputs, labels_gender = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs_gender = model(inputs)
            loss_cla = cla_loss(outputs_gender, labels_gender.long())
            loss_cla.backward()
            optimizer.step()
            
            # 计算精度
            preds_gender = torch.argmax(outputs_gender, dim=1)
            acc_gender = torch.eq(preds_gender, labels_gender).sum().item() / labels_gender.size(0)
            print(f'\rEpoch {epoch+1}, Batch {i+1}, Loss(gender): {loss_cla.item():.4f}, Accuracy(gender): {acc_gender:.4f}',end='')
        
        scheduler.step()
        # 在验证集上验证
        model.eval()
        val_loss_cla = 0
        val_acc_gender = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels_gender = data[0].to(device), data[1].to(device)
                outputs_gender = model(inputs)
                loss_cla = cla_loss(outputs_gender, labels_gender.long())
                val_loss_cla += loss_cla.item()
                preds_gender = torch.argmax(outputs_gender, dim=1)
                val_acc_gender += accuracy_score(labels_gender.cpu(), preds_gender.cpu())
        val_loss_cla /= len(val_loader)
        val_acc_gender /= len(val_loader)
        print(f'\nEpoch {epoch+1}, Validation Loss(gender): {val_loss_cla:.4f}, Validation Accuracy(gender): {val_acc_gender:.4f}')

    # 7. 评估模型
    # 在测试集上测试
    model.eval()
    test_loss_cla = 0
    test_acc_gender = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels_gender= data[0].to(device), data[1].to(device)
            outputs_gender = model(inputs)
            loss_cla = cla_loss(outputs_gender, labels_gender.long())
            test_loss_cla += loss_cla.item()
            preds_gender = torch.argmax(outputs_gender, dim=1)
            test_acc_gender += accuracy_score(labels_gender.cpu(), preds_gender.cpu())
    test_loss_cla /= len(test_loader)
    test_acc_gender /= len(test_loader)
    print(f'\nTest Loss(gender): {test_loss_cla:.4f}, Test Accuracy(gender): {test_acc_gender:.4f}')

model4 = AlexNet2().to(device)
print("Training model4...")
TrainFunc2(model4)