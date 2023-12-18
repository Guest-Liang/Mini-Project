import torch.nn as nn
import torchvision

class VGG11Binary(nn.Module):
    def __init__(self):
        super(VGG11Binary, self).__init__()
        # 加载预训练的 VGG11 模型
        self.vgg = torchvision.models.vgg11(pretrained=True)
        
        # 冻结预训练模型的所有层
        for param in self.vgg.parameters():
            param.requires_grad = False

        # 修改全连接层
        num_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(num_features, 1)  # 修改为二分类任务，输出节点为1

    def forward(self, x):
        x = self.vgg(x)
        return x

# 创建 VGG11Binary 模型实例
model = VGG11Binary()

# 打印模型结构
print(model)
