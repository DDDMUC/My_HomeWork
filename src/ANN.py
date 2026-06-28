# src/ANN.py
import torch
import torch.nn as nn

# =====================================================================
# 1. 房价预测网络 (回归任务：输入4维 -> 输出1维)
# =====================================================================
class HouseANN(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 回归任务输出层不加激活函数，输出连续实数
        )
    def forward(self, x):
        return self.net(x)

# =====================================================================
# 2. 泰坦尼克号存活预测网络 (二分类任务：输入7维 -> 输出2维)
# =====================================================================
class TitanicANN(nn.Module):
    def __init__(self, input_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 输出 2 维 logits，配合 CrossEntropyLoss 训练
        )
    def forward(self, x):
        return self.net(x)

# =====================================================================
# 3. CIFAR-10 图像分类网络 (多分类任务：输入3072维 -> 输出10维)
# =====================================================================
class Cifar10ANN(nn.Module):
    def __init__(self, input_dim=3072):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10) # 输出 10 维，对应 10 个图像类别
        )
    def forward(self, x):
        # 在前向传播中自动将三维图像张量 [Batch, 3, 32, 32] 展平为一维向量 [Batch, 3072]
        x = x.view(x.size(0), -1)
        return self.net(x)