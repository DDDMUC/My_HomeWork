import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 设置确定性随机种子以保证结果重现性
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. 加载并清洗 10 维分类数据
# =====================================================================
def load_pytorch_dataset():
    df = pd.read_csv('data0515.csv')
    X_raw = df.iloc[:, 0:10].values
    y_raw = df['label'].values
    
    X_tensor = torch.tensor(X_raw, dtype=torch.float32)
    y_tensor = torch.tensor(y_raw, dtype=torch.long)
    return X_tensor, y_tensor

X_data, y_data = load_pytorch_dataset()

# =====================================================================
# 2. 实验 (1) 结构设计: BatchNorm 对比
# =====================================================================

class NetWithoutBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
    def forward(self, x):
        return self.net(x)

class NetWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.BatchNorm1d(64), # 引入一维批归一化
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
    def forward(self, x):
        return self.net(x)

def run_experiment_1():
    print("\n=== 正在启动实验 (1): BatchNorm 对比测试 ===")
    epochs = 120
    lr = 0.01
    
    # 1. 训练不带 BN 的网络
    model_no_bn = NetWithoutBN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_no_bn.parameters(), lr=lr)
    loss_no_bn = []
    
    for epoch in range(epochs):
        model_no_bn.train()
        optimizer.zero_grad()
        outputs = model_no_bn(X_data)
        loss = criterion(outputs, y_data)
        loss.backward()
        optimizer.step()
        loss_no_bn.append(loss.item())
        
    model_no_bn.eval()
    with torch.no_grad():
        preds = model_no_bn(X_data).argmax(dim=1)
        acc_no_bn = (preds == y_data).float().mean().item()
    print(f"[+] 无 BN 层模型训练结束 | 最终 Loss: {loss_no_bn[-1]:.4f} | 最终 Accuracy: {acc_no_bn * 100:.2f}%")

    # 2. 训练带 BN 的网络
    model_with_bn = NetWithBN()
    optimizer_bn = optim.SGD(model_with_bn.parameters(), lr=lr)
    loss_with_bn = []
    
    for epoch in range(epochs):
        model_with_bn.train()
        optimizer_bn.zero_grad()
        outputs = model_with_bn(X_data)
        loss = criterion(outputs, y_data)
        loss.backward()
        optimizer_bn.step()
        loss_with_bn.append(loss.item())
        
    model_with_bn.eval()
    with torch.no_grad():
        preds_bn = model_with_bn(X_data).argmax(dim=1)
        acc_with_bn = (preds_bn == y_data).float().mean().item()
    print(f"[+] 带 BN 层模型训练结束 | 最终 Loss: {loss_with_bn[-1]:.4f} | 最终 Accuracy: {acc_with_bn * 100:.2f}%")

    # 绘制实验 (1) 图像
    plt.figure(figsize=(8, 5))
    plt.plot(loss_no_bn, label='Without BatchNorm', color='red', linestyle='--')
    plt.plot(loss_with_bn, label='With BatchNorm', color='green')
    plt.title('Experiment 1: Loss Curve Comparison (With vs. Without BN)')
    plt.xlabel('Epochs')
    plt.ylabel('CrossEntropy Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('experiment_1_bn_comparison.png')
    plt.show()

# =====================================================================
# 3. 实验 (2) 结构设计: 损失方案对比 (CE vs. MSE)
# =====================================================================

class NetSingleOutput(nn.Module):
    """
    方案 ② 所需模型：最末层仅输出一个浮点数节点
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # 输出 1 个连续浮点数
        )
    def forward(self, x):
        return self.net(x)

def run_experiment_2():
    print("\n=== 正在启动实验 (2): Softmax+CE 与 单输出+MSE 对比测试 ===")
    epochs = 150
    lr = 0.005
    
    # ---- 方案 ①: Softmax + CrossEntropy ----
    model_ce = NetWithoutBN()
    criterion_ce = nn.CrossEntropyLoss()
    optimizer_ce = optim.Adam(model_ce.parameters(), lr=lr)
    
    loss_ce_history = []
    acc_ce_history = []
    
    for epoch in range(epochs):
        model_ce.train()
        optimizer_ce.zero_grad()
        outputs = model_ce(X_data)
        loss = criterion_ce(outputs, y_data)
        loss.backward()
        optimizer_ce.step()
        
        loss_ce_history.append(loss.item())
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            acc = (preds == y_data).float().mean().item()
            acc_ce_history.append(acc)

    # ---- 方案 ②: 单神经元 + MSE ----
    model_mse = NetSingleOutput()
    criterion_mse = nn.MSELoss()
    optimizer_mse = optim.Adam(model_mse.parameters(), lr=lr)
    
    loss_mse_history = []
    acc_mse_history = []
    
    # 标签转换为浮点回归目标形式 (shape: N x 1)
    y_target_float = y_data.float().unsqueeze(1)
    
    for epoch in range(epochs):
        model_mse.train()
        optimizer_mse.zero_grad()
        outputs = model_mse(X_data) # 输出为 float
        loss = criterion_mse(outputs, y_target_float)
        loss.backward()
        optimizer_mse.step()
        
        loss_mse_history.append(loss.item())
        with torch.no_grad():
            # 评估方案：将输出的浮点数四舍五入到最近的整数，并限制在 0-9 范围
            preds_rounded = torch.clamp(torch.round(outputs), 0, 9).long()
            acc = (preds_rounded.squeeze() == y_data).float().mean().item()
            acc_mse_history.append(acc)

    # 绘制实验 (2) 图 1: Loss 比较图
    plt.figure(figsize=(8, 5))
    plt.plot(loss_ce_history, label='Scheme 1: Softmax + CrossEntropy', color='blue')
    plt.plot(loss_mse_history, label='Scheme 2: Single Neuron + MSE', color='purple', linestyle='--')
    plt.title('Experiment 2: Loss Curves Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.grid(True)
    plt.legend()
    plt.savefig('experiment_2_loss_comparison.png')
    plt.show()

    # 绘制实验 (2) 图 2: 准确率(随时间变化) 比较图
    plt.figure(figsize=(8, 5))
    plt.plot(acc_ce_history, label='Scheme 1 Accuracy', color='blue')
    plt.plot(acc_mse_history, label='Scheme 2 Accuracy', color='purple', linestyle='--')
    plt.title('Experiment 2: Accuracy Curves Comparison (Over Epochs)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('experiment_2_accuracy_comparison.png')
    plt.show()

if __name__ == "__main__":
    run_experiment_1()
    run_experiment_2()