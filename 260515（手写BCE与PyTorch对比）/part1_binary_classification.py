import os
import math
import matplotlib.pyplot as plt

# =====================================================================
# 1. 矩阵与向量基础运算引擎（无第三方库）
# =====================================================================

def mat_mul_vec(matrix, vector):
    return [sum(m_ij * v_j for m_ij, v_j in zip(row, vector)) for row in matrix]

def vec_add(vec1, vec2):
    return [v1 + v2 for v1, v2 in zip(vec1, vec2)]

def relu(vector):
    return [max(0.0, x) for x in vector]

def relu_derivative(z_vector, grad_vector):
    return [g if z > 0 else 0.0 for z, g in zip(z_vector, grad_vector)]

def sigmoid(x):
    # 防止指数溢出
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# =====================================================================
# 2. 原生数据加载与清洗
# =====================================================================

def load_classification_data():
    csv_filename = "分类.csv"
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"[-] 错误: 未能在当前路径找到 '{csv_filename}'")
        
    data = []
    with open(csv_filename, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('('): # 跳过表头
                continue
            parts = line.split(',')
            # 跳过空行或分割线占位符如 ",,,,,"
            if all(p == '' for p in parts) or len(parts) < 6:
                continue
            try:
                vals = [float(p) for p in parts]
                # 前5列为 X，最后一列为 y
                data.append({"x": vals[0:5], "y": int(vals[5])})
            except ValueError:
                continue
    return data


# =====================================================================
# 3. 二分类神经网络类
# =====================================================================

class HandwrittenClassifier:
    def __init__(self):
        # 确定性初始化参数
        import random
        random.seed(42)
        self.W1 = [[random.gauss(0, 0.2) for _ in range(5)] for _ in range(4)] # 4x5
        self.b1 = [0.1 for _ in range(4)]
        self.W2 = [[random.gauss(0, 0.2) for _ in range(4)] for _ in range(3)] # 3x4
        self.b2 = [0.1 for _ in range(3)]
        self.W3 = [[random.gauss(0, 0.2) for _ in range(3)]] # 1x3
        self.b3 = [0.1]

    def forward(self, x):
        z1 = vec_add(mat_mul_vec(self.W1, x), self.b1)
        a1 = relu(z1)
        z2 = vec_add(mat_mul_vec(self.W2, a1), self.b2)
        a2 = relu(z2)
        z3_val = sum(w * a for w, a in zip(self.W3[0], a2)) + self.b3[0]
        a3_val = sigmoid(z3_val) # 二分类核心：引入 Sigmoid 压缩到 (0, 1)
        
        self.state = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'a3': a3_val}
        return a3_val

    def backward(self, y_true, lr=0.1):
        x = self.state['x']
        z1, a1 = self.state['z1'], self.state['a1']
        z2, a2 = self.state['z2'], self.state['a2']
        a3_val = self.state['a3']

        # 1. 计算输出层误差项 (BCE Loss 结合 Sigmoid 的偏导极简形式)
        dz3 = a3_val - y_true
        dW3 = [[dz3 * a_j for a_j in a2]]
        db3 = [dz3]

        # 2. 隐藏层 2
        da2 = [self.W3[0][j] * dz3 for j in range(3)]
        dz2 = relu_derivative(z2, da2)
        dW2 = [[dz2[i] * a_j for a_j in a1] for i in range(3)]
        db2 = list(dz2)

        # 3. 隐藏层 1
        da1 = [sum(self.W2[i][j] * dz2[i] for i in range(3)) for j in range(4)]
        dz1 = relu_derivative(z1, da1)
        dW1 = [[dz1[i] * x_j for x_j in x] for i in range(4)]
        db1 = list(dz1)

        # 4. 参数更新 (梯度下降)
        for i in range(4):
            for j in range(5):
                self.W1[i][j] -= lr * dW1[i][j]
            self.b1[i] -= lr * db1[i]

        for i in range(3):
            for j in range(4):
                self.W2[i][j] -= lr * dW2[i][j]
            self.b2[i] -= lr * db2[i]

        for j in range(3):
            self.W3[0][j] -= lr * dW3[0][j]
        self.b3[0] -= lr * db3[0]


# =====================================================================
# 4. 主训练循环与可视化
# =====================================================================

if __name__ == "__main__":
    dataset = load_classification_data()
    model = HandwrittenClassifier()
    
    epochs = 150
    lr = 0.05
    loss_history = []
    acc_history = []
    
    print("[*] 开始手写神经网络二分类迭代训练...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_preds = 0
        
        # 逐样本在线更新
        for sample in dataset:
            x, y = sample["x"], sample["y"]
            y_pred = model.forward(x)
            
            # 计算 Binary Cross Entropy 损失
            loss_val = - (y * math.log(max(y_pred, 1e-15)) + (1 - y) * math.log(max(1 - y_pred, 1e-15)))
            epoch_loss += loss_val
            
            # 记录准确数
            pred_label = 1 if y_pred >= 0.5 else 0
            if pred_label == y:
                correct_preds += 1
                
            model.backward(y, lr=lr)
            
        avg_loss = epoch_loss / len(dataset)
        accuracy = correct_preds / len(dataset)
        loss_history.append(avg_loss)
        acc_history.append(accuracy)
        
        if (epoch + 1) % 15 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Avg Loss: {avg_loss:.5f} | Accuracy: {accuracy*100:.2f}%")

    print("[+] 训练完成。")
    
    # 绘制双指标监控曲线
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('BCE Loss', color=color)
    ax1.plot(loss_history, color=color, label='BCE Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(acc_history, color=color, linestyle='--', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Training Loss and Accuracy Curves (Binary Classification)')
    fig.tight_layout()
    plt.grid(True)
    plt.show()