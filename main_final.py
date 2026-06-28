# main_final.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.data_preprocess import clean_dirty_beans

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================================================================
# 1. 用于大作业对比的 PyTorch MLP 模型定义
# =====================================================================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=16, num_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# =====================================================================
# 2. 统一评测与指标计算流水线
# =====================================================================
def run_project_evaluation():
    os.makedirs('results', exist_ok=True)
    
    # 1. 载入并清洗 Dry Bean 脏数据集
    train_path = "data/dry_bean/Dry_Bean_Dataset_Dirty_train.csv"
    val_path = "data/dry_bean/Dry_Bean_Dataset_Dirty_val.csv"
    test_path = "data/dry_bean/Dry_Bean_Dataset_Dirty_test.csv"
    
    # 确保数据集路径存在，否则自动友情提示
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"[-] 错误: 无法找到数据文件。请确保已将 3 个 CSV 文件放入文件夹 'data/dry_bean/' 下。")
        return
        
    X_train, y_train, X_val, y_val, X_test, y_test, _, classes = clean_dirty_beans(
        train_path, val_path, test_path
    )
    
    print("=" * 65)
    print("             ML_Project 期末大作业统一性能与鲁棒性评测系统")
    print("=" * 65)
    
    # ---- A. 训练自学算法：随机森林 (Random Forest) ----
    print("[*] 正在训练 随机森林 (自学算法)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # ---- B. 训练基础算法一：支持向量机 (SVM) ----
    print("[*] 正在训练 支持向量机 (SVM)...")
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
   # ---- C. 训练基础算法二：人工神经网络 (MLP) ----
    print("[*] 正在训练 多层感知机神经网络 (MLP)...")
    mlp = MLPClassifier(input_dim=16, num_classes=7).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.005)
    
    X_tr_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_tr_tensor = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    
    # 执行 100 轮拟合迭代
    for epoch in range(100):
        mlp.train()
        optimizer.zero_grad()
        outputs = mlp(X_tr_tensor)
        loss = criterion(outputs, y_tr_tensor)
        loss.backward()
        optimizer.step()
        
    print("[+] 所有算法训练完毕，开始执行多维度对比测算...")
    print("-" * 65)
    
    # 2. 测算过拟合程度（训练集准确率 vs 测试集准确率）
    rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
    rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
    
    svm_train_acc = accuracy_score(y_train, svm.predict(X_train))
    svm_test_acc = accuracy_score(y_test, svm.predict(X_test))
    
    mlp.eval()
    with torch.no_grad():
        mlp_tr_preds = mlp(X_tr_tensor).argmax(dim=1).cpu().numpy()
        mlp_train_acc = accuracy_score(y_train, mlp_tr_preds)
        
        X_te_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        mlp_te_preds = mlp(X_te_tensor).argmax(dim=1).cpu().numpy()
        mlp_test_acc = accuracy_score(y_test, mlp_te_preds)
        
    print("维度 1: 算法泛化与过拟合分析（准确率对比）:")
    print(f"  - 随机森林 : 训练集 Acc = {rf_train_acc*100:.2f}% | 测试集 Acc = {rf_test_acc*100:.2f}% | 差异 = {(rf_train_acc - rf_test_acc)*100:.2f}%")
    print(f"  - 支持向量 : 训练集 Acc = {svm_train_acc*100:.2f}% | 测试集 Acc = {svm_test_acc*100:.2f}% | 差异 = {(svm_train_acc - svm_test_acc)*100:.2f}%")
    print(f"  - 神经网络 : 训练集 Acc = {mlp_train_acc*100:.2f}% | 测试集 Acc = {mlp_test_acc*100:.2f}% | 差异 = {(mlp_train_acc - mlp_test_acc)*100:.2f}%")
    print("-" * 65)

    # 3. 测算推理速度 (计算预测 500 次的耗时毫秒数)
    print("维度 2: 算法在线推理速度（Inference Speed）对比:")
    X_speed_test = np.tile(X_test, (3, 1))[:500]
    
    t0 = time.time()
    for _ in range(10): _ = rf.predict(X_speed_test)
    rf_time = ((time.time() - t0) / 10.0) * 1000.0
    
    t0 = time.time()
    for _ in range(10): _ = svm.predict(X_speed_test)
    svm_time = ((time.time() - t0) / 10.0) * 1000.0
    
    X_speed_tensor = torch.tensor(X_speed_test, dtype=torch.float32).to(DEVICE)
    t0 = time.time()
    for _ in range(10):
        with torch.no_grad(): _ = mlp(X_speed_tensor)
    mlp_time = ((time.time() - t0) / 10.0) * 1000.0
    
    print(f"  - 随机森林 预测 500 条样本均时: {rf_time:.3f} 毫秒")
    print(f"  - 支持向量 预测 500 条样本均时: {svm_time:.3f} 毫秒")
    print(f"  - 神经网络 预测 500 条样本均时: {mlp_time:.3f} 毫秒")
    print("-" * 65)

    # 4. 算法抗噪鲁棒性加噪抗干扰对比测试
    print("维度 3: 算法抗噪鲁棒性（Robustness）测试中...")
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    
    rf_robustness = []
    svm_robustness = []
    mlp_robustness = []
    
    for sigma in noise_levels:
        X_test_noisy = X_test + np.random.normal(0, sigma, X_test.shape)
        
        rf_pred = rf.predict(X_test_noisy)
        rf_robustness.append(accuracy_score(y_test, rf_pred))
        
        svm_pred = svm.predict(X_test_noisy)
        svm_robustness.append(accuracy_score(y_test, svm_pred))
        
        X_noisy_tensor = torch.tensor(X_test_noisy, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            mlp_pred = mlp(X_noisy_tensor).argmax(dim=1).cpu().numpy()
        mlp_robustness.append(accuracy_score(y_test, mlp_pred))
        
    print("  - 鲁棒性加噪测试完成。正在生成测试曲线...")
    
    # 绘制鲁棒性学术衰减曲线图
    plt.figure(figsize=(9, 5))
    plt.plot(noise_levels, [x*100 for x in rf_robustness], marker='o', color='blue', label='Random Forest (Self-studied)')
    plt.plot(noise_levels, [x*100 for x in svm_robustness], marker='s', color='red', linestyle='--', label='SVM (RBF Kernel)')
    plt.plot(noise_levels, [x*100 for x in mlp_robustness], marker='^', color='green', linestyle=':', label='MLP (ANN)')
    
    plt.title('Algorithm Robustness Analysis: Accuracy Decay under Gaussian Noise', fontsize=11, fontweight='bold')
    plt.xlabel('Noise Standard Deviation (Sigma)', fontsize=10)
    plt.ylabel('Test Set Accuracy (%)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plot_path = 'results/dry_bean_robustness_comparison.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"[+] 算法抗噪衰减曲线图已成功保存至: {plot_path}")
    print("=" * 65)

if __name__ == "__main__":
    run_project_evaluation()