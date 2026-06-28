# run_rf_importance.py
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.data_preprocess import clean_dirty_beans

# 1. 载入并清洗数据
train_path = "data/dry_bean/Dry_Bean_Dataset_Dirty_train.csv"
val_path = "data/dry_bean/Dry_Bean_Dataset_Dirty_val.csv"
test_path = "data/dry_bean/Dry_Bean_Dataset_Dirty_test.csv"

# 确保结果保存文件夹存在
os.makedirs("results", exist_ok=True)

if not os.path.exists(train_path):
    print(f"[-] 错误: 无法找到训练数据，请确认已将数据放入 {train_path}")
    exit()

X_train, y_train, X_val, y_val, X_test, y_test, feature_names, classes = clean_dirty_beans(
    train_path, val_path, test_path
)

print("[*] 数据清洗完成，正在训练随机森林以提取特征重要性...")

# 2. 训练随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 3. 提取特征重要性（基尼不纯度减少量）
importances = rf.feature_importances_
indices = np.argsort(importances) # 升序排列用于横向柱状图

# 4. 绘制特征重要性条形图
plt.figure(figsize=(10, 6))
plt.title("Dry Bean Feature Importance Analysis (Random Forest Baseline)", fontsize=12, fontweight='bold')
plt.barh(range(X_train.shape[1]), importances[indices], color='#2980B9', align='center')
plt.yticks(range(X_train.shape[1]), [feature_names[i] for i in indices], fontsize=9)
plt.xlabel("Relative Importance (Mean Decrease in Impurity)", fontsize=10)
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 保存图像
plot_path = "results/dry_bean_feature_importance.png"
plt.savefig(plot_path, dpi=300)
print(f"[+] 特征重要性条形图已成功保存至: {plot_path}")
plt.show()