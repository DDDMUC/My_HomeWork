# test.py
import os
import pickle
from sklearn.metrics import accuracy_score
from src.data_preprocess import clean_titanic_data

def run_testing():
    print("[*] 正在加载并清洗泰坦尼克号测试数据...")
    _, _, X_test, y_test = clean_titanic_data(
        'data/titanic/titanic_train_knn.csv',
        'data/titanic/titanic_test_knn.csv'
    )
    
    model_path = 'models/svm_titanic_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[-] 无法找到已保存的模型：{model_path}，请先运行 train.py 进行训练。")
        
    print(f"[*] 正在从 {model_path} 加载模型...")
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
        
    print("[*] 正在对测试集（共262条数据）进行存活预测...")
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[+] SVM 测试集预测准确率: {accuracy * 100:.2f}%")
    
    # 确保 results 目录存在并写入结果日志
    os.makedirs('results', exist_ok=True)
    with open('results/accuracy.txt', 'w', encoding='utf-8') as f:
        f.write(f"算法: SVM\n数据: Titanic\n测试集预测准确率: {accuracy * 100:.2f}%\n")
    print("[*] 测试评估日志已写入 results/accuracy.txt")

if __name__ == "__main__":
    run_testing()