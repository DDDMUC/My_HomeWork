# train.py
import os
import pickle
from src.data_preprocess import clean_titanic_data
from src.SVM import TitanicSVM

def run_training():
    print("[*] 正在加载并清洗泰坦尼克号训练数据...")
    X_train, y_train, _, _ = clean_titanic_data(
        'data/titanic/titanic_train_knn.csv',
        'data/titanic/titanic_test_knn.csv'
    )
    
    print("[*] 正在构建支持向量机（SVM）分类模型...")
    clf = TitanicSVM(kernel='rbf', C=1.0)
    clf.fit(X_train, y_train)
    
    # 确保 models 文件夹存在
    os.makedirs('models', exist_ok=True)
    model_path = 'models/svm_titanic_model.pkl'
    
    print(f"[*] 正在将训练好的模型参数保存至 {model_path} ...")
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
        
    print("[+] SVM 模型参数导出成功，训练流程结束。")

if __name__ == "__main__":
    run_training()