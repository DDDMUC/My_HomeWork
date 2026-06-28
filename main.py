# main.py
import argparse
import os
from src.data_preprocess import clean_house_data
from src.linear_regression import HouseLinearRegression
import train as titanic_train
import test as titanic_test

def main():
    parser = argparse.ArgumentParser(description="ML_Project 统一算法命令行调度中心")
    parser.add_argument('--algo', type=str, required=True, choices=['logistic', 'linear', 'knn', 'svm'], help='指定运行的算法')
    parser.add_argument('--data', type=str, required=True, choices=['titanic', 'house', 'mnist', 'cifar10'], help='指定运行的数据集')
    parser.add_argument('--process', type=str, required=True, choices=['train', 'test'], help='指定执行阶段：训练(train) 或 测试(test)')
    
    args = parser.parse_args()
    
    print(f"[系统指令] 算法: {args.algo} | 数据集: {args.data} | 阶段: {args.process}")
    print("=" * 50)
    
    if args.data == 'titanic' and args.algo == 'svm':
        if args.process == 'train':
            titanic_train.run_training()
        elif args.process == 'test':
            titanic_test.run_testing()
            
    elif args.data == 'house' and args.algo == 'linear':
        if args.process == 'train':
            print("[*] 正在加载并清洗房价回归数据...")
            X, y = clean_house_data('data/house/house_data.csv')
            
            # 由于房价数据集极小且无独立测试集划分要求，此处直接进行拟合与自评估
            print("[*] 正在训练线性回归模型...")
            reg = HouseLinearRegression()
            reg.fit(X, y)
            
            print("[*] 正在进行房价预测评估...")
            predictions = reg.predict(X)
            
            # 计算简单的均方误差评估
            mse = ((predictions - y) ** 2).mean()
            print(f"[+] 线性回归拟合均方误差 (MSE): {mse:.4f}")
            
            os.makedirs('results', exist_ok=True)
            with open('results/accuracy.txt', 'a', encoding='utf-8') as f:
                f.write(f"算法: Linear Regression\n数据: House\n训练集拟合MSE: {mse:.4f}\n")
            print("[*] 房价回归评估结果已写入 results/accuracy.txt")
        else:
            print("[*] 房价数据集属于回归拟合任务，其训练与评估已合并在 train 阶段执行。")
    else:
        print(f"[-] 暂不支持的调度组合：--algo={args.algo} --data={args.data}。请按照系统大框架扩展其他算法。")

if __name__ == "__main__":
    main()