[![FAQ](https://img.shields.io/badge/💡-FAQ_&_Debug_Log-9cf?style=flat-square)](FAQ.md)

# 🚀 ML_Project: 模块化多任务机器学习与深度学习系统

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-PyTorch%20%7C%20Scikit--Learn-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

本项目是一个高度工程化的、开箱即用的多任务机器学习平台。通过统一解耦的数据预处理中心，实现用统一的命令行入口灵活调度**自定义逻辑回归、SVM、KNN、KD-Tree、线性回归以及多层感知机（MLP/ANN）等多套算法，处理回归、二分类及高维图像多分类任务。

---

## 📂 项目文件系统拓扑图

```text
ML_Project/
├── data/                       # 统一数据集仓库
│   ├── titanic/                # 泰坦尼克号生存数据csv
│   └── house/                  # 房价回归数据csv
├── src/                        # 核心特征与模型源码库
│   ├── __init__.py             # 初始化脚本
│   ├── data_preprocess.py     # 统一特征清洗、异常值过滤与归一化引擎
│   ├── SVM.py                 # 支持向量机模型类 (Scikit-Learn 封装)
│   ├── linear_regression.py   # 线性回归模型类 (Scikit-Learn 封装)
│   ├── KNN.py                 # 原生手写 L2 距离 KNN 分类器
│   ├── logistic_regression.py # 原生手写 L2 正则化小批量逻辑回归
│   └── ANN.py                 # 回归与分类三任务人工神经网络结构定义 (PyTorch 架构)
├── models/                     # 序列化导出的模型权重仓库 (.pkl / .pth)
├── results/                    # 预测指标评估日志与训练监控可视化曲线
├── requirements.txt            # 项目依赖声明
├── train.py                    # 独立的 SVM 训练入口
├── test.py                     # 独立的 SVM 评估与推理入口
├── main.py                     # 传统机器学习统一调度中心
└── main_ann.py                 # 统一人工神经网络（ANN）执行引擎
