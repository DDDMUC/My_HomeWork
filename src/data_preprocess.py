# ML_Project/src/data_preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 干豆数据集脏标签映射表（纠正拼写错误与大小写混乱）
BEAN_LABEL_MAP = {
    'S3K3R': 'SEKER',
    'SEKER': 'SEKER',
    'BARBUNYA': 'BARBUNYA',
    'BOMBAY': 'BOMBAY',
    'CALI': 'CALI',
    'DERMASON': 'DERMASON',
    'DERMOSAN': 'DERMASON',  # 兼容拼写变体
    'HOROZ': 'HOROZ',
    'SIRA': 'SIRA'
}

def clean_titanic_data(train_path, test_path):
    """
    针对泰坦尼克号数据集的统一清洗与归一化
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 剔除全0列
    zero_cols = [col for col in train_df.columns if (train_df[col] == 0).all()]
    train_df = train_df.drop(columns=zero_cols, errors='ignore')
    test_df = test_df.drop(columns=zero_cols, errors='ignore')
    
    # 年龄中位数填充
    age_median = train_df['Age'].median()
    train_df['Age'] = train_df['Age'].fillna(age_median)
    test_df['Age'] = test_df['Age'].fillna(age_median)
    
    # 票价异常值处理与填充
    normal_fares = train_df['Fare'][train_df['Fare'] <= 600]
    fare_median = normal_fares.median()
    train_df['Fare'] = np.where((train_df['Fare'] > 600) | train_df['Fare'].isna(), fare_median, train_df['Fare'])
    test_df['Fare'] = np.where((test_df['Fare'] > 600) | test_df['Fare'].isna(), fare_median, test_df['Fare'])
    
    # 港口缺失值处理
    embarked_mode = train_df['Embarked'].mode()[0]
    train_df['Embarked'] = train_df['Embarked'].fillna(embarked_mode)
    test_df['Embarked'] = test_df['Embarked'].fillna(embarked_mode)
    
    # 类别特征 One-Hot 编码
    cat_cols = ['Sex', 'Pclass', 'Embarked']
    combined = pd.concat([train_df, test_df], keys=['train', 'test'])
    combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True, dtype=float)
    
    train_encoded = combined_encoded.xs('train')
    test_encoded = combined_encoded.xs('test')
    
    # 划分 X 与 y
    X_train_raw = train_encoded.drop(columns=['Passengerid', '2urvived'], errors='ignore')
    y_train = train_encoded['2urvived'].values
    X_test_raw = test_encoded.drop(columns=['Passengerid', '2urvived'], errors='ignore')
    y_test = test_encoded['2urvived'].values
    
    # 归一化 (Min-Max Scaling)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    return X_train, y_train, X_test, y_test

def clean_house_data(file_path):
    """
    针对房价数据的清洗与特征提取
    """
    df = pd.read_csv(file_path)
    X_raw = df[['x1', 'x2', 'x3', 'x4']].values
    y = df['y'].values
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)
    return X, y

def clean_dirty_beans(train_path, val_path, test_path):
    """
    针对期末干豆（Dry Bean）脏数据集的高级清洗引擎（修复了列表引用更新失效的底层 Bug）
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # 提取纯特征列名（除最后的 Class 之外）
    feature_cols = [col for col in train_df.columns if col != 'Class']
    
    # 1. 统一转换特征格式、将异常负数置为 NaN、修正分类标签
    cleaned_dfs = []
    for df in [train_df, val_df, test_df]:
        df = df.copy()
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = np.where(df[col] <= 0, np.nan, df[col])
            
        df['Class'] = df['Class'].astype(str).str.strip().str.upper()
        df['Class'] = df['Class'].map(BEAN_LABEL_MAP).fillna('UNKNOWN')
        cleaned_dfs.append(df)
        
    # 解包写回原始变量，确保真正修改
    train_df, val_df, test_df = cleaned_dfs
        
    # 2. 仅计算训练集的中位数，严格防范数据泄露
    medians = train_df[feature_cols].median()
    
    # 3. 剔除无效类别并执行缺失值中位数插补
    final_dfs = []
    for df in [train_df, val_df, test_df]:
        df = df[df['Class'] != 'UNKNOWN'].copy()
        df[feature_cols] = df[feature_cols].fillna(medians)
        final_dfs.append(df)
        
    train_df, val_df, test_df = final_dfs

    # 4. 建立类别标签与整数索引的映射
    classes = sorted(list(train_df['Class'].unique()))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    X_train = train_df[feature_cols].values
    y_train = train_df['Class'].map(class_to_idx).values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['Class'].map(class_to_idx).values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['Class'].map(class_to_idx).values
    
    # Z-Score 标准化缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, feature_cols, classes