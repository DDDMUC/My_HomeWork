

# 💡 常见问题与排雷日记 (FAQ & Debug Log)

本项目在开发、重构以及多任务调试过程中，经历了一系列严谨的数理逻辑推导与底层工程排雷。以下是核心技术问题与真实 Debug 过程的归纳整理，旨在记录技术演进轨迹：

## 1. 算法与数据数理逻辑篇 (Theory & Logic)

### Q1.1: 为什么在对干豆数据集（Dry Bean）进行中位数插补（Imputation）时，必须严格使用训练集（Train Set）的统计量，而不能使用全数据集？

**A**: 为了防止**数据泄露（Data Leakage）**。

测试集和验证集在模型训练阶段一般应保持“不可见性”。

如果直接计算整个 CSV（包含测试和验证数据）的中位数来填充空值，那么测试集的分布特征就已经提前渗透到了数据预处理流水线中，容易导致评估出的准确率偏高。

因此，本项目在 `src/data_preprocess.py` 的 `clean_dirty_beans` 函数中，**严格仅提取 `train_df` 的中位数向量**作为全局标尺，去离线填充验证集和测试集的 `NaN` 值。

---

### Q1.2: 为什么针头穿刺深度估算（20260603作业）可以不依赖“针头顶部坐标（0cm处）”进行鲁棒计算？
**A**: 

在真实的临床场景中，医生的手指抓握通常会遮挡针头顶部（0cm 处）。
为了解决该工程限制，我们利用手柄（Cylinder 套管）在物理世界上恒等于 $2\text{ cm}$（1cm至3cm处）的硬约束，在图像中计算手柄的像素跨度 $H_{\text{handle}}$，从而获得尺度比例尺：
$$k = \frac{2.0}{H_{\text{handle}}} \quad (\text{cm/pixel})$$
因此，我们只需在图像中追踪**手柄下边缘（3cm处）至皮肤表面入射点**的垂直像素差 $D_{\text{skin\_to\_handle}}$，即可计算出暴露在外的下段针长：
$$L_{\text{exposed}} = D_{\text{skin\_to\_handle}} \times k$$
由于下段针总长为 $2\text{ cm}$，最终扎入体内的深度 $P$ 可以通过以下完全无需针头顶部的公式获得：
$$P = 2.0 - L_{\text{exposed}} = 2.0 \times \left(1 - \frac{D_{\text{skin\_to\_handle}}}{H_{\text{handle}}}\right) \quad (\text{cm})$$

---

## 2. 真实工程排雷日记 


### Q2.1: 在数据预处理中，为什么调用 `df['Class'].astype(str).str.strip().upper()` 会抛出 `AttributeError: 'Series' object has no attribute 'upper'`？
*   **问题现场**：
    在对干豆脏标签进行大小写规整化时，试图通过链式调用直接转大写，程序中断崩溃。
*   **成因剖析**：
    在 Pandas 框架中，`.str` 是专门用于处理文本序列的访问器（Accessor）。执行 `df['Class'].astype(str).str.strip()` 之后，返回的对象是一个普通的 Pandas `Series`。`Series` 对象本身并没有内建的 `upper()` 方法。
*   **解决方案**：
    每次进行链式字符串处理时，都必须显式通过 `.str` 重新获取访问器。修正后的代码为：
    ```python
    df['Class'] = df['Class'].astype(str).str.strip().str.upper()
    ```

---

### Q2.2: 在训练支持向量机（SVM）时，为什么会抛出 `ValueError: Input X contains NaN`？数据清洗明明已经执行了 `fillna`。
*   **问题现场**：
    在运行大作业评测脚本时，随机森林顺利通过，但运行到 `svm.fit(X_train, y_train)` 时程序中断崩溃。
*   **成因剖析**：
    在清洗脏数据时，为了剔除未识别的类别行，我们使用了以下循环：
    ```python
    for df in [train_df, val_df, test_df]:
        df = df[df['Class'] != 'UNKNOWN']  # 发生局部重绑定！
        df[feature_cols] = df[feature_cols].fillna(medians)
    ```
    在 Python 中，循环变量 `df` 的重赋值 `df = df[...]` 仅仅改变了该循环内部局部引用的指向，**并没有真正修改原始的 `train_df`、`val_df` 变量**。导致先前置为 `NaN` 的异常值没有被真正填充，进而引发 SVM 拒绝拟合。
*   **解决方案**：
    放弃循环内的直接重赋值，采用显式的列表解包和原地复制（`copy`）更新：
    ```python
    final_dfs = []
    for df in [train_df, val_df, test_df]:
        df = df[df['Class'] != 'UNKNOWN'].copy()
        df[feature_cols] = df[feature_cols].fillna(medians)
        final_dfs.append(df)
    train_df, val_df, test_df = final_dfs # 显式解包写回原变量
    ```

---

### Q2.3: 运行 `main_ann.py` 进行 PyTorch 训练时，为什么报 `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1047x9 and 7x64)`？
*   **问题现场**：
    Titanic 神经网络前向传播时，在线性层 `F.linear(input, self.weight, self.bias)` 处抛出矩阵乘法不匹配错误。
*   **成因剖析**：
    原先的网络结构定义中，默认输入特征维度为 7（`input_dim=7`）。但在我们更新了高标准特征预处理后，加入了 `sibsp`、`Parch` 并对 `Sex`、`Pclass`、`Embarked` 执行了更彻底的 One-Hot 哑变量编码，使特征矩阵的实际列数变为了 **9**。
    由于 $(1047 \times 9) \times (7 \times 64)$ 在线性代数上无法进行矩阵乘法，导致运行时报错。
*   **解决方案**：
    将 `main_ann.py` 中静态写死的模型初始化修改为**动态特征感知初始化**。直接读取清洗后数据的实际形状 `X_train.shape[1]` 传入网络：
    ```python
    # 动态获取特征维度，保证网络结构对特征工程的自适应兼容
    model = TitanicANN(input_dim=X_train.shape[1]).to(DEVICE)
    ```
