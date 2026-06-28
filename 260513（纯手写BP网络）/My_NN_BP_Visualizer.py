import os
import math
import random
import tkinter as tk
from tkinter import messagebox

# =====================================================================
# 1. 纯手写原生数学矩阵运算（不使用任何第三方库如 NumPy）
# =====================================================================

def transpose(matrix):
    """矩阵转置"""
    return [list(row) for row in zip(*matrix)]

def mat_mul_vec(matrix, vector):
    """矩阵与向量相乘 (M x N) * (N x 1) -> (M x 1)"""
    return [sum(m_ij * v_j for m_ij, v_j in zip(row, vector)) for row in matrix]

def vec_add(vec1, vec2):
    """向量加法"""
    return [v1 + v2 for v1, v2 in zip(vec1, vec2)]

def relu(vector):
    """ReLU 激活函数"""
    return [max(0.0, x) for x in vector]

def relu_derivative(z_vector, grad_vector):
    """ReLU 激活函数的导数 (z > 0 则导数为1，否则为0)"""
    return [g if z > 0 else 0.0 for z, g in zip(z_vector, grad_vector)]

def format_matrix(matrix):
    """学术级格式化输出二维矩阵"""
    lines = []
    for row in matrix:
        formatted_row = "  ".join(f"{x:7.4f}" for x in row)
        lines.append(f"[  {formatted_row}  ]")
    return "\n".join(lines)

def format_vector(vector):
    """学术级格式化输出一维向量"""
    formatted_vec = "  ".join(f"{x:7.4f}" for x in vector)
    return f"[  {formatted_vec}  ]"


# =====================================================================
# 2. 原生神经网络计算引擎
# =====================================================================

class CustomNeuralNetwork:
    def __init__(self):
        # 设定随机种子以确保初始化重现性
        random.seed(42)
        
        # 1. 初始化权重和偏置参数 (使用标准正态分布缩放的 Xavier/He 初始化思想)
        self.W1 = [[random.gauss(0, 0.3) for _ in range(5)] for _ in range(4)] # 4x5
        self.b1 = [0.1 for _ in range(4)] # 4
        
        self.W2 = [[random.gauss(0, 0.3) for _ in range(4)] for _ in range(3)] # 3x4
        self.b2 = [0.1 for _ in range(3)] # 3
        
        self.W3 = [[random.gauss(0, 0.3) for _ in range(3)]] # 1x3
        self.b3 = [0.1] # 1

    def forward(self, x):
        """前向传播计算各节点状态值"""
        # 第一层: 输入 -> 隐藏层1
        z1 = vec_add(mat_mul_vec(self.W1, x), self.b1)
        a1 = relu(z1)
        
        # 第二层: 隐藏层1 -> 隐藏层2
        z2 = vec_add(mat_mul_vec(self.W2, a1), self.b2)
        a2 = relu(z2)
        
        # 第三层: 隐藏层2 -> 输出
        z3_val = sum(w * a for w, a in zip(self.W3[0], a2)) + self.b3[0]
        a3_val = z3_val # 线性激活输出
        
        # 储存中间状态用于反向传播
        self.state = {
            'x': x,
            'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2,
            'z3': [z3_val], 'a3': [a3_val]
        }
        return a3_val

    def backward(self, y_true):
        """反向传播计算梯度并返回保存参数和梯度副本"""
        x = self.state['x']
        z1, a1 = self.state['z1'], self.state['a1']
        z2, a2 = self.state['z2'], self.state['a2']
        a3_val = self.state['a3'][0]

        # 保存更新前的参数副本用于GUI展示
        old_params = {
            'W1': [list(r) for row in self.W1 for r in [row]], 'b1': list(self.b1),
            'W2': [list(r) for row in self.W2 for r in [row]], 'b2': list(self.b2),
            'W3': [list(r) for row in self.W3 for r in [row]], 'b3': list(self.b3)
        }

        # 1. 计算输出层梯度 (Layer 3)
        dz3 = 2.0 * (a3_val - y_true)
        
        dW3 = [[dz3 * a_j for a_j in a2]]
        db3 = [dz3]

        # 2. 计算隐藏层2梯度 (Layer 2)
        # da2 = (W3)^T * dz3
        da2 = [self.W3[0][j] * dz3 for j in range(3)]
        dz2 = relu_derivative(z2, da2)
        
        dW2 = [[dz2[i] * a_j for a_j in a1] for i in range(3)]
        db2 = list(dz2)

        # 3. 计算隐藏层1梯度 (Layer 1)
        # da1 = (W2)^T * dz2
        da1 = [sum(self.W2[i][j] * dz2[i] for i in range(3)) for j in range(4)]
        dz1 = relu_derivative(z1, da1)
        
        dW1 = [[dz1[i] * x_j for x_j in x] for i in range(4)]
        db1 = list(dz1)

        gradients = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2,
            'dW3': dW3, 'db3': db3
        }

        # 4. 执行梯度下降更新参数（学习率即步长 = 1.0）
        learning_rate = 1.0
        
        for i in range(4):
            for j in range(5):
                self.W1[i][j] -= learning_rate * dW1[i][j]
            self.b1[i] -= learning_rate * db1[i]

        for i in range(3):
            for j in range(4):
                self.W2[i][j] -= learning_rate * dW2[i][j]
            self.b2[i] -= learning_rate * db2[i]

        for j in range(3):
            self.W3[0][j] -= learning_rate * dW3[0][j]
        self.b3[0] -= learning_rate * db3[0]

        return old_params, gradients


# =====================================================================
# 3. 纯手写 CSV 数据解析引擎与安全数据回退
# =====================================================================

def load_regression_csv():
    """解析回归数据文件"""
    csv_filename = "回归.csv"
    
    # 离线安全回退数据（防止答辩现场由于文件未放在同一路径或编码问题报错）
    fallback_data = [
        {"x": [0.2, 0.4, 0.6, 0.8, 0.1], "y": 0.73},
        {"x": [0.9, 0.3, 0.2, 0.5, 0.7], "y": 1.02},
        {"x": [0.1, 0.8, 0.5, 0.4, 0.9], "y": 0.88}
    ]
    
    if not os.path.exists(csv_filename):
        print(f"[*] 警告: 未在当前路径检测到 '{csv_filename}'，已自动启用备用数据集保证系统稳定运行。")
        return fallback_data
        
    try:
        data = []
        with open(csv_filename, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            # 过滤空行并去除行尾换行符
            lines = [line.strip() for line in lines if line.strip()]
            if len(lines) < 2:
                return fallback_data
            
            # 解析数据行
            for line in lines[1:]:
                parts = line.split(',')
                if len(parts) >= 7:
                    x_vals = [float(p) for p in parts[1:6]]
                    y_val = float(parts[6])
                    data.append({"x": x_vals, "y": y_val})
        return data if len(data) == 3 else fallback_data
    except Exception as e:
        print(f"[-] CSV解析出现异常 {e}，已自动安全回退。")
        return fallback_data


# =====================================================================
# 4. 可视化交互控制台 (Tkinter GUI)
# =====================================================================

class NeuralNetVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("20260513作业 纯手写简易神经网络 BP 算法及可视化分析系统")
        self.root.geometry("1180x680")
        self.root.configure(bg="#F5F6F9")

        # 加载数据和神经网络实例
        self.dataset = load_regression_csv()
        self.nn = CustomNeuralNetwork()
        self.current_sample_idx = 0

        self.setup_ui()
        self.draw_initial_state()

    def setup_ui(self):
        # 1. 顶部标题栏
        title_lbl = tk.Label(
            self.root, 
            text="简单神经网络前向计算和 BP 参数更新可视化展示", 
            font=("Microsoft YaHei", 16, "bold"), 
            bg="#2C3E50", fg="white", pady=12
        )
        title_lbl.pack(fill=tk.X)

        # 主内容区域容器
        main_frame = tk.Frame(self.root, bg="#F5F6F9", pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 2. 左侧神经网络结构拓扑图画布
        self.canvas_frame = tk.LabelFrame(
            main_frame, text=" 神经网络拓扑图及节点实时状态 (Activation) ",
            font=("Microsoft YaHei", 10, "bold"), bg="white", bd=2, relief=tk.GROOVE
        )
        self.canvas_frame.pack(side=tk.LEFT, padx=15, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, width=540, height=500, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 3. 右侧矩阵参数和梯度实时状态文本面板
        right_frame = tk.Frame(main_frame, bg="#F5F6F9")
        right_frame.pack(side=tk.RIGHT, padx=15, fill=tk.BOTH)

        self.panel_frame = tk.LabelFrame(
            right_frame, text=" 神经网络参数与反向梯度监控面板 ",
            font=("Microsoft YaHei", 10, "bold"), bg="white", bd=2, relief=tk.GROOVE
        )
        self.panel_frame.pack(fill=tk.BOTH, expand=True)

        self.text_display = tk.Text(
            self.panel_frame, width=65, height=27, 
            font=("Consolas", 10), bg="#2C3E50", fg="#ECF0F1",
            insertbackground="white", padx=10, pady=10, relief=tk.FLAT
        )
        self.text_display.pack(fill=tk.BOTH, expand=True)

        # 4. 底部控制交互栏
        control_frame = tk.Frame(self.root, bg="#BDC3C7", pady=8)
        control_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.info_lbl = tk.Label(
            control_frame, 
            text="系统就绪，点击下方按钮开始顺序执行第一组样本...", 
            font=("Microsoft YaHei", 11, "bold"), bg="#BDC3C7", fg="#2C3E50"
        )
        self.info_lbl.pack(side=tk.LEFT, padx=20)

        self.next_btn = tk.Button(
            control_frame, text="输入下一组样本并计算更新 >>", 
            font=("Microsoft YaHei", 11, "bold"), bg="#27AE60", fg="white",
            activebackground="#2ECC71", activeforeground="white",
            relief=tk.FLAT, padx=15, pady=4, command=self.process_step
        )
        self.next_btn.pack(side=tk.RIGHT, padx=20)

    def draw_initial_state(self):
        """画出未输入数据前的神经网络连线拓扑图"""
        self.canvas.delete("all")
        self.draw_network_topology(None, None, None, None)
        self.text_display.insert(tk.END, "请点击下方按钮，输入数据开始计算...")

    def draw_network_topology(self, x_vals, a1_vals, a2_vals, a3_val):
        """动态绘制神经网络拓扑层并实时更新椭圆节点值"""
        # 定义层间水平间距，节点垂直坐标
        layer_x = [60, 200, 340, 480]
        
        # 节点垂直方向 Y 坐标映射
        node_ys = [
            [50, 140, 230, 320, 410],  # 数据层 (5节点)
            [95, 185, 275, 365],       # 隐藏层1 (4节点)
            [140, 230, 320],           # 隐藏层2 (3节点)
            [230]                      # 输出层 (1节点)
        ]

        # 1. 绘制层标题标注
        layer_names = ["数据层(Input)", "隐藏层1(ReLU)", "隐藏层2(ReLU)", "输出层(Linear)"]
        for i, name in enumerate(layer_names):
            self.canvas.create_text(
                layer_x[i], 20, text=name, 
                font=("Microsoft YaHei", 10, "bold"), fill="#7F8C8D"
            )

        # 2. 绘制层间全连接突触线条
        for l in range(3):
            x1 = layer_x[l]
            x2 = layer_x[l+1]
            for i, y1 in enumerate(node_ys[l]):
                for j, y2 in enumerate(node_ys[l+1]):
                    self.canvas.create_line(x1, y1, x2, y2, fill="#BDC3C7", width=1)

        # 3. 绘制节点圆圈并填入实时数值
        colors = ["#E0F2F1", "#E8F5E9", "#E8F5E9", "#FFEBEE"]  # 对应各层配色
        border_colors = ["#00695C", "#2E7D32", "#2E7D32", "#C62828"]
        
        node_values = [x_vals, a1_vals, a2_vals, [a3_val] if a3_val is not None else None]

        for l in range(4):
            x = layer_x[l]
            r_x, r_y = 26, 20 # 椭圆的长短半轴
            for i, y in enumerate(node_ys[l]):
                # 绘制节点阴影椭圆
                self.canvas.create_oval(
                    x - r_x, y - r_y, x + r_x, y + r_y, 
                    fill=colors[l], outline=border_colors[l], width=2
                )
                
                # 获取并格式化实时激活值
                val_str = "None"
                if node_values[l] is not None:
                    val_str = f"{node_values[l][i]:.3f}"
                
                # 在椭圆内显示节点数值
                self.canvas.create_text(
                    x, y, text=val_str, 
                    font=("Consolas", 10, "bold"), fill="#2C3E50"
                )

    def process_step(self):
        """核心处理事件：前向传导 -> 反向传播（BP） -> 参数梯度更新 -> 实时绘制与状态刷新"""
        if self.current_sample_idx >= 3:
            messagebox.showinfo("训练完成", "三组数据已经全部循环运行计算完毕！")
            return

        sample = self.dataset[self.current_sample_idx]
        x, y_true = sample["x"], sample["y"]

        # 1. 前向传播
        y_pred = self.nn.forward(x)

        # 获取前向传播后每一层的激活值
        a1 = self.nn.state['a1']
        a2 = self.nn.state['a2']

        # 2. 反向传播计算梯度并完成更新
        old_params, grads = self.nn.backward(y_true)

        # 3. 动态刷新左侧拓扑图状态值
        self.canvas.delete("all")
        self.draw_network_topology(x, a1, a2, y_pred)

        # 4. 动态写入右侧控制台参数监控（当前参数与本次迭代的参数梯度值）
        self.text_display.delete("1.0", tk.END)
        
        log_content = []
        log_content.append(f"============================================================")
        log_content.append(f" 正在处理样本组 [{self.current_sample_idx + 1}/3]")
        log_content.append(f"============================================================")
        log_content.append(f" 输入特征向量 X : {format_vector(x)}")
        log_content.append(f" 真实期望值  Y : {y_true:.4f}")
        log_content.append(f" 网络预测输出 Y_hat: {y_pred:.4f}")
        log_content.append(f" 本次迭代 MSE 损失 : {(y_pred - y_true)**2:.6f}")
        log_content.append(f"==================================================== 参数监控\n")

        # 写入参数矩阵(W)与偏置项(b)以及本次反向传播对应的梯度(dW, db)
        log_content.append("--- Layer 1 ---")
        log_content.append("更新前的 W1 权重矩阵:")
        log_content.append(format_matrix(old_params['W1']))
        log_content.append("W1 本次计算梯度 Dw1:")
        log_content.append(format_matrix(grads['dW1']))
        log_content.append("更新前的 b1 偏置向量:")
        log_content.append(format_vector(old_params['b1']))
        log_content.append("b1 本次计算梯度 db1:")
        log_content.append(format_vector(grads['db1']))
        log_content.append("\n" + "-"*35)

        log_content.append("--- Layer 2 ---")
        log_content.append("更新前的 W2 权重矩阵:")
        log_content.append(format_matrix(old_params['W2']))
        log_content.append("W2 本次计算梯度 Dw2:")
        log_content.append(format_matrix(grads['dW2']))
        log_content.append("更新前的 b2 偏置向量:")
        log_content.append(format_vector(old_params['b2']))
        log_content.append("b2 本次计算梯度 db2:")
        log_content.append(format_vector(grads['db2']))
        log_content.append("\n" + "-"*35)

        log_content.append("--- Layer 3 ---")
        log_content.append("更新前的 W3 权重矩阵:")
        log_content.append(format_matrix(old_params['W3']))
        log_content.append("W3 本次计算梯度 Dw3:")
        log_content.append(format_matrix(grads['dW3']))
        log_content.append("更新前的 b3 偏置:")
        log_content.append(format_vector(old_params['b3']))
        log_content.append("b3 本次计算梯度 db3:")
        log_content.append(format_vector(grads['db3']))
        
        self.text_display.insert(tk.END, "\n".join(log_content))

        # 提示更新底部状态信息
        self.info_lbl.configure(
            text=f"已成功完成样本组 [{self.current_sample_idx + 1}/3] 的前向与反向梯度参数更新！",
            fg="#27AE60"
        )

        self.current_sample_idx += 1
        if self.current_sample_idx == 3:
            self.next_btn.configure(text="全部三组计算完毕", state=tk.DISABLED, bg="#95A5A6")


# =====================================================================
# 5. 系统主入口
# =====================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetVisualizer(root)
    root.mainloop()