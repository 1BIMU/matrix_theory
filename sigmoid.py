import numpy as np
import matplotlib.pyplot as plt

# 定义sigmoid函数 f(w, b, x)
def sigmoid(w, b, x):
    return 1 / (np.exp(-(w * x + b)) + 1)

# 定义梯度下降函数 grad_de
def grad_de(a, w, b, sets, max_iter=10000, tol=1e-5):
    length = len(sets)
    for iteration in range(max_iter):
        w_rate = 0
        b_rate = 0
        for i in range(length):
            xi, yi = sets[i]
            prediction = sigmoid(w, b, xi)
            error = prediction - yi
            w_rate += error * xi
            b_rate += error
        
        # 计算平均梯度
        w_rate /= length
        b_rate /= length
        
        # 更新w和b
        w -= a * w_rate
        b -= a * b_rate
        
        # 如果梯度很小，则停止
        if abs(w_rate) < tol and abs(b_rate) < tol:
            print(f"梯度下降在第{iteration}次迭代时收敛")
            break
    
    return w, b

# 设定学习率和初始值
alpha = 0.01  # 学习率
w = 0
b = 0

# 设置随机数种子，以便结果可复现
np.random.seed(0)

# 生成数据对
num_data_points = 100
X_data = np.random.rand(num_data_points) * 10
Y_data = np.zeros(100)
for j in range(100):
    if X_data[j] >= 5:
        Y_data[j] = 1

# 组成训练集
sets = np.column_stack((X_data, Y_data))

# 使用梯度下降求解斜率和截距
w, b = grad_de(alpha, w, b, sets)

# 绘制数据点和拟合直线
X_plot = np.linspace(0, 10, 100)
Y_line = sigmoid(w, b, X_plot)  # 使用sigmoid函数来绘制预测结果

plt.figure(figsize=(8, 6))
plt.scatter(X_data, Y_data, color='r', label='数据点')
plt.plot(X_plot, Y_line, label=f'拟合模型', color='b')
plt.title('逻辑回归拟合')
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.legend()
plt.grid(True)
plt.show()
