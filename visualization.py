import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
###########第一章 非线性空间的可视化##################
# 定义 x 和 y 的取值范围
def unlinear_visualization():
    x = np.linspace(1, 100, 100)  # x 轴从 1 到 100
    y = np.linspace(1, 200, 200)  # y 轴从 1 到 200

    # 创建一个网格，x 和 y 的组合
    X, Y = np.meshgrid(x, y)

    # 计算每个 (x, y) 点的加法运算结果 f(x, y) = x^2 + y
    Z = X**2 + Y

    # 创建一个 3D 图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制表面图
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # 设置标签
    ax.set_xlabel('X Axis (1 to 100)')
    ax.set_ylabel('Y Axis (1 to 200)')
    ax.set_zlabel('f(x, y) = x^2 + y')

    # 设置标题
    ax.set_title('Nonlinear Space Visualization: f(x, y) = x^2 + y')

    # 显示图形
    plt.show()

#################################

def linear_visualization():
    x = np.linspace(1, 100, 100)  # x 轴从 1 到 100
    y = np.linspace(1, 200, 200)  # y 轴从 1 到 200

    # 创建一个网格，x 和 y 的组合
    X, Y = np.meshgrid(x, y)

    # 计算每个 (x, y) 点的加法运算结果 f(x, y) = x + y
    Z = X + Y

    # 创建一个 3D 图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制表面图
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # 设置标签
    ax.set_xlabel('X Axis (1 to 100)')
    ax.set_ylabel('Y Axis (1 to 200)')
    ax.set_zlabel('f(x, y) = x + y')

    # 设置标题
    ax.set_title('linear Space Visualization: f(x, y) = x + y')

    # 显示图形
    plt.show()

def norm_visualization():
    def lp_norm(v, p):
        return np.sum(np.abs(v) ** p) ** (1 / p)

    # 生成一个方向上的向量，假设方向为 (1, 0)
    direction = np.array([1,2,3,4])

    # 设置不同的Lp范数值，假设我们计算L1到L10范数
    p_values = list(range(1, 11))

    # 生成一个不同长度的向量，长度从 0 到 10
    lengths = np.linspace(0, 100, 10000)

    # 用于存储每个Lp范数的值
    lp_norms = {p: [] for p in p_values}

    # 计算不同长度下的Lp范数
    for length in lengths:
        vector = direction * length  # 生成长度为 length 的向量
        for p in p_values:
            lp_norms[p].append(lp_norm(vector, p))

    # 绘制结果
    plt.figure(figsize=(10, 6))

    # 绘制每个Lp范数的变化曲线
    for p in p_values:
        plt.plot(lengths, lp_norms[p], label=f'L{p} Norm')

    plt.xlabel('Vector Length')
    plt.ylabel('Lp Norm')
    plt.title('Lp Norms of a Vector in the Direction (1,2,3,4) as its Length Varies')
    plt.legend()
    plt.grid(True)
    plt.show()

def spetrum_radius_visualization():
        # 计算矩阵的谱半径
    def spectral_radius(A):
        eigenvalues = np.linalg.eigvals(A)
        return np.max(np.abs(eigenvalues))

    # 生成随机矩阵
    def generate_random_matrix(size):
        return np.random.rand(size, size)

    # 计算不同的矩阵LP范数
    def matrix_lp_norms(A, p_values):
        norms = []
        for p in p_values:
            norm = np.linalg.norm(A, p)
            norms.append(norm)
        return norms

    # 设置矩阵大小
    size = 10
    A = generate_random_matrix(size)

    # 计算谱半径
    rho_A = spectral_radius(A)

    # 定义不同的LP范数的p值
    p_values = [1, 2, np.inf]  # 1范数, 2范数, ∞范数

    # 计算不同的LP范数
    lp_norms = matrix_lp_norms(A, p_values)

    # 打印结果
    print(f"Spectral Radius (ρ(A)): {rho_A}")
    for p, norm in zip(p_values, lp_norms):
        print(f"LP Norm (p={p}): {norm}")

    # 可视化谱半径与LP范数的关系
    norm_labels = [f"LP Norm (p={p})" for p in p_values]

    # 创建图形
    plt.figure(figsize=(8, 5))
    plt.bar(norm_labels, lp_norms, color=['skyblue', 'salmon', 'lightgreen'])
    plt.axhline(y=rho_A, color='green', linestyle='--', label=f"Spectral Radius (ρ(A)) = {rho_A:.2f}")
    plt.ylabel('Norm Value')
    plt.title(f'LP Norms and Spectral Radius of a {size}x{size} Matrix')

    # 添加谱半径线
    plt.legend()
    plt.show()

def A_converge():
    # 函数：计算矩阵A的谱半径（最大特征值的绝对值）
    def spectral_radius(A):
        eigenvalues = np.linalg.eigvals(A)
        return max(np.abs(eigenvalues))

    # 创建一个2x2矩阵A，矩阵元素可调
    def create_matrix(alpha, beta):
        # 创建一个简单的2x2矩阵A
        A = np.array([[alpha, beta],
                    [beta, alpha]])
        return A

    # 绘制函数：展示谱半径的变化
    def plot_spectral_radius():
        # 设置alpha和beta的范围
        alpha_vals = np.linspace(-1, 1, 100)
        beta_vals = np.linspace(-1, 1, 100)
        
        # 用来存储谱半径
        spectral_radii = np.zeros((len(alpha_vals), len(beta_vals)))
        
        # 计算每个alpha, beta组合下的谱半径
        for i, alpha in enumerate(alpha_vals):
            for j, beta in enumerate(beta_vals):
                A = create_matrix(alpha, beta)
                spectral_radii[i, j] = spectral_radius(A)
        
        # 绘制谱半径的热图
        plt.figure(figsize=(8, 6))
        plt.contourf(alpha_vals, beta_vals, spectral_radii, levels=np.linspace(0, 1.5, 100), cmap='coolwarm')
        plt.colorbar(label='Spectral Radius $\\rho(A)$')
        plt.title('Spectral Radius of Matrix $A$')
        plt.xlabel('alpha')
        plt.ylabel('beta')
        plt.axhline(0, color='black',linewidth=0.8)
        plt.axvline(0, color='black',linewidth=0.8)
        plt.show()

    # 调用绘制函数
    plot_spectral_radius()


if __name__ == '__main__':
    A_converge()
