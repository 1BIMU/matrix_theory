import numpy as np

A = np.array([[2, 1, 0, 2],
              [0, 0, 1, 2],
              [2, 1, 1, 4]])

# 进行奇异值分解
U, S, Vt = np.linalg.svd(A)

# 构造Σ的广义逆 Σ+
S_inv = np.zeros_like(A.T, dtype=float)  # 广义逆的形状
for i in range(len(S)):
    if S[i] > 1e-10:  # 设置一个小阈值，避免除零错误
        S_inv[i, i] = 1 / S[i]

# 计算广义逆 A+
A_pinv = Vt.T @ S_inv @ U.T

print("通过SVD分解计算的广义逆 A+ 为：")
print(A_pinv)
print("验证：AA^+A = A")
print(A@A_pinv@A)
