import numpy as np
from sympy import Matrix
# 编程求解矩阵函数的数值解
"""用于矩阵论课程作业
    2023210047 王天翼"""

def matrix_function(A, coefficient):
    A = Matrix(A)
    P, J = A.jordan_form() #A = PJ(P^-1)
    invP = P.inv()
        
    LEN = len(coefficient)
    Jk_list = [np.eye(A.shape[0])]
    for k in range(LEN - 1):
        Jk_list.append(np.dot(Jk_list[-1], J)) #J^k
     
    f_J = 0
    for k in range(LEN):
        f_J = f_J + coefficient[k] * Jk_list[k] #f(J)
     
    f_A = np.dot(np.dot(P, f_J), invP) #f(A) = P f(J) (P^-1)
    return f_A
     
######test######
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
coefficient = [1, 2, 3]
print(matrix_function(A, coefficient))