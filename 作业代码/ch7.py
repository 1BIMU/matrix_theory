"""this code belongs to BIMU for BUPT matrix calculation courses
    2023210047 王天翼"""

import numpy as np
from sympy import Matrix
def inner_dot(a,b):
    return np.trace(a@b.T)

def orthogonal_vector(a):
    length = a.shape[-1]
    result = 0
    b = np.zeros(a.shape)
    b[0] = a[0]
    for i in range(length):
        for j in range(0,i):
            result += (inner_dot(a[i],b[j])/inner_dot(b[j],b[j]))*b[j]
        b[i] = a[i]-result
        result = 0
    return b

def normalize(a):
    length1 = a.shape[0]
    length2 = a.shape[1]
    for i in range(length2):
        a[:,i] = a[:,i]/np.linalg.norm(a[:,i])
    return a

def matrix_under_basis(T,basis):
    dim = basis.shape[-1]
    result = np.zeros((dim,dim))
    for i in range(dim):
        T_ei = T(basis[i])
        for j in range(dim):
            result[i,j] = inner_dot(T_ei,basis[j])
    return result

def Jordan_form(M):#求M的Jordan标准型
    matrix = Matrix(M)
    P,J = matrix.jordan_form()
    return P,J,P.inv()


def func(J,k):
    result = np.eye(J.shape[0])
    for i in range(k):
        result=result@J
    return result

def T(x):
    return x+2*x.T


a = np.array([[[-1,1],[0,0]],[[-1,0],[1,0]],[[0,0],[0,1]]]) 
orthogonal_vectors = orthogonal_vector(a)
orthogonal_vectors = normalize(orthogonal_vectors)
print("正交基：")
print(orthogonal_vectors)
A = matrix_under_basis(T,orthogonal_vectors)
print("矩阵A：")
print(A)
P,Jor_A,P_inv = Jordan_form(A)
result =  func(Jor_A,10)
print("以下为采用矩阵论框架计算得到：")
print(P_inv@result@P)
print("以下为直接计算得到:")
print()
