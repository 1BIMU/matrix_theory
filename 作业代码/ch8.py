"""this code belongs to BIMU for BUPT matrix calculation courses
    2023210047 王天翼 for studies only"""
import numpy as np
from sympy import Matrix
import time
def vetical_vector_norm(x,n):
    """args: x is a vector, n defines the order of the norm"""
    if n == np.inf:
        return np.max(np.abs(x))
    return np.sum(np.abs(x)**n)**(1/n)        

def inner_dot(a,b):
    return np.trace(a@b.T)

def orthogonal_vector(a):
    length = a.shape[-1]
    result = 0
    b = np.zeros(a.shape)
    b[0] = a[0]
    for i in range(length+1):
        for j in range(0,i):
            result += (inner_dot(a[i],b[j])/inner_dot(b[j],b[j]))*b[j]
        b[i] = a[i]-result
        result = 0
    return b

def normalize(a):
    length = a.shape[-1]+1
    for i in range(length):
        a[i] = a[i]/np.linalg.norm(a[i])
    return a

def matrix_under_basis(T,basis):
    dim = basis.shape[-1]+1
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
    for j in range(J.shape[0]):
        result[j,j] = J[j,j]**k
    return result

def T(x):
    y = np.dot(np.array([[0.2,0],[0.5,0.4]]) , x) + np.dot(x,np.array([[0.2,-0.2],[0.2,-0.2]]))
    return y

def mapping(x,basis):
    result = 0
    for i in range(x.shape[0]):
        result += x[i]*basis[i]
    return result

def mapping_orth(x,basis):#求x在正交基下的坐标
    result = np.zeros(basis.shape[-1]+1)
    for i in range(basis.shape[-1]+1):
        result[i] = inner_dot(x,basis[i])
    return result.T

def ls_vector_norm(x,n):
    """args: x is a coordinate in a linear space, n defines the order of the norm"""
    if n == np.inf:
        return np.max(np.abs(x))
    return np.sum(np.abs(x)**n)**(1/n)        

x = np.array([1,2,-3]).T
basis = np.array([[[-1,1],[0,0]],[[-1,0],[1,0]],[[0,0],[0,1]]]) 
A = np.array([[4,-4],[0,-3]])
orth_vector = orthogonal_vector(basis)
orth_basis = normalize(orth_vector)
orth_vector_A = mapping_orth(A,orth_basis)
A1 = orth_vector_A.copy()
print(f"第一题：x的列向量范数，这里尝试无穷范数：{vetical_vector_norm(x,np.inf)}")
print("==========================================================================")
print(f"第二题，矩阵空间元素A的向量范数：这里尝试无穷范数{vetical_vector_norm(orth_vector_A,np.inf)}")
print("==========================================================================")
def func_straight(T,x,k):
    result = x
    for i in range(k):
        result = T(result)
    return result

for k in range(30):
    A = T(A)
    orth_vector_A = mapping_orth(A,orth_basis)
    print(f"第{k+1}次迭代，矩阵A的向量范数：{ls_vector_norm(orth_vector_A,2),}")