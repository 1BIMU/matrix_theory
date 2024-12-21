import numpy as np
import copy
from sympy import Matrix

class linear_space(object):
     
    def __init__(self, basis = [], number_field = complex):
        self.basis = basis #基
        self.number_field = number_field #数域
 
    def dim(self):
        return(len(self.basis)) #维数
     
 
class inner_product_space(linear_space): #内积空间
     
    def __init__(self, basis = [], number_field = complex, inner_product= 'None'):
        linear_space.__init__(self, basis, number_field)
        self.inner_product = inner_product
        self.true_basis = basis #真基
        self.gram_schmidt()
        
         
    def gram_schmidt(self): #施密特单位正交化
        temp_vectors = copy.copy(self.basis)
        Len = self.dim()
        result = []
        for k in range(Len):
            # 当前处理的向量
            current_vector = temp_vectors[k]
            # 当前向量归一化
            current_vector = current_vector / np.sqrt(self.inner_product(current_vector, current_vector))
            # 从其他向量中移除当前向量的投影
            for j in range(k+1, Len):
                temp_vectors[j] = temp_vectors[j] - (self.inner_product(current_vector, temp_vectors[j])) * current_vector
            # 添加正交化后的向量到结果中
            result.append(current_vector)
        self.basis = result#欧式空间中把所有的基都正交化
    
     
class element(object):
     
    def __init__(self, linear_space, info = 'coordinate', infomation = []):
        self.linear_space = linear_space #线性空间
        if info == 'coordinate':
            self.list2coordinate(infomation) #坐标
        if info == 'value':
            self.vector2coordinate(infomation) #值
             
    def list2np(self, coordinate):
        self.coordinate = []
        for line in coordinate:
            self.coordinate.append(self.linear_space.number_field(line))
        self.coordinate = np.array(self.coordinate)
  
    def vector2coordinate(self, vector): #计算坐标
        Len = self.linear_space.dim()
        self.coordinate = []
        for k in range(0, Len):
            self.coordinate.append(self.linear_space.inner_product(vector, self.linear_space.basis[k]))#向量与正交基的内积即为坐标
        self.coordinate = np.array(self.coordinate)
     
    def origin_vector(self): #计算原点向量
        v = self.linear_space.basis[0] * self.coordinate[0]
        Len = self.linear_space.dim()
        for k in range(1, Len):
            v += self.linear_space.basis[k] * self.coordinate[k]
        return v
    
class linear_transformation(object):
     
    def __init__(self, inner_product_space, transformation):
        self.inner_product_space = inner_product_space #内积空间
        self.trans = transformation #变换
        self.trans2matrix_stdbasis() #线性变换在标准正交基下的矩阵
        self.trans2matrix_basis(self.inner_product_space.true_basis)

         
    def trans2matrix_stdbasis(self): #线性变换在标准正交基下的矩阵
        te = []
        for line in self.inner_product_space.basis:
            te.append(self.trans(line))
        Len = self.inner_product_space.dim()
        self.std_matrix = np.zeros([Len,Len])
        for j in range(0, Len):
            for k in range(0, Len):
                self.std_matrix[k,j] = self.inner_product_space.inner_product(te[j], self.inner_product_space.basis[k])
    
    def trans2matrix_basis(self, basis): #线性变换在原基下的矩阵
        A = self.std_matrix
        basis = self.inner_product_space.true_basis
        C = []
        Len = len(basis) #等于T.linear_space.dim()
        for j in range(0, Len):
            xj = element(self.inner_product_space, 'value', basis[j]) 
            C.append(xj.coordinate)
        C = np.array(C).transpose() 
        invC = np.linalg.inv(C)
        self.matrix = np.dot(invC, np.dot(A,C)) #B = C^-1 *A*C

    def linear_trans(self, input_ele): #利用矩阵相乘来计算坐标
        output_ele = copy.copy(input_ele)
        output_ele.coordinate = np.dot(self.matrix, input_ele.coordinate)
        return output_ele
 
    def dot(self, arr): #矩阵相乘
        y = np.dot(self.matrix, arr) 
        return y
 
    def apply_function(self, func): #将线性变换的函数作用于基本线性变换中，并且修改矩阵
        self.matrix = func(self.dot)(np.eye(self.inner_product_space.dim()))
        self.trans = func(self.trans)

def get_Jordan_form(matrix): #计算Jordan块
    A = Matrix(matrix)
    P, J = A.jordan_form()
    return np.array(P), np.array(J)

def EX_11():
    ##课本例题11

    def T(x):
        return x@np.array([[0,1],[4,0]])
        
    def inner_product(x,y):#内积
        return np.sum(x*y)
        
    x_ls = linear_space([np.array([[1,1],[1,1]]),np.array([[1,1],[1,0]]),np.array([[1,1],[0,0]]),np.array([[1,0],[0,0]])], complex)
    x_inner_product_space = inner_product_space(x_ls.basis, complex, inner_product)
    x_linear_transformation = linear_transformation(x_inner_product_space, T)
    print(x_linear_transformation.matrix)

def EX_12():
    ##课本例题12，计算jordan型
    A = np.array([[-1,1,0],[-4,3,0],[1,0,2]])
    P, J = get_Jordan_form(A)
    print(f"得到P矩阵为：{P}")
    print(f"得到Jordan型矩阵为：{J}")
    

EX_11()
