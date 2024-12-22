"""this code belongs to BIMU for BUPT matrix calculation courses
    2023210047 王天翼"""

import numpy as np

def inner_dot(a,b):
    return np.dot(a,b)

def orthogonal_vector(a):
    length1 = a.shape[0]
    length2 = a.shape[1]
    result = 0
    b = np.zeros((length1,length2))
    b[:,0] = a[:,0]
    for i in range(length2):
        for j in range(0,i):
            result += (inner_dot(a[:,i],b[:,j])/inner_dot(b[:,j],b[:,j]))*b[:,j]
        b[:,i] = a[:,i]-result
        result = 0
    return b

def normalize(a):
    length1 = a.shape[0]
    length2 = a.shape[1]
    for i in range(length2):
        a[:,i] = a[:,i]/np.linalg.norm(a[:,i])
    return a

a = np.array([[1, 1,-1,1], [1, 0,0,-1], [0, 1,0,-1],[0,0,1,1]]) 
orthogonal_vectors = orthogonal_vector(a)
orthogonal_vectors = normalize(orthogonal_vectors)
print(orthogonal_vectors)