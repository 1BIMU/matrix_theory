import numpy as np

def lu_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=np.float64)
    U = np.zeros_like(A, dtype=np.float64)

    for i in range(n):
        # Compute U
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        
        # Compute L
        for j in range(i, n):
            if i == j:
                L[i, i] = 1  # Diagonal of L is 1
            else:
                L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    return L, U

def forward_substitution(L, b):

    n = L.shape[0]
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        y[i] = (b[i] - sum(L[i, j] * y[j] for j in range(i)))
    return y

def backward_substitution(U, y):
    n = U.shape[0]
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    return x

def lu_solve(A, b):
    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x

def lu_decomposition_with_pivoting(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=np.float64)
    U = np.zeros_like(A, dtype=np.float64)
    P = np.eye(n)  # Permutation matrix

    # Copy A to avoid modifying the input
    A = A.copy()

    for i in range(n):
        # Partial pivoting: find the pivot row
        pivot = np.argmax(np.abs(A[i:, i])) + i
        if pivot != i:
            # Swap rows in A and update P and L
            A[[i, pivot]] = A[[pivot, i]]
            P[[i, pivot]] = P[[pivot, i]]
            if i > 0:
                L[[i, pivot], :i] = L[[pivot, i], :i]

        # Compute U and L
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

        L[i, i] = 1  # Diagonal elements of L are 1

    return P, L, U



if __name__ == "__main__":
    """N = 300
    b = np.random.random([N,1])
    
    A = np.zeros([N,N])
    a0 = np.random.random([1,N])
    a1 = np.hstack((a0,a0))
    
    for k in range(0,N):
        A[k,:] = a1[0, N-k:2*N-k]
    x = lu_solve(A, b)
    print(x)"""
    A = np.array([[2,-1,3],[1,2,1],[2,4,2]])
    L,U = lu_decomposition(A)
    print(f"L矩阵为：{L}")
    print(f"U矩阵为：{U}")

