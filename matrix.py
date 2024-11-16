import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Basic Operations
print("Matrix Addition (A + B):\n", A + B)
print("Matrix Subtraction (A - B):\n", A - B)
print("Scalar Multiplication (2 * A):\n", 2 * A)
print("Element-wise Multiplication (A * B):\n", A * B)
print("Dot Product (A @ B):\n", np.dot(A, B))

# Matrix Properties
print("Transpose of A:\n", A.T)
print("Determinant of A:\n", np.linalg.det(A))
print("Inverse of A:\n", np.linalg.inv(A))
print("Rank of A:\n", np.linalg.matrix_rank(A))
print("Trace of A:\n", np.trace(A))

# Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues of A:\n", eigenvalues)
print("Eigenvectors of A:\n", eigenvectors)

# Solving Ax = b
b = np.array([5, 6])
x = np.linalg.solve(A, b)
print("Solution to Ax = b:\n", x)

# Decompositions
U, S, Vt = np.linalg.svd(A)
print("SVD Decomposition of A:\n", "U:\n", U, "Sigma:\n", S, "V^T:\n", Vt)
