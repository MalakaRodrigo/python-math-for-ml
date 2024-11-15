import numpy as np

# 1. Creating Vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# -------------------------------
# 2. Basic Vector Operations
# -------------------------------

print("\nVector Addition (v1 + v2):", v1 + v2)
print("Vector Subtraction (v1 - v2):", v1 - v2)
print("Dot Product (v1 . v2):", np.dot(v1, v2))
print("Scalar Multiplication (3 * v1):", 3 * v1)

# -------------------------------
# 3. Advanced Vector Operations
# -------------------------------

# Magnitude (Norm) of a vector
print("\nMagnitude of v1:", np.linalg.norm(v1))

# Cross Product (for 3D vectors only)
print("Cross Product (v1 x v2):", np.cross(v1, v2))

# Unit Vector (Normalization)
unit_vector = v1 / np.linalg.norm(v1)
print("Unit Vector of v1:", unit_vector)

# Element-wise Multiplication (Hadamard Product)
print("Element-wise Multiplication (v1 * v2):", v1 * v2)

# Angle Between Vectors (in degrees)
angle_in_radians = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
angle_in_degrees = np.degrees(angle_in_radians)
print("Angle between v1 and v2 (in degrees):", angle_in_degrees)

# Projection of v1 onto v2
projection = (np.dot(v1, v2) / np.dot(v2, v2)) * v2
print("Projection of v1 onto v2:", projection)

# Outer Product (Tensor Product)
# The outer product is a mathematical operation that takes two vectors and produces a matrix. 
# Unlike the dot product, which results in a scalar, the outer product results in a matrix that represents all possible combinations of the components of the two vectors.
outer_product = np.outer(v1, v2)
print("Outer Product (v1 (*) v2):\n", outer_product)

# Random Vector Generation
random_vector = np.random.rand(3)
print("Random Vector:", random_vector)
