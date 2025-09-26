import numpy as np
import matplotlib.pyplot as plt
from utils import plot_lines

# ===============================================
# 1. Representing System of Linear Equations using Matrices
# ===============================================

# Coefficients of the system as a matrix.
A = np.array([[-1, 3],[3, 2]], dtype=np.dtype(float))

# Outputs of the system as a vector.
b = np.array([7, 1], dtype=np.dtype(float))

print("Matrix A:")
print(A)
print("Array b:")
print(b)

print(f"Shape of A: {np.shape(A)}")             # Shape of A: (2, 2)
print(f"Shape of A: {np.shape(b)}")             # Shape of A: (2,)

# Solve system of linear equations.
x = np.linalg.solve(A, b)
print(f"Solution: {x}")                         # Solution: [-1.  2.]

# Calculate the determinant of matrix A.
d = np.linalg.det(A)
print(f"Determinant of matrix A: {d:.2f}")      # Determinant of matrix A: -11.00

# ===============================================
# 2. Visualizing 2x2 Systems as Plotlines
# ===============================================

A_system = np.hstack((A, b.reshape((2, 1))))
print(A_system)

plot_lines(A_system)

# ===============================================
# 3. System of Linear Equations with No Solutions
# ===============================================

A_2 = np.array([
        [-1, 3],
        [3, -9]
    ], dtype=np.dtype(float))

b_2 = np.array([7, 1], dtype=np.dtype(float))

d_2 = np.linalg.det(A_2)

print(f"Determinant of matrix A_2: {d_2:.2f}")  # Determinant of matrix A_2: 0.00

# Check if matrix is singular (determinant close to zero)
if np.abs(d_2) < 1e-10:
    print("Matrix is singular - no unique solutions exists")
else:
    try:
        x_2 = np.linalg.solve(A_2, b_2)
        print(f"Solution: {x_2}")
    except np.linalg.LinAlgError as err:
        print(err)

A_2_system = np.hstack((A_2, b_2.reshape((2, 1))))
print(A_2_system)

plot_lines(A_2_system)

# ===============================================
# 4. System of Linear Equations with Infinite Solutions
# ===============================================

b_3 = np.array([7, -21], dtype=np.dtype(float))

A_3_system = np.hstack((A_2, b_3.reshape((2, 1))))
print(A_3_system)

plot_lines(A_3_system)