import numpy as np

A = np.array([[4, 3, 1],[1, -5, 7],[5, -2, 8]], dtype=np.dtype(float))
b = np.array([6, 8, 14], dtype=np.dtype(float))

x = np.linalg.solve(A, b)
print(f"Solution: {x}")