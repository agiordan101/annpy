import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([0, 0, 1, 0])

print(a * b)
print(np.dot(a, b))
print(np.matmul(a, b))