import numpy as np

a = np.zeros((7,7))
b = np.zeros((7,7))

for i in range(7):
    for j in range(7):
        a[i][j] = 2
        b[i][j] = 3

print(a)
print(b)

print(b-a)

