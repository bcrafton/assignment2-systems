
import numpy as np

x = np.array([
[1,2],
[3,4]
])

w = np.array([
[2,0],
[0,3]
])

y = x @ w
print (y)

w = np.array([2, 3])
y = x * w
print (y)
