
import numpy as np

a = [[1, 2, 3, 4, 5],
	[1, 2, 3, 4, 5],
	[1, 2, 3, 4, 5]]


print(np.array_split(a, 2))

# a = [1, 2, 3, 4, 5]
# b = [1, 2, 3, 4, 5]

# seed = np.random.get_state()
# np.random.shuffle(a)
# np.random.set_state(seed)
# np.random.shuffle(b)

# print(a)
# print(b)
