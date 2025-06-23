a = np.array([[1,2],[3,4],[5,6],[7,8]])
b = np.array([[1,1],[2,2]])
print(f"a:\n{a}")
print(f"b:\n{b}")
print(f"dot:\n{a @ b.T}")

a_sq0 = np.sum(a ** 2, axis=1)
a_sq = a_sq0[:, None]
b_sq0 = np.sum(b ** 2, axis=1)
b_sq = b_sq0[None, :]
print(f"a_sq0:\n{a_sq0}")
print(f"a_sq:\n{a_sq}")
print(f"b_sq0:\n{b_sq0}")
print(f"b_sq:\n{b_sq}")
print(f"dist:\n{a_sq + b_sq - 2 * a @ b.T}")
# new_labels = np.argmin(distances_sq, axis=1)

distances = np.empty((a.shape[0], b.shape[0]), dtype=np.float32)
for i in range(b.shape[0]):
    distances[:, i] = np.sum((a - b[i])**2, axis=1)
print(distances)

distances = np.empty((a.shape[0], b.shape[0]), dtype=np.float32)
np.subtract.outer(a[:,0], b[:,0], out=distances)
distances **= 2
tmp = np.subtract.outer(a[:,1], b[:,1])
distances += tmp * tmp
print(distances)