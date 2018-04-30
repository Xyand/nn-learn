import matplotlib.pyplot as plt
import numpy as np

n = 100
x1 = 10*np.random.rand(*(n,))
x2 = 10*np.random.rand(*(n,))

#X_small = np.array([[1, 1], [1, -1]], dtype='float').T
X_small = np.stack([x1, x2], axis=-1).T
X_large = X_small +  100*np.array([1, 3])[:, np.newaxis]

y  = X_large[1, :] - 1.5*X_large[0, :]**0.5
print("Y_shape:", y.shape, X_small.shape)
# y = np.array([2, 0], dtype='float')

r = 1e2
w1_ = np.linspace(-r, r, 200)
w2_ = np.linspace(-r, r, 200)

w1, w2 = np.meshgrid(w1_, w2_)

X = X_large
print(X.shape)
print('X', X)
X = X - X.mean(axis=1, keepdims=True)
print('X', X)
w = np.stack([w1, w2], axis=-1)

print(w.shape)
print(w1.shape)
print('X', X)

X_mesh = X[np.newaxis, np.newaxis, ...]
E = (w.dot(X) - y)**2
print('E', E.shape)
E = np.sum(E, axis=-1)/ n

print('Minmax', np.min(E), np.max(E))

plt.imshow(np.log(E)/2, extent=[-r, r, -r, r])

plt.show()

print(E.shape)
