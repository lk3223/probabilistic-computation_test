import numpy as np
import matplotlib.pyplot as plt

#xxx = np.arange(-1, 1, 0.01)

xc = 0.7
k = 48.68

xxx = np.arange(-2, 2.02, 0.02)
Vx = 5*xxx/k + xc
x = k*(Vx-xc)

# mi的最终写入概率
P = 1/(1 + np.exp(-x))
Fig = np.column_stack((Vx, P))

plt.plot(Vx, P)
plt.savefig('xxx1.jpg')
np.savetxt("xxx1.txt", Fig)
