import numpy as np
import matplotlib.pyplot as plt


a=np.random.randn(4,4)

plt.figure()
plt.imshow(a,cmap='jet')
plt.show()