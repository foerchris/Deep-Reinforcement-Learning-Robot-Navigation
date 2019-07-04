import matplotlib.pyplot as plt
import numpy as np
import math as m

X = np.random.random((100, 100)) # sample 2D array
plt.imshow(X, cmap="gray")
plt.show()

print(m.exp(-0.2))

init_lr = 1e-3
final_lr = 1e-5
epoch = 10
lr_decay_epoch = 120

lr = init_lr - (init_lr - final_lr)*(1-m.exp(-epoch/lr_decay_epoch))
print('learning rate' + str(lr))
