import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker


num = 20
user1 = np.array([0.435995, 0.025926, 0.549662, 0.435322, 0.420368, 0.330335, 0.204649, 0.619271, 0.299655, 0.266827, 0.621134, 0.529142, 0.134580, 0.513578, 0.184440, 0.785335, 0.853975, 0.494237, 0.846561, 0.079645])
user2 = np.array([0.505246, 0.065287, 0.428122, 0.096531, 0.127160, 0.596745, 0.226012, 0.106946, 0.220306, 0.349826, 0.467787, 0.201743, 0.640407, 0.483070, 0.505237, 0.386893, 0.793637, 0.580004, 0.162299, 0.700752])

item1 = np.array([0.964551, 0.500008, 0.889520, 0.341614, 0.567144, 0.427546, 0.436747, 0.776559, 0.535604, 0.953742, 0.544208, 0.082095, 0.366342, 0.850851, 0.406275, 0.027202, 0.247177, 0.067144, 0.993852, 0.970580])
item2 = np.array([0.800258, 0.601817, 0.764960, 0.169225, 0.293023, 0.524067, 0.356624, 0.045679, 0.983153, 0.441355, 0.504000, 0.323541, 0.259745, 0.386890, 0.832017, 0.736747, 0.379211, 0.013017, 0.797405, 0.269389])

data1 = np.array([user1, user2])
data2 = np.array([item1, item2])

df1 = pd.DataFrame(data1, columns=[i+1 for i in range(num)], index=['User 1', 'User 2'])
df2 = pd.DataFrame(data2, columns=[i+1 for i in range(num)], index=['Item 1', 'Item 2'])

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

cax1 = ax1.matshow(df1, interpolation='nearest', cmap='binary', vmin=0, vmax=1)
ax2.matshow(df2, interpolation='nearest', cmap='binary', vmin=0, vmax=1)

ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax1.set_xticklabels([''] + [str(i+1) for i in range(num)])
ax1.set_yticklabels([''] + ['User 1', 'User 2', 'Item 1', 'Item 2'])
ax1.set_xlabel('(a)')

ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax2.set_xticklabels([''] + [str(i+1) for i in range(num)])
ax2.set_yticklabels([''] + ['Item 1', 'Item 2'])
ax2.set_xlabel('(b)')

cb = fig.colorbar(cax1, ax=[ax1, ax2], shrink=0.9,fraction=0.016)#, pad=0.17)
cb.set_ticks(np.linspace(0, 1, 11))


# plt.xticks([i+1 for i in range(20)])
# plt.yticks(['User 1', 'User 2', 'Item 1', 'Item 2'])
# fig.xaxis.set_major_locator(ticker.MultipleLocator(1))
# fig.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.show()
