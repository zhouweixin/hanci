import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

data1 = np.array([
    ["great", 0.4123],
    ["party", 0.2951],
    ["favors", 0.1964],
    ["material", 0.3513],
    ["thin", 0.244],
    ["big", 0.6323],
    ["enough", 0.4332],
    ["wrap", 0.5139],
    ["adult", 0.007],
    ["head", 0.4625],
    ["comfortably", 0.5807],
    ["great", 0.4001],
    ["price", 0.5467],
    ["favors", 0.3988]]).T

data2 = np.array([
        ["printing", 0.3609],
        ["good", 0.5334],
        ["enough", 0.4421],
        ["birthday", 0.3264],
        ["party", 0.1892],
        ["moms", 0.1724],
        ["wanted", 0.3762],
        ["thin", 0.4271],
        ["sewn", 0.3858],
        ["poorly", 0.4411]]).T

data3 = np.array([
        ["love", 0.3624],
        ["monster", 0.2903],
        ["high", 0.3012],
        ["quality", 0.412],
        ["carftmanship", 0.3787],
        ["beauty", 0.4857],
        ["product", 0.0568],
        ["outstanding", 0.5346],
        ["star", 0.0902],
        ["doll", 0.0719]]).T

data4 = np.array([
        ["owner", 0.1176],
        ["liked", 0.4454],
        ["monster", 0.1292],
        ["high", 0.3122],
        ["recommended", 0.321],
        ["girls", 0.0752],
        ["younger", 0.3796],
        ["years", 0.1132],
        ["old", 0.4002]]).T

np.random.seed(2)
num = 20
user1 = np.random.random(num)
user2 = np.random.random(num)

item1 = np.random.random(num)
item2 = np.random.random(num)

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
