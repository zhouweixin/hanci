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

a1 = data1[1].astype(float)
a2 = data2[1].astype(float)
a3 = data3[1].astype(float)
a4 = data4[1].astype(float)

d1 = [a1]
d2 = [a2]
d3 = [a3]
d4 = [a4]

labels1 = ['Case 1a']
labels2 = ['Case 1b']
labels3 = ['Case 2a']
labels4 = ['Case 2b']

df1 = pd.DataFrame(d1, columns=data1[0], index=labels1)
df2 = pd.DataFrame(d2, columns=data2[0], index=labels2)
df3 = pd.DataFrame(d3, columns=data3[0], index=labels3)
df4 = pd.DataFrame(d4, columns=data4[0], index=labels4)

fig = plt.figure()

ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

cax1 = ax1.matshow(df1, interpolation='nearest', cmap='binary', vmin=0, vmax=1)
cax2 = ax2.matshow(df2, interpolation='nearest', cmap='binary', vmin=0, vmax=1)
cax3 = ax3.matshow(df3, interpolation='nearest', cmap='binary', vmin=0, vmax=1)
cax4 = ax4.matshow(df4, interpolation='nearest', cmap='binary', vmin=0, vmax=1)

cb = fig.colorbar(cax1, ax=[ax1, ax2, ax3, ax4], shrink=0.9)
cb.set_ticks(np.linspace(0, 1, 11))

tick_spacing = 1
ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax1.set_xticklabels([''] + list(df1.columns))
ax1.set_yticklabels([''] + list(df1.index))

ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax2.set_xticklabels([''] + list(df2.columns))
ax2.set_yticklabels([''] + list(df2.index))

ax3.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax3.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax3.set_xticklabels([''] + list(df3.columns))
ax3.set_yticklabels([''] + list(df3.index))

ax4.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax4.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax4.set_xticklabels([''] + list(df4.columns))
ax4.set_yticklabels([''] + list(df4.index))

plt.show()
