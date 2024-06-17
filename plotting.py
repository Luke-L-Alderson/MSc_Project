from helpers import tsne_plt, umap_plt, pca_plt
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

# # recurrency
# tsne_plt("75 Hz_1 Hz_0")
# tsne_plt("75 Hz_1 Hz_1")

# Rate Changes (Heat Map)
tsne_plt("1 Hz_1 Hz_1")
tsne_plt("200 Hz_1 Hz_1")

umap_plt("1 Hz_1 Hz_1")
umap_plt("200 Hz_1 Hz_1")

pca_plt("1 Hz_1 Hz_1")
pca_plt("200 Hz_1 Hz_1")
# Heatmap (Seaborn)
labels = ["1", "50", "100", "150", "200"]
labels = [1, 50, 100, 150, 200]
hmap = pd.read_csv("rates heatmap data.csv", index_col=0)#, header=labels, index_col=labels)

plt.figure(figsize = (10,7))
plt.title('Rates Heat Map (Errors normalised by frequency)')
ax = sns.heatmap(hmap*100, vmin=0.0, vmax=40, cmap = 'cividis', annot = True, fmt = '.2f')
plt.xlabel('Rate On, Hz')
plt.ylabel('Rate Off, Hz')
ax.invert_yaxis()
ax.set_facecolor('y')
# plt.savefig('heatmap.png')
# plt.savefig('heatmap.pdf')