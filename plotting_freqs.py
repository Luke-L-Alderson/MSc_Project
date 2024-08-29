import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ''' Plot for the ns hyperparameter Sweep '''
# fig = plt.figure(figsize=(5, 5))
# df = pd.read_csv("ns_hp_sweep.csv")
# df["Test Loss"] = df["Test Loss"]/2.42
# df = df.pivot(index="bs", columns="lr", values="Test Loss")

# ns_hp = sns.lineplot(df, markers="o", dashes=False, legend=False)
# ns_hp.set(xscale='log',
#           xlabel="Batch Size",
#           ylabel='Test Loss')

# legend_handles, _= ns_hp.get_legend_handles_labels()
# #ns_hp.legend(title="Learning Rate", handles=legend_handles, labels=['1e-6', '1e-5', '1e-4', '1e-3', '1e-2'], bbox_to_anchor=(1,1))
# plt.xscale('log', base=2)
# ns_hp.set_ylim(0, 2)
# sns.set(font_scale = 1)
# sns.set_style("darkgrid")

# ''' Plot for the s hyperparameter Sweep '''
# df = pd.read_csv("hp_sweep.csv")
# df["Test Loss"] = df["Test Loss"]/2.42
# df = df.pivot(index="bs", columns="lr", values="Test Loss")

# fig = plt.figure(figsize=(5, 5))
# ns_hp = sns.lineplot(df, markers="o", dashes=False)
# ns_hp.set(xscale='log',
#           xlabel="Batch Size",
#           ylabel='Test Loss')

# legend_handles, _= ns_hp.get_legend_handles_labels()
# ns_hp.legend(title="Learning Rate", handles=legend_handles, labels=['1e-6', '1e-5', '1e-4', '1e-3', '1e-2'], bbox_to_anchor=(1,1))
# plt.xscale('log', base=2)
# ns_hp.set_ylim(0, 2)
# sns.set(font_scale = 1)
# sns.set_style("darkgrid")
# sns.move_legend(ns_hp, "upper right")

# ''' Plot for the kernel sweep '''
# df1 = pd.read_csv("kernel_search.csv")
# df1["Test Loss"] = df1["Test Loss"]/2.42
# df1["Spiking"] = "Spiking"
# df2 = pd.read_csv("ns_kernel_search.csv")
# df2["Test Loss"] = df2["Test Loss"]/2.42
# df2["Spiking"] = "Non-Spiking"
# df3 = pd.concat([df1, df2])
# fig = plt.figure(figsize=(10, 5))
# ns_hp = sns.scatterplot(df3, x="kernel_size", y="Test Loss", hue="Spiking")
# ns_hp.set(xlabel="Kernel Size",
#           ylabel='Test Loss',
#           xticks=[3, 5, 7, 9, 11, 13])
# ns_hp.set_ylim(0, 2)
# ns_hp.legend(title=[])

''' On Rate Plot '''
sns.set_style("darkgrid")
# df = pd.DataFrame({"Maximum Rate": [1, 5, 50, 100, 150, 200],
#                    "Test Loss": [3.983, 3.162, 1.518, 1.226, 1.109, 1.039]})
df = pd.read_csv("freqs.csv")
fig = plt.figure(figsize=(5, 5))
ns_hp = sns.lineplot(df, x="rate_on", y="Test Loss", marker="o", dashes=False, color="black")
ns_hp.set(xlabel=r"Maximum Rate,  $f_\text{on}$",
          ylabel='Test Loss')
ns_hp.set_ylim(0, 3)
ns_hp.legend(title=[])

''' Heatmap '''

fig = plt.figure(figsize=(5,5))
df = pd.read_csv("frequencies.csv")
#df = df[df["norm_type"]=="range"]

df = df.pivot(index="rate_on", columns="rate_off", values="Test Loss")
diags = []
for i in range(len(df)):
    #df.iloc[i, i] = np.nan
    diags.append(df.iloc[i, i])

print(np.array(diags).mean())
print(np.array(diags).std())
df_norm = (df - df.min(axis=None))/(df.max(axis=None)-df.min(axis=None))
hmap = sns.heatmap(df_norm, cmap="cividis", linewidths=1, linecolor='black')
hmap.invert_yaxis()
hmap.set_facecolor("black")
plt.xlabel(r"Minimum Rate, $f_\text{off}$")
plt.ylabel(r"Maximum Rate, $f_\text{on}$")

# ''' Stability '''
# df = pd.read_csv("stability.csv")
# fig = plt.figure(figsize=(10, 5))
# ns_hp = sns.boxplot(df, hue="recurrence", y="Test Loss", x="kernel_size")
# ns_hp.set(xlabel="Kernel Size",
#           ylabel='Test Loss',
#           ylim=(1, 3))
# ns_hp.legend(title=[])