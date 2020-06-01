from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

weights = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.125, 0.437, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.086, 0.517, 0.0, 0.0, 0.113, 0.0, 0.0, 0.0, 0.506, 0.0, 0.231, 0.0, 0.0, 0.0, 0.171, 0.0, 0.0, 0.0],
    [0.0, 0.087, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.603, 0.0, 0.1, 0.0, 0.0, 0.0, 0.311, 0.0, 0.07, 0.0, 0.0]])
aspects = ['staff', 'perks', 'bagels', 'menu', 'crowded', 'grilled branzino']


sns.set()
data_word = ['shit',',','this','food','is','very','disappointment','.']
data_index = ['model1', 'model2']
data_att = [[0.3276,0.0003,0.0009,0.0000,0.0010,0.0192,0.6497,0.0013],
            [0.0184,0.0000,0.0005,0.0000,0.0000,0.0000,0.9810,0.0000]
            ]

d = pd.DataFrame(data = data_att,index = data_index, columns=data_word)

f, ax = plt.subplots()
sns.heatmap(d, vmin=0, vmax=1.0, ax=ax, cmap="OrRd", annot=True)
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=45)
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=45, horizontalalignment='right')
plt.show()

