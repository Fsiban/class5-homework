#myclass5-homework

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import combinations as cnb



# Creating Dataframe from Files
breast_cancer_df = pd.read_csv(filepath_or_buffer='breast-cancer/wdbc.data',sep=',',header=None)

# Adding data set headers to the data frame (I only need 11 columns 2 to 12 columns out of the 32)
breast_cancer_df.columns = ['ID_number', 'Diagnosis', 'Radius', 'Texture','Perimeter', 'Area', 'Smoothness',
                            'Compactness', 'Concavity', 'Concave_Points', 'Symmetry','Fractal_Dimension', '13', '14', '15',
                            '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32']


# redefining the data set header to only display 11 columns

breast_cancer = breast_cancer_df.drop(['ID_number', '13', '14', '15', '16', '17', '18', '19', '20', '21',
                                       '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32'], axis=1)

# checking if only 11 columns are displayed
# for i in breast_cancer:
#     print(i)
#     print()

# Display maximum columns when printing data

pd.set_option('display.max_columns', None)

# Print dat set in table format
print(breast_cancer.to_string())
print()


# Compute and print summary statistics
print(breast_cancer.describe())
print()

# Creating main plot folder and plot subfolders to distinguish the different plot types
os.makedirs('plots', exist_ok=True)
os.makedirs('plots/line_plots',exist_ok=True)
os.makedirs('plots/scatter_plots',exist_ok=True)
os.makedirs('plots/heatmap_plots', exist_ok=True)


# Plotting line chart to visualise data 1-feature (column) at a time,

for i in breast_cancer:
    plt.plot(breast_cancer[i], color='green')
    plt.title(i + '_by_Index')
    plt.xlabel('Index')
    plt.ylabel(i)
    plt.savefig(f'plots/line_plots/' + i + '_by_index_plot.png', format='png')
    plt.clf()


# using the combination function to make a list of combinations without repeats
comb = cnb(breast_cancer.columns, 2)


#Plotting scatterplot to visualise 2-features (columns)
for i in comb:
    plt.scatter(breast_cancer[i[0]], breast_cancer[i[1]], color='b')
    plt.title(i[0]+' to '+i[1])
    plt.xlabel(i[0])
    plt.ylabel(i[1])
    plt.savefig(f'plots/scatter_plots/'+i[0]+'_to_'+i[1]+'.png', format='png')
    plt.close()

# Plotting heat map to visualise 2-features
# create a list of correlations using corr()
corr = breast_cancer.corr()


# create a plot figure
fig, ax = plt.subplots()

# create heatmap table
im = ax.imshow(corr.values)

# set labels for the table
ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.columns)))
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.columns)

# Rotate the tick labels and set their alignment to the table.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        text = ax.text(j, i, np.around(corr.iloc[i, j], decimals=2),
                       ha="center", va="center", color="black")
plt.savefig(f'plots/heatmap_plots/heatmap.png',format='png')
plt.close()