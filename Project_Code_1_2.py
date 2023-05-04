# -*- coding: utf-8 -*-
"""
Created on Mon May  9 01:53:48 2022

@author: Warsii
"""



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
import os
import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers
import tensorflow as tf

from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from plotly.offline import plot, iplot, init_notebook_mode
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from plotly.offline import plot, iplot, init_notebook_mode
from matplotlib.colors import LinearSegmentedColormap

import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold,GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score , accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, classification_report , auc

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

#%%

#Loading the data
#EDA- Explatory Data Analysis 
 
data = pd.read_csv('heart_failure.csv')

print("There are {:,} observations and {} columns in the data set.".format(data.shape[0], data.shape[1]))
print("There are {} missing values in the data.".format(data.isna().sum().sum()))


#%%
#Looking for missing values 

print(data.isna().sum())

#%%
#Dropping the null values
data.dropna()

#%%

#all rows control for null values
print(data.isnull().values.any())

#%%
#list of all the columns for all data.

print('Data Show Columns:\n')
print(data.columns)


#%%
#list of the first five rows of the dataset.


print(data.head(5))
#%%
#Looking for duplicate values.

bool_series = data.duplicated()

#%%
print(data.drop_duplicates())


#%%

print('Data Show Describe\n')
print(data.describe())

#%%
print('Data Show Info\n')
data.info()

print('\n')

print(data.dtypes)

#%%
print("Shape of the data:" , data.shape)

#%%

print(data.HeartDisease.value_counts())


#%%
# Oldpeak had some negative values so we removed those negative values and replaced it with zeros.

#data.Oldpeak[data.Oldpeak < 0] = 0

data.Oldpeak[data.Oldpeak < 0] = data.Oldpeak.mean()

#%%


#%%
#Looking for zero values in the numeric columns. 
#Age
count = (data['Age'] == 0).sum()
print('Count of zeros in Column  Age : ', count)

#Cholesterol
count = (data['Cholesterol'] == 0).sum()
print('Count of zeros in Column  Cholesterol : ', count)

#RestingBP
count = (data['RestingBP'] == 0).sum()
print('Count of zeros in Column  RestingBP : ', count)

#MaxHR
count = (data['MaxHR'] == 0).sum()
print('Count of zeros in Column  MaxHR : ', count)

#Oldpeak
count = (data['Oldpeak'] == 0).sum()
print('Count of zeros in Column  Oldpeak : ', count)

count = (data['ST_Slope'] ==0).sum()
print('Count of zeros in Column  ST Slope : ', count)
#As we found some zero values in the columns that is not a good dataset to work with. So we will use boxplot in order to 
# find out the outliers and then we will use some technique to reduce or eliminate the outliers in the dataset.

#%%
#Mean values of the numeric columns

print(data['Cholesterol'].mean())
print(data['RestingBP'].mean())
print(data['Age'].mean())
print(data['MaxHR'].mean())
print(data['Oldpeak'].mean())
#%%

# Creating dataset
np.random.seed(10)
data_1 = np.random.normal(data['Cholesterol'])
data_2 = np.random.normal(data['Age'])
data_3 = np.random.normal(data['RestingBP'])
data_4 = np.random.normal(data['MaxHR'])
data_5 = np.random.normal(data['Oldpeak'])
data_all = [data_1, data_2, data_3, data_4, data_5]
 
fig = plt.figure(figsize =(10, 7))
ax = fig.add_subplot(111)
 
# Creating axes instance
bp = ax.boxplot(data_all, patch_artist = True,
                notch ='True', vert = 0)
 
colors = ['#0000FF', '#00FF00',
          '#FFFF00', '#FF00FF' , '#94041f']
 
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
 
# changing color and linewidth of
# whiskers
for whisker in bp['whiskers']:
    whisker.set(color ='#8B008B',
                linewidth = 1.5,
                linestyle =":")
 
# changing color and linewidth of
# caps
for cap in bp['caps']:
    cap.set(color ='#8B008B',
            linewidth = 2)
 
# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color ='red',
               linewidth = 3)
 
# changing style of fliers
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)
     
# x-axis labels
ax.set_yticklabels(['data_1', 'data_2',
                    'data_3', 'data_4', 'data_5'])
 
# Adding title
plt.title("Customized box plot")
 
# Removing top axes and right axes
# ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
     
# show plot
plt.show()


#%%



#%%
#Replacing the zero values in the columns

data['Cholesterol'].replace(0,data['Cholesterol'].mean(axis=0),inplace=True)
data['RestingBP'].replace(0,data['RestingBP'].mean(axis=0),inplace=True)

#%%
#counting the target value.

print(data.HeartDisease.value_counts())

#%%


#%%

data1 = data.copy()

#%%

print(data1.Sex[:10])
#%%
#visualization of target variable
sns.set_theme(style="darkgrid")
sns.countplot(x="HeartDisease", data= data1)
plt.xlabel("HeartDisease (0 = No Disease , 1= Heart Disease)")
plt.show()

#%%


# #%%
# #Reducing the outliers of Oldpeak

# data1.boxplot('Oldpeak')

# #%%


# for x in ['Oldpeak']:
#     q75,q25 = np.percentile(data1.loc[:,x],[75,25])
#     intr_qr = q75-q25
 
#     max = q75+(1.5*intr_qr)
#     min = q25-(1.5*intr_qr)
 
#     data1.loc[data1[x] < min,x] = np.nan
#     data1.loc[data1[x] > max,x] = np.nan
    
    
# #%%
#New boxplot shows that the outliers are now handled.

# data1.boxplot('Oldpeak')

# #%%


# #%%
# #Reducing the outliers of Cholesterol.
# data1.boxplot('Cholesterol')

# #%%


# for x in ['Cholesterol']:
#     q75,q25 = np.percentile(data1.loc[:,x],[75,25])
#     intr_qr = q75-q25
 
#     max = q75+(1.5*intr_qr)
#     min = q25-(1.5*intr_qr)
 
#     data1.loc[data1[x] < min,x] = np.nan
#     data1.loc[data1[x] > max,x] = np.nan
    
    
# #%%
# #New boxplot shows that the outliers are now handled.
# data1.boxplot('Cholesterol')


# #%%
# #Reducing the outliers of RestingBP.
# data1.boxplot('RestingBP')

# #%%


# for x in ['RestingBP']:
#     q75,q25 = np.percentile(data1.loc[:,x],[75,25])
#     intr_qr = q75-q25
 
#     max = q75+(1.5*intr_qr)
#     min = q25-(1.5*intr_qr)
 
#     data1.loc[data1[x] < min,x] = np.nan
#     data1.loc[data1[x] > max,x] = np.nan
    
    
# #%%
# data1.boxplot('RestingBP')

#%%

print(data1.isnull().sum())
#%%
data1 = data1.dropna(axis = 0)


print(data1.shape)




#%%

print(data1.HeartDisease.value_counts())


#%%

#%%%
#Count Plot for Categorical features--- Sex, ChestPainTYpe, FastingBS, RestingECG, ExerciseAngina, ST_Slope


fig = plt.figure(figsize=(18,15))
gs = fig.add_gridspec(3,3)
gs.update(wspace=0.5, hspace=0.25)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])
ax5 = fig.add_subplot(gs[1,2])
ax6 = fig.add_subplot(gs[2,0])


background_color = "#ffffff"
color_palette = ["#222645","#394073","#60668F","#888CAB","#D7D8E3"]
fig.patch.set_facecolor(background_color) 
ax0.set_facecolor(background_color) 
ax1.set_facecolor(background_color) 
ax2.set_facecolor(background_color) 
ax3.set_facecolor(background_color) 
ax4.set_facecolor(background_color) 
ax5.set_facecolor(background_color) 
ax6.set_facecolor(background_color) 


# Title of the plot
ax0.spines["bottom"].set_visible(False)
ax0.spines["left"].set_visible(False)
ax0.spines["top"].set_visible(False)
ax0.spines["right"].set_visible(False)
ax0.tick_params(left=False, bottom=False)
ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.text(0.5,0.5,
         'Count plot for the \n nominal features\n_____________________',
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=18, fontweight='bold',
         fontfamily='serif',
         color="#000000")

# Sex count
ax1.text(0.3, 750, 'Sex', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax1,data=data1,x='Sex',palette=color_palette)
ax1.set_xlabel("")
ax1.set_ylabel("")


# ChestPainType count
ax2.text(0.3, 400, 'ChestPainType', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax2.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax2,data=data1,x='ChestPainType',palette=color_palette)
ax2.set_xlabel("")
ax2.set_ylabel("")

# FastingBS count
ax3.text(0.5, 900, 'FastingBS', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax3.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax3,data=data1,x='FastingBS',palette=color_palette)
ax3.set_xlabel("")
ax3.set_ylabel("")

# RestingECG count
ax4.text(0.75, 550, 'RestingECG', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax4.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax4,data=data1,x='RestingECG',palette=color_palette)
ax4.set_xlabel("")
ax4.set_ylabel("")

# ExerciseAngina count
ax5.text(0.75, 620, 'ExerciseAngina', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax5.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax5,data=data1,x='ExerciseAngina',palette=color_palette)
ax5.set_xlabel("")
ax5.set_ylabel("")

# ST_Slope count
ax6.text(1.5, 520, 'ST_Slope', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax6.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax6,data=data1,x='ST_Slope',palette=color_palette)
ax6.set_xlabel("")
ax6.set_ylabel("")


for s in ["top","right","left"]:
    ax1.spines[s].set_visible(False)
    ax2.spines[s].set_visible(False)
    ax3.spines[s].set_visible(False)
    ax4.spines[s].set_visible(False)
    ax5.spines[s].set_visible(False)
    ax6.spines[s].set_visible(False)

#%%
#Distribution of the numeric features respect to target feature 


fig = plt.figure(figsize=(18,18))
gs = fig.add_gridspec(5,2)
gs.update(wspace=0.5, hspace=0.5)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1])
ax4 = fig.add_subplot(gs[2,0])
ax5 = fig.add_subplot(gs[2,1])
ax6 = fig.add_subplot(gs[3,0])
ax7 = fig.add_subplot(gs[3,1])
ax8 = fig.add_subplot(gs[4,0])
ax9 = fig.add_subplot(gs[4,1])

background_color = "#ffffff"
color_palette = ["#222645","#394073","#60668F","#888CAB","#D7D8E3"]
fig.patch.set_facecolor(background_color) 
ax0.set_facecolor(background_color) 
ax1.set_facecolor(background_color) 
ax2.set_facecolor(background_color)
ax3.set_facecolor(background_color)
ax4.set_facecolor(background_color)
ax5.set_facecolor(background_color) 
ax6.set_facecolor(background_color) 
ax7.set_facecolor(background_color)
ax8.set_facecolor(background_color)
ax9.set_facecolor(background_color)

# Age title
ax0.text(0.5,0.5,"Distribution of age\naccording to\n target feature (HeartDisease)\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 18,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax0.spines["bottom"].set_visible(False)
ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.tick_params(left=False, bottom=False)

# Age
ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax1, data=data1, x='Age',hue="HeartDisease", fill=True,palette=["#222645","#394073"], alpha=.5, linewidth=0)
ax1.set_xlabel("")
ax1.set_ylabel("")

# RestingBP title
ax2.text(0.5,0.5,"Distribution of RestingBP\naccording to\n target feature (HeartDisease)\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 18,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax2.spines["bottom"].set_visible(False)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.tick_params(left=False, bottom=False)

# RestingBP
ax3.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax3, data=data1, x='RestingBP',hue="HeartDisease", fill=True,palette=["#222645","#394073"], alpha=.5, linewidth=0)
ax3.set_xlabel("")
ax3.set_ylabel("")

# Chol title
ax4.text(0.5,0.5,"Distribution of Cholesterol\naccording to\n target feature (HeartDisease)\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 18,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax4.spines["bottom"].set_visible(False)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.tick_params(left=False, bottom=False)

# Chol
ax5.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax5, data=data1, x='Cholesterol',hue="HeartDisease", fill=True,palette=["#222645","#394073"], alpha=.5, linewidth=0)
ax5.set_xlabel("")
ax5.set_ylabel("")

# MaxHR title
ax6.text(0.5,0.5,"Distribution of MaxHR\naccording to\n target feature (HeartDisease)\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 18,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax6.spines["bottom"].set_visible(False)
ax6.set_xticklabels([])
ax6.set_yticklabels([])
ax6.tick_params(left=False, bottom=False)

# MaxHR
ax7.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax7, data=data1, x='MaxHR',hue="HeartDisease", fill=True,palette=["#222645","#394073"], alpha=.5, linewidth=0)
ax7.set_xlabel("")
ax7.set_ylabel("")

# Oldpeak title
ax8.text(0.5,0.5,"Distribution of Oldpeak\naccording to\n target feature (HeartDisease)\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 18,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax8.spines["bottom"].set_visible(False)
ax8.set_xticklabels([])
ax8.set_yticklabels([])
ax8.tick_params(left=False, bottom=False)

# Oldpeak
ax9.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax9, data=data1, x='Oldpeak',hue="HeartDisease", fill=True,palette=["#222645","#394073"], alpha=.5, linewidth=0)
ax9.set_xlabel("")
ax9.set_ylabel("")

for i in ["top","left","right"]:
    ax0.spines[i].set_visible(False)
    ax1.spines[i].set_visible(False)
    ax2.spines[i].set_visible(False)
    ax3.spines[i].set_visible(False)
    ax4.spines[i].set_visible(False)
    ax5.spines[i].set_visible(False)
    ax6.spines[i].set_visible(False)
    ax7.spines[i].set_visible(False)
    ax8.spines[i].set_visible(False)
    ax9.spines[i].set_visible(False)

#%%

#1.Heart Attack according to Sex--- Sex
#2.Distribution of Chest Pain --- ChestPainType
#3.Heart Failure according to FastingBS-- FastingBS
#4.Heart Failure according to RestingECG---- RestingECG
#5.Heart Failure according to ExerciseAngina--- Exercise Angina
#6.Heart Failure according to ST_Slope---- ST_Slope 

fig = plt.figure(figsize=(18,18))
gs = fig.add_gridspec(6,2)
gs.update(wspace=0.5, hspace=0.5)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1])
ax4 = fig.add_subplot(gs[2,0])
ax5 = fig.add_subplot(gs[2,1])
ax6 = fig.add_subplot(gs[3,0])
ax7 = fig.add_subplot(gs[3,1])
ax8 = fig.add_subplot(gs[4,0])
ax9 = fig.add_subplot(gs[4,1])
ax10 = fig.add_subplot(gs[5,0])
ax11 = fig.add_subplot(gs[5,1])


background_color = "#ffffff"
color_palette = ["#222645","#394073","#60668F","#888CAB","#D7D8E3"]
fig.patch.set_facecolor(background_color) 
ax0.set_facecolor(background_color) 
ax1.set_facecolor(background_color) 
ax2.set_facecolor(background_color)
ax3.set_facecolor(background_color)
ax4.set_facecolor(background_color)
ax5.set_facecolor(background_color) 
ax6.set_facecolor(background_color) 
ax7.set_facecolor(background_color)
ax8.set_facecolor(background_color)
ax9.set_facecolor(background_color)
ax10.set_facecolor(background_color)
ax11.set_facecolor(background_color)


# Sex title
ax0.text(0.5,0.5,"Heart Failure\naccording to\nsex\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 18,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax0.text(1,.5,"0 - Female\n1 - Male",
        horizontalalignment = 'left',
         verticalalignment = 'center',
        fontsize = 14
        )
ax0.spines["bottom"].set_visible(False)
ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.tick_params(left=False, bottom=False)

# Sex
ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax1,data=data1,x='Sex',palette=["#222645","#394073"], hue='HeartDisease')
ax1.set_xlabel("")
ax1.set_ylabel("")

# ChestPainType title
# 1 = Typical Angina, 2 = Atypical Angina, 3 = Non-anginal Pain, 4 = Asymptomatic
ax2.text(0.5,0.5,"Distribution of Chest Pain\n_____________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 18,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax2.spines["bottom"].set_visible(False)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.tick_params(left=False, bottom=False)
ax2.text(1,.5,"1 - Typical Angina\n2 - Atypical Angina\n3 - Non-anginal Pain\n4 - Asymptomatic",
        horizontalalignment = 'left',
         verticalalignment = 'center',
        fontsize = 14
        )

# ChestPainType
ax3.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax3, data=data1, x='ChestPainType',hue="HeartDisease", fill=True,palette=["#222645","#394073"], alpha=.5, linewidth=0)
ax3.set_xlabel("")
ax3.set_ylabel("")


# FastingBS title
ax4.text(0.5,0.5,"Heart Failure\naccording to\nFastingBS \n ((fasting blood sugar > 120 mg/dl))\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 18,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax4.text(1,.5,"0 - False\n1 - True",
        horizontalalignment = 'left',
         verticalalignment = 'center',
        fontsize = 14
        )
ax4.spines["bottom"].set_visible(False)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.tick_params(left=False, bottom=False)

# FastingBS
ax5.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax5,data=data1,x='FastingBS',palette=["#222645","#394073"], hue='HeartDisease')
ax5.set_xlabel("")
ax5.set_ylabel("")


# RestingECG title
ax6.text(0.5,0.5,"Heart Failure\naccording to\nRestingECG \n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 18,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax6.text(1,.5,"0 - Normal\n1 - ST\n2- LVH",
        horizontalalignment = 'left',
         verticalalignment = 'center',
        fontsize = 14
        )
ax6.spines["bottom"].set_visible(False)
ax6.set_xticklabels([])
ax6.set_yticklabels([])
ax6.tick_params(left=False, bottom=False)

# RestingECG
ax7.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax7, data=data1, x='RestingECG',hue="HeartDisease", fill=True,palette=["#222645","#394073"], alpha=.5, linewidth=0)
ax7.set_xlabel("")
ax7.set_ylabel("")


# ExerciseAngina title
ax8.text(0.5,0.5,"Heart Failure\naccording to ExerciseAngina \n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 18,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax8.text(1,.5,"0 - No\n1 - Yes",
        horizontalalignment = 'left',
         verticalalignment = 'center',
        fontsize = 14
        )
ax8.spines["bottom"].set_visible(False)
ax8.set_xticklabels([])
ax8.set_yticklabels([])
ax8.tick_params(left=False, bottom=False)

# ExerciseAngina
ax9.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax9,data=data1,x='ExerciseAngina',palette=["#222645","#394073"], hue='HeartDisease')
ax9.set_xlabel("")
ax9.set_ylabel("")



# ST_Slope title
ax10.text(0.5,0.5,"Heart Failure\naccording to ST_Slope \n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 18,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax10.text(1,.5,"1 - UP\n2 - FLAT\n3 - DOWN",
        horizontalalignment = 'left',
         verticalalignment = 'center',
        fontsize = 14
        )
ax10.spines["bottom"].set_visible(False)
ax10.set_xticklabels([])
ax10.set_yticklabels([])
ax10.tick_params(left=False, bottom=False)

# ST_Slope
ax11.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax11, data=data1, x='ST_Slope',hue="HeartDisease", fill=True,palette=["#222645","#394073"], alpha=.5, linewidth=0)
ax11.set_xlabel("")
ax11.set_ylabel("")




for i in ["top","left","right"]:
    ax0.spines[i].set_visible(False)
    ax1.spines[i].set_visible(False)
    ax2.spines[i].set_visible(False)
    ax3.spines[i].set_visible(False)
    ax4.spines[i].set_visible(False)
    ax5.spines[i].set_visible(False)
    ax6.spines[i].set_visible(False)
    ax7.spines[i].set_visible(False)
    ax8.spines[i].set_visible(False)
    ax9.spines[i].set_visible(False)
    ax10.spines[i].set_visible(False)
    ax11.spines[i].set_visible(False)
    
#%%

#Distribution of heart disease 

fig, (ax21, ax22) = plt.subplots(nrows=1, ncols=2,
                    sharey=False, figsize= (14,6)
                                 )

ax21 = data1['HeartDisease'].value_counts().plot.pie( x="Heart disease", y = 'no. of patients', 
                                                     autopct = "%1.0f%%", labels=['Heart Disease', 'Normal'],
                                                     startangle = 60, ax=ax21)

ax21.set(title='Percentage of Heart disease patients in Dataset')

ax22 = data1['HeartDisease'].value_counts().plot(kind="barh", ax=ax22)

for i,j in enumerate(data1['HeartDisease'].value_counts().values):
    ax22.text(.5,i,j,fontsize = 12)
    
ax22.set(title ='No. of Heart disease patients in Dataset')

plt.show()    



#%%
#Pairplot of the dataset

sns.set_theme(style="ticks")
sns.pairplot(data1, hue="HeartDisease")

#%%
#Distribution of the numeric variables.---eitta rakhbo nahole uporer ta

fig, ax = plt.subplots(1,6, figsize=(20, 4))
sns.displot(data1, x="Age",)
sns.displot(data1, x="Cholesterol")
sns.displot(data1, x="RestingBP")
sns.displot(data1, x="MaxHR")
sns.displot(data1, x="Oldpeak")


#%%
#Density of the numeric variables by Heart Disease

sns.displot(data1, x="Age", hue="HeartDisease", stat="density")
sns.displot(data1, x="Cholesterol", hue="HeartDisease", stat="density")
sns.displot(data1, x="RestingBP", hue="HeartDisease", stat="density")
sns.displot(data1, x="MaxHR", hue="HeartDisease", stat="density")
sns.displot(data1, x="Oldpeak", hue="HeartDisease", stat="density")


#%%
#Number of people who have heart disease according to age 
plt.figure(figsize=(15,6))
sns.countplot(x='Age',data = data1, hue = 'HeartDisease',palette='GnBu')
plt.show()

#Number of people who have heart disease according to Cholesterol

plt.figure(figsize=(15,6))
sns.countplot(x='Cholesterol',data = data1, hue = 'HeartDisease',palette='GnBu')
plt.show()

#Number of people who have heart disease according to RestingBP

plt.figure(figsize=(15,6))
sns.countplot(x='RestingBP',data = data1, hue = 'HeartDisease',palette='GnBu')
plt.show()

#Number of people who have heart disease according to MaxHR

plt.figure(figsize=(15,6))
sns.countplot(x='MaxHR',data = data1, hue = 'HeartDisease',palette='GnBu')
plt.show()


#Number of people who have heart disease according to Oldpeak
plt.figure(figsize=(15,6))
sns.countplot(x='Oldpeak',data = data1, hue = 'HeartDisease',palette='GnBu')
plt.show()

#%%
#Heart Disease in function of Age and Max Heart Rate
# Create another figure
plt.figure(figsize=(9, 7))

# Scatter with postivie examples
plt.scatter(data1.Age[data1.HeartDisease==1],
            data.MaxHR[data1.HeartDisease==1],
            c="salmon")

# Scatter with negative examples
plt.scatter(data1.Age[data1.HeartDisease==0],
            data1.MaxHR[data1.HeartDisease==0],
            c="lightgreen")

# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);


#%%
#Heart Disease in function of Age and RestingBP
# Create another figure
plt.figure(figsize=(9, 7))

# Scatter with postivie examples
plt.scatter(data1.Age[data1.HeartDisease==1],
            data.RestingBP[data1.HeartDisease==1],
            c="salmon")

# Scatter with negative examples
plt.scatter(data1.Age[data1.HeartDisease==0],
            data1.RestingBP[data1.HeartDisease==0],
            c="lightgreen")

# Add some helpful info
plt.title("Heart Disease in function of Age and RestingBP")
plt.xlabel("Age")
plt.ylabel("RestingBP")
plt.legend(["Disease", "No Disease"]);

#%%
#Heart Disease in function of Age and Cholesterol
# Create another figure
plt.figure(figsize=(9, 7))

# Scatter with postivie examples
plt.scatter(data1.Age[data1.HeartDisease==1],
            data.Cholesterol[data1.HeartDisease==1],
            c="salmon")

# Scatter with negative examples
plt.scatter(data1.Age[data1.HeartDisease==0],
            data1.Cholesterol[data1.HeartDisease==0],
            c="lightgreen")

# Add some helpful info
plt.title("Heart Disease in function of Age and Cholesterol")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.legend(["Disease", "No Disease"]);

#%%

#Heart Disease in function of Age and Oldpeak
# Create another figure
plt.figure(figsize=(9, 7))

# Scatter with postivie examples
plt.scatter(data1.Age[data1.HeartDisease==1],
            data.Oldpeak[data1.HeartDisease==1],
            c="salmon")

# Scatter with negative examples
plt.scatter(data1.Age[data1.HeartDisease==0],
            data1.Oldpeak[data1.HeartDisease==0],
            c="lightgreen")

# Add some helpful info
plt.title("Heart Disease in function of Age and Cholesterol")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.legend(["Disease", "No Disease"]);


#%%

#Heart Disease in function of MaxHR and Cholesterol
# Create another figure
plt.figure(figsize=(9, 7))

# Scatter with postivie examples
plt.scatter(data1.MaxHR[data1.HeartDisease==1],
            data.Cholesterol[data1.HeartDisease==1],
            c="salmon")

# Scatter with negative examples
plt.scatter(data1.MaxHR[data1.HeartDisease==0],
            data1.Cholesterol[data1.HeartDisease==0],
            c="lightgreen")

# Add some helpful info
plt.title("Heart Disease in function of MaxHR and Cholesterol")
plt.xlabel("MaxHR")
plt.ylabel("Cholesterol")
plt.legend(["Disease", "No Disease"]);

#%%

#Heart Disease in function of MaxHR and RestingBP
# Create another figure
plt.figure(figsize=(9, 7))

# Scatter with postivie examples
plt.scatter(data1.MaxHR[data1.HeartDisease==1],
            data.RestingBP[data1.HeartDisease==1],
            c="salmon")

# Scatter with negative examples
plt.scatter(data1.MaxHR[data1.HeartDisease==0],
            data1.RestingBP[data1.HeartDisease==0],
            c="lightgreen")

# Add some helpful info
plt.title("Heart Disease in function of MaxHR and RestingBP")
plt.xlabel("MaxHR")
plt.ylabel("RestingBP")
plt.legend(["Disease", "No Disease"]);


#%%

#Heart Disease in function of RestingBP and Cholesterol
# Create another figure
plt.figure(figsize=(9, 7))

# Scatter with postivie examples
plt.scatter(data1.RestingBP[data1.HeartDisease==1],
            data.Cholesterol[data1.HeartDisease==1],
            c="salmon")

# Scatter with negative examples
plt.scatter(data1.RestingBP[data1.HeartDisease==0],
            data1.Cholesterol[data1.HeartDisease==0],
            c="lightgreen")

# Add some helpful info
plt.title("Heart Disease in function of RestingBP and Cholesterol")
plt.xlabel("RestingBP")
plt.ylabel("Cholesterol")
plt.legend(["Disease", "No Disease"]);

#%%
#Heart Disease in function of Cholesterol and Oldpeak
# Create another figure
plt.figure(figsize=(9, 7))

# Scatter with postivie examples
plt.scatter(data1.Cholesterol[data1.HeartDisease==1],
            data.Oldpeak[data1.HeartDisease==1],
            c="salmon")

# Scatter with negative examples
plt.scatter(data1.Cholesterol[data1.HeartDisease==0],
            data1.Oldpeak[data1.HeartDisease==0],
            c="lightgreen")

# Add some helpful info
plt.title("Heart Disease in function of Cholesterol and Oldpeak")
plt.xlabel("Cholesterol")
plt.ylabel("Oldpeak")
plt.legend(["Disease", "No Disease"]);

#%%
#Correlation Matrix
# Let's make our correlation matrix a little prettier
corr_matrix = data1.corr()
fig, ax = plt.subplots(figsize=(15, 15))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)




#%%



#############################################################
#                                                           # 
#----------------MODEL CREATION ---------------             #
#                                                           #
#                                                           #
#############################################################



#%%

print(data1.shape)

#%%

data_numeric = data1[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak' ]]



#%%

def roc_auc_curve(classifiers):
    '''
    Given a list of classifiers, this function plots the ROC curves
    
    '''       
    plt.figure(figsize=(12, 8))   
        
    for clf in zip(classifiers):
        clf.fit(X_train, y_train)
        
        pred_proba = clf.predict_proba(X_test)[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_test, pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=3, label=  +' ROC curve (area = %0.2f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) curves', fontsize=20)
        plt.legend(loc="lower right")


#%%

sc = StandardScaler()


#%%
dataX = data1.drop('HeartDisease',axis=1)

dataY=data1['HeartDisease']

#%%

X_train,X_test,y_train,y_test = train_test_split(dataX,dataY,test_size=0.2,random_state=21)


#%%

X_test.to_csv('D:/test_for.csv')


#%%

print('X_train',X_train.shape)
print('X_test',X_test.shape)
print('y_train',y_train.shape)
print('y_test',y_test.shape)

#%%

X_train[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak' ]] = sc.fit_transform(X_train[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak' ]])
X_test[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak' ]] = sc.transform(X_test[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak' ]])

#%%

print(X_test)


#%%

logisticRegr = LogisticRegression(C=0.1)

logisticRegr.fit(X_train, y_train)

#%%
#Train data accuracy-


logisticRegr_X_train_Prediction = logisticRegr.predict(X_train)
logisticRegr_training_data_accuracy = accuracy_score(logisticRegr_X_train_Prediction, y_train)*100

print('Training Data Accuracy of Logistic Regression:' ,logisticRegr_training_data_accuracy)


#%%
#predicted Y

Y_pred_logisticRegr = logisticRegr.predict(X_test)
print(Y_pred_logisticRegr)

#%%
#Actucal Y

print(y_test)

#%%

logisticRegr_test_data_accuracy = accuracy_score(y_test, Y_pred_logisticRegr)*100

print('Testing Data Accuracy of Logistic Regression:', logisticRegr_test_data_accuracy)

#%%

cf_matrix_LR = confusion_matrix(y_test, Y_pred_logisticRegr)

print(cf_matrix_LR)


#%%
#Evaluation parameters = precision and recall and F1 SCore

#precision
print(precision_score(y_test, Y_pred_logisticRegr))

#Recall
print(recall_score(y_test, Y_pred_logisticRegr))

#F1 score
print(f1_score(y_test, Y_pred_logisticRegr))

#Classification report
print(classification_report(y_test, Y_pred_logisticRegr))

#%%


#%%
fpr, tpr, _ = metrics.roc_curve(y_test,  logisticRegr.predict_proba(X_test)[:, 1])
auc_log = metrics.roc_auc_score(y_test, logisticRegr.predict_proba(X_test)[:, 1])

plt.plot(fpr,tpr,label="AUC="+str(auc_log))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve for logistic Regression')
plt.legend(loc=4)
plt.show()

#%%

group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix_LR.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix_LR.flatten()/np.sum(cf_matrix_LR)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cf_matrix_LR, annot=labels, fmt='', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


#%%

from numpy import array

X_train_art = np,array(X_train)
y_train_art = np,array(y_train)
X_test_art = np,array(X_test)
y_test_art = np,array(X_test)


#%%

X_train_re =  X_train_art.reshape(X_train_art, (X_train_art.shape[0], X_train_art.shape[1], 1))

X_train_re.shape

#%%

regressor = Sequential()

#%%
from keras.layers import LSTM
from keras.layers import Dropout

regressor.add(LSTM (units=50, return_sequences=True, input_shape = (X_train.shape[1],1  )))
regressor.add(Dropout(0.2))

#%%



# #%%
# from imblearn.over_sampling import SMOTE

# #%%
# smote = SMOTE()
# X_train_smote, y_train_smote = smote.fit_resample(X_train.astype('float'),y_train)


# #%%

# from collections import Counter

# #%%

# print('before smote', Counter(y_train))

# print('after smote', Counter(y_train_smote))

# #%%

# logisticRegr.fit(X_train_smote, y_train_smote)

# Y_pred_logisticRegr = logisticRegr.predict(X_test)


# logisticRegr_test_data_accuracy = accuracy_score(y_test, Y_pred_logisticRegr)
# print(logisticRegr_test_data_accuracy)

# #%%


# print(confusion_matrix(y_test, Y_pred_logisticRegr))


# #%%
# #Evaluation parameters = precision and recall and F1 SCore

# #precision
# print(precision_score(y_test, Y_pred_logisticRegr))

# #Recall
# print(recall_score(y_test, Y_pred_logisticRegr))

# #F1 score
# print(f1_score(y_test, Y_pred_logisticRegr))

# #Classification report
# print(classification_report(y_test, Y_pred_logisticRegr))


#%%


#%%

#Decision Tree


DecisionTree = DecisionTreeClassifier()


#%%
#HYPERPARAMETERS DECISION TREE

param_dist = {
    "criterion":["gini","entropy"],
    "max_depth":[1,2,3,4,5,6,7,None]
    
    }
 

#%%

grid_dt = GridSearchCV(DecisionTree, param_grid= param_dist, cv=10, n_jobs = -1)

#%%


grid_dt.fit(X_train, y_train)

#%%

y_train_grid_dt = grid_dt.predict(X_train)


print('Training Data Accuracy of Decision Tree:' ,accuracy_score(y_train, y_train_grid_dt))

#%%

y_pred_grid_dt = grid_dt.predict(X_test)
print('Testing Data Accuracy of Decision Tree:' , accuracy_score(y_pred_grid_dt,y_test))

#%%
print(grid_dt.best_estimator_)

#%%

print(grid_dt.best_score_)


#%%

cf_matrix_DT = confusion_matrix(y_test, y_pred_grid_dt)

print(cf_matrix_DT)


#%%
#Evaluation parameters = precision and recall and F1 SCore

#precision
print(precision_score(y_test, y_pred_grid_dt))

#Recall
print(recall_score(y_test, y_pred_grid_dt))

#F1 score
print(f1_score(y_test, y_pred_grid_dt))

#Classification report
print(classification_report(y_test, y_pred_grid_dt))


#%%
#ROC CURVE

fpr, tpr, _ = metrics.roc_curve(y_test,  grid_dt.predict_proba(X_test)[:, 1])
auc_log = metrics.roc_auc_score(y_test, grid_dt.predict_proba(X_test)[:, 1])

plt.plot(fpr,tpr,label="AUC="+str(auc_log))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve for Decision Tree')
plt.legend(loc=4)
plt.show()

#%%
group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix_DT.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix_DT.flatten()/np.sum(cf_matrix_DT)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cf_matrix_DT, annot=labels, fmt='', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()

#%%




#%%


#%%
#USING HYPERPARAMETERS

#RANDOM FOREST

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]



#%%



# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid)



#%%

rf_Model = RandomForestClassifier()

#%%

rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)

#%%

rf_Grid.fit(X_train, y_train)

#%%
rfgrid_X_train_Prediction = rf_Grid.predict(X_train)
rfgrid_training_data_accuracy = accuracy_score(rfgrid_X_train_Prediction, y_train)*100

print('Training Data Accuracy of Random Forest:' ,rfgrid_training_data_accuracy)



#%%

y_pred_RandomForestGrid = rf_Grid.predict(X_test)

print('Testing Data Accuracy of Random Forest:' ,accuracy_score(y_pred_RandomForestGrid,y_test))

#%%

print(rf_Grid.best_params_)

#%%

cf_matrix_RF = confusion_matrix(y_test, y_pred_RandomForestGrid)

print(cf_matrix_RF)


#%%
#Evaluation parameters = precision and recall and F1 SCore

#precision
print(precision_score(y_test, y_pred_RandomForestGrid))

#Recall
print(recall_score(y_test, y_pred_RandomForestGrid))

#F1 score
print(f1_score(y_test, y_pred_RandomForestGrid))

#Classification report
print(classification_report(y_test, y_pred_RandomForestGrid))

#%%

fpr, tpr, _ = metrics.roc_curve(y_test,  rf_Grid.predict_proba(X_test)[:, 1])
auc_log = metrics.roc_auc_score(y_test, rf_Grid.predict_proba(X_test)[:, 1])

plt.plot(fpr,tpr,label="AUC="+str(auc_log))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend(loc=4)
plt.show()
#%%

group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix_RF.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix_RF.flatten()/np.sum(cf_matrix_RF)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cf_matrix_RF, annot=labels, fmt='', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()



#%%




#%%

#KNN

knn_clf = KNeighborsClassifier()

knn_clf.fit(X_train,y_train)

#%%

knn_X_train_Prediction = knn_clf.predict(X_train)
knn_training_data_accuracy = accuracy_score(knn_X_train_Prediction, y_train)*100

print('Training Data Accuracy of KNN:' ,knn_training_data_accuracy)



#%%
y_pred_KNN = knn_clf.predict(X_test)


#%%

print('Testing Data Accuracy of KNNP:' , accuracy_score(y_test,y_pred_KNN))


#%%

cf_matrix_KNN = confusion_matrix(y_test, y_pred_KNN)

print(cf_matrix_KNN)


#%%
#Evaluation parameters = precision and recall and F1 SCore

#precision
print(precision_score(y_test, y_pred_KNN))

#Recall
print(recall_score(y_test, y_pred_KNN))

#F1 score
print(f1_score(y_test, y_pred_KNN))

#Classification report
print(classification_report(y_test, y_pred_KNN))

#%%


fpr, tpr, _ = metrics.roc_curve(y_test,  knn_clf.predict_proba(X_test)[:, 1])
auc_log = metrics.roc_auc_score(y_test, knn_clf.predict_proba(X_test)[:, 1])

plt.plot(fpr,tpr,label="AUC="+str(auc_log))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve for KNN')
plt.legend(loc=4)
plt.show()



#%%


group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix_KNN.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix_KNN.flatten()/np.sum(cf_matrix_KNN)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cf_matrix_KNN, annot=labels, fmt='', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()




#%%



#%%

svm_clf = SVC(kernel='rbf', gamma=0.1, C=1.0,probability=True)

svm_clf.fit(X_train, y_train)

#%%

svm_X_train_Prediction = svm_clf.predict(X_train)
svm_training_data_accuracy = accuracy_score(knn_X_train_Prediction, y_train)*100

print('Training Data Accuracy of SVM:' ,svm_training_data_accuracy)

#%%


y_pred_SVM = svm_clf.predict(X_test)

#%%

print('Testing Data Accurcay of SVM' ,accuracy_score(y_test,y_pred_SVM)*100)


#%%


cf_matrix_SVM = confusion_matrix(y_test, y_pred_SVM)

print(cf_matrix_SVM)


#%%
#Evaluation parameters = precision and recall and F1 SCore

#precision
print(precision_score(y_test, y_pred_SVM))

#Recall
print(recall_score(y_test, y_pred_SVM))

#F1 score
print(f1_score(y_test, y_pred_SVM))

#Classification report
print(classification_report(y_test, y_pred_SVM))

#%%


fpr, tpr, _ = metrics.roc_curve(y_test,  svm_clf.predict_proba(X_test)[:, 1])
auc_log = metrics.roc_auc_score(y_test, svm_clf.predict_proba(X_test)[:, 1])

plt.plot(fpr,tpr,label="AUC="+str(auc_log))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve for SVM')
plt.legend(loc=4)
plt.show()




#%%


group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix_SVM.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix_SVM.flatten()/np.sum(cf_matrix_SVM)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cf_matrix_SVM, annot=labels, fmt='', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()

#%%











#%%
#NN model

nn = tf.keras.models.Sequential()

#%%
 #Adding First Hidden Layer
nn.add(tf.keras.layers.Dense(units=16,activation="relu"))


#Adding Second Hidden Layer
nn.add(tf.keras.layers.Dense(units=8,activation="relu"))


#Adding Output Layer
nn.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

#Compiling ANN
nn.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=['accuracy'])


#%%

#Fitting ANN
history = nn.fit(X_train,y_train,validation_data=(X_test, y_test),epochs = 100, batch_size = 10)

#%%
#model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

#%%

#Model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


#%%
model1= 'NN'
nntrain = nn.predict(X_train)
nn_acc_score=(nn.evaluate(X_train, y_train)[1])
print("Training Accuracy of Artificial Neural Network model:",nn_acc_score*100,'\n')

#%%

model1= 'NN'
nnpred = nn.predict(X_test)
nn_acc_score=(nn.evaluate(X_test, y_test)[1])
print("Testing Accuracy of Artificial Neural Network model:",nn_acc_score*100,'\n')

#%%






#%%



#%%

gbm = lgb.LGBMClassifier(class_weight='balanced', random_state=92)

# Parameter tuning
grid = {'boosting_type': ['gbdt', 'dart'],
        'num_leaves': [int(x) for x in np.linspace(start = 20, stop = 50, num = 7)],
        'max_depth' : [-1, 3, 7, 14, 21],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.5, 1],
        'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)],
        'min_split_gain': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'min_child_samples': [3, 5, 7],
        'subsample': [0.5, 0.8, 0.95],
        'colsample_bytree': [0.6, 0.75, 1]}

gbm_cv=RandomizedSearchCV(estimator=gbm, param_distributions=grid, scoring='roc_auc', 
                         n_iter=100, cv=5, random_state=92, n_jobs=-1)

gbm_cv.fit(X_train, y_train)

# Best params
gbm = gbm_cv.best_estimator_

#%%

y_train_gbm = gbm.predict(X_train)

print('Training data accuracy of lgbm:' , accuracy_score(y_train, y_train_gbm))


#%%

y_pred_gbm = gbm.predict(X_test)

#%%

print('Testing data accuracy of lgbm' ,accuracy_score(y_test, y_pred_gbm))

#%%


cf_matrix_gbm = confusion_matrix(y_test, y_pred_gbm)

print(cf_matrix_gbm)


#%%
#Evaluation parameters = precision and recall and F1 SCore

#precision
print(precision_score(y_test, y_pred_gbm))

#Recall
print(recall_score(y_test, y_pred_gbm))

#F1 score
print(f1_score(y_test, y_pred_gbm))

#Classification report
print(classification_report(y_test, y_pred_gbm))

#%%


fpr, tpr, _ = metrics.roc_curve(y_test,  gbm.predict_proba(X_test)[:, 1])
auc_log = metrics.roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1])

plt.plot(fpr,tpr,label="AUC="+str(auc_log))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve for LGBM')
plt.legend(loc=4)
plt.show()



#%%


group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix_gbm.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix_gbm.flatten()/np.sum(cf_matrix_gbm)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cf_matrix_gbm, annot=labels, fmt='', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()






#%%



        
#%%

#XGBOOST 

param_grid = dict(
    n_estimators=stats.randint(10, 1000),
    max_depth=stats.randint(1, 10),
    learning_rate=stats.uniform(0, 1)
)

xgb_clf = XGBClassifier(use_label_encoder=False)
xgb_cv = RandomizedSearchCV(
    xgb_clf, param_grid, cv=3, n_iter=50, 
    scoring='accuracy', n_jobs=-1, verbose=1
)
xgb_cv.fit(X_train, y_train)
best_params = xgb_cv.best_params_
print(f"Best paramters: {best_params}")

xgb_clf = XGBClassifier(**best_params)
xgb_clf.fit(X_train, y_train)


#%%

y_train_xgb_clf = xgb_clf.predict(X_train)

print('Training data accuracy of xgboost: ' , accuracy_score(y_train, y_train_xgb_clf)*100)

#%%

y_pred_xgb_clf = xgb_clf.predict(X_test)

print("Testing data accuracy of xgboost: ", accuracy_score(y_test, y_pred_xgb_clf))

#%%

cf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb_clf)

print(cf_matrix_xgb)


#%%
#Evaluation parameters = precision and recall and F1 SCore

#precision
print(precision_score(y_test, y_pred_xgb_clf))

#Recall
print(recall_score(y_test, y_pred_xgb_clf))

#F1 score
print(f1_score(y_test, y_pred_xgb_clf))

#Classification report
print(classification_report(y_test, y_pred_xgb_clf))

#%%


fpr, tpr, _ = metrics.roc_curve(y_test,  xgb_clf.predict_proba(X_test)[:, 1])
auc_log = metrics.roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1])

plt.plot(fpr,tpr,label="AUC="+str(auc_log))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve for XGboost')
plt.legend(loc=4)
plt.show()



#%%


group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix_xgb.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix_xgb.flatten()/np.sum(cf_matrix_xgb)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cf_matrix_xgb, annot=labels, fmt='', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()





#%%




#%%

#NAIVE BAYES


nb_clf = GaussianNB()  
nb_clf.fit(X_train, y_train)  

#%%

y_train_nb_clf = nb_clf.predict(X_train)

print("training data acc of naive bayes: " , accuracy_score(y_train, y_train_nb_clf))

#%%

y_pred_nb_clf = nb_clf.predict(X_test)

print("testing data acc of naive bayes: " , accuracy_score(y_test, y_pred_nb_clf)*100)

#%%

cf_matrix_NB = confusion_matrix(y_test, y_pred_nb_clf)

print(cf_matrix_NB)


#%%
#Evaluation parameters = precision and recall and F1 SCore

#precision
print(precision_score(y_test, y_pred_nb_clf))

#Recall
print(recall_score(y_test, y_pred_nb_clf))

#F1 score
print(f1_score(y_test, y_pred_nb_clf))

#Classification report
print(classification_report(y_test, y_pred_nb_clf))

#%%


fpr, tpr, _ = metrics.roc_curve(y_test,  nb_clf.predict_proba(X_test)[:, 1])
auc_log = metrics.roc_auc_score(y_test, nb_clf.predict_proba(X_test)[:, 1])

plt.plot(fpr,tpr,label="AUC="+str(auc_log))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve for naive Bayes')
plt.legend(loc=4)
plt.show()



#%%


group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix_NB.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix_NB.flatten()/np.sum(cf_matrix_NB)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cf_matrix_NB, annot=labels, fmt='', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()

#%%


#MLP






#%%

#Multilayer percerptron

mlp_clf = MLPClassifier()

mlp_clf.fit(X_train, y_train)

#%%


y_train_mlp_clf = mlp_clf.predict(X_train)

print("Training data accuracy of MLP " , accuracy_score(y_train, y_train_mlp_clf))

#%%

y_pred_mlp_clf = mlp_clf.predict(X_test)


print("Testing data accuracy of MLP " , accuracy_score(y_test, y_pred_mlp_clf))

#%%

#%%


cf_matrix_MLP = confusion_matrix(y_test, y_pred_mlp_clf)

print(cf_matrix_MLP)


#%%
#Evaluation parameters = precision and recall and F1 SCore

#precision
print(precision_score(y_test, y_pred_mlp_clf))

#Recall
print(recall_score(y_test, y_pred_mlp_clf))

#F1 score
print(f1_score(y_test, y_pred_mlp_clf))

#Classification report
print(classification_report(y_test, y_pred_mlp_clf))

#%%


fpr, tpr, _ = metrics.roc_curve(y_test,  mlp_clf.predict_proba(X_test)[:, 1])
auc_log = metrics.roc_auc_score(y_test, mlp_clf.predict_proba(X_test)[:, 1])

plt.plot(fpr,tpr,label="AUC="+str(auc_log))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve for MLP')
plt.legend(loc=4)
plt.show()




#%%


group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix_MLP.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix_MLP.flatten()/np.sum(cf_matrix_MLP)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cf_matrix_MLP, annot=labels, fmt='', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()




#%%



#%%


from sklearn.ensemble import AdaBoostClassifier

#%%
# Import Support Vector Classifier

DT_ADA = DecisionTreeClassifier(criterion='gini', max_depth=7)

# Create adaboost classifer object
Ada_clf =AdaBoostClassifier(n_estimators=50, base_estimator=DT_ADA,learning_rate=1)

# Train Adaboost Classifer
Ada_clf.fit(X_train, y_train)


#%%

y_train_Ada_clf = Ada_clf.predict(X_train)

print("Training data accuracy of AdaBoost " , accuracy_score(y_train, y_train_Ada_clf))

#%%
#Predict the response for test dataset

y_pred_Ada_clf = Ada_clf.predict(X_test)


print("Testing data accuracy of AdaBoost " , accuracy_score(y_test, y_pred_Ada_clf))

#%%

cf_matrix_AdaBoost = confusion_matrix(y_test, y_pred_Ada_clf)

print(cf_matrix_AdaBoost)

#%%
#Evaluation parameters = precision and recall and F1 SCore

#precision
print(precision_score(y_test, y_pred_Ada_clf))

#Recall
print(recall_score(y_test, y_pred_Ada_clf))

#F1 score
print(f1_score(y_test, y_pred_Ada_clf))

#Classification report
print(classification_report(y_test, y_pred_Ada_clf))

#%%


fpr, tpr, _ = metrics.roc_curve(y_test,  Ada_clf.predict_proba(X_test)[:, 1])
auc_log = metrics.roc_auc_score(y_test, Ada_clf.predict_proba(X_test)[:, 1])

plt.plot(fpr,tpr,label="AUC="+str(auc_log))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve for AdaBoost')
plt.legend(loc=4)
plt.show()



#%%


group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix_AdaBoost.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix_AdaBoost.flatten()/np.sum(cf_matrix_AdaBoost)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cf_matrix_AdaBoost, annot=labels, fmt='', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()

#%%



#%%


#Predictive system



input_data = (67,1,4,100.0,299.0,0,2,125,1,0.9,2)


input_data_as_numpy_array = np.asarray(input_data)


input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


prediction= gbm.predict(input_data_reshaped)

print(prediction)

if(prediction [0] ==0):
    print('dont Have heart disease')
else:
    print('yes have hearrt disease')    



#%%


import csv

filename = 'test.csv'

with open(filename, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    for row in datareader:
        print(row)

for r in row:
    input_data = row[r]


input_data_as_numpy_array = np.asarray(input_data)


input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


prediction= Ada_clf.predict(input_data_reshaped)

print(prediction)

if(prediction [0] ==0):
    print('dont Have heart disease')
else:
    print('yes have hearrt disease')    


#%%

#Prediction on testing.

data_test = pd.read_csv('test_for.csv')

print(data_test.shape)
print(data_test)


#%%
#X_train prediction on the model

Y_train_pred_on_test = mlp_clf.predict(data_test) 

#%%
#Converting to DataFrame'

df1 = pd.DataFrame(Y_train_pred_on_test)

#%%

print(accuracy_score(y_test, Y_train_pred_on_test))
#%%
#Giving a column name

df1.columns = ['HeartDisease']

#%%
#Replacing the 1,0 with 'Yes' , 'No'.

df1.HeartDisease.replace((1,0), ('Yes','No'), inplace=True)

#%%















