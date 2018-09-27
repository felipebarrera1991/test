import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#importing data set using pandas frameworks
data = pd.read_csv('bank-full.csv', sep=';', quotechar='"', encoding='utf8',parse_dates=True)
data.shape

data.head(5)
data.info()

#=========================================================================================================%
# First question
#=========================================================================================================%
# Housing and loan variables was used
data['housing'].unique()
data['loan'].unique()

# Create a new Dataframe with 0 and 1 for 'Housing' and 'loan' variables
# and grouping by variable 'job'
new_df = pd.get_dummies(data[['job', 'housing', 'loan']], columns=['housing', 'loan'])
new_df_group = new_df.groupby('job', as_index=False).sum()

# Calculating trend value
new_df_group['housing'] = new_df_group['housing_yes']/(new_df_group['housing_yes'] + new_df_group['housing_no'])
new_df_group['loan'] = new_df_group['loan_yes']/(new_df_group['loan_yes'] + new_df_group['loan_no'])

# Sort values by highest to lowest housing loan variables
new_df_group = new_df_group.sort_values(by='housing', ascending=False)

# Initialize the matplotlib figure and seabond sytle
f, ax = plt.subplots(figsize=(10,7))
sns.set(style='whitegrid')
sns.set_color_codes('pastel')
sns.barplot(x='housing', y='job', data=new_df_group, label='Housing', color='r')

sns.set_color_codes('muted')
sns.barplot(x='loan', y='job', data=new_df_group, label='Load', color='r')

ax.legend(ncol=2, loc='lower right', frameon=True)
ax.set(xlim=(0, 1), ylabel='Job', xlabel='Ratio')
sns.despine(left=True, bottom=True)
plt.savefig('question_1.png');

#=========================================================================================================%
# Second question
#=========================================================================================================%

# initially the variable "y" must be grouped, then the .count() operation is applied
new_df2 = data.groupby('y', as_index=False).count()

# initialize the matplotlib figure and seabond sytle
f, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
sns.barplot(x='campaign', y='y', data=new_df2, ax=ax1)
ax1.set(xlim=(0, new_df2['campaign'].max()*1.2), ylabel='Client Subscribed', xlabel='Campaing')
sns.despine(left=True, bottom=True)
# Pie plot
explode = (0, 0.1)  
ax2.pie(new_df2['campaign'], explode=explode, autopct='%1.1f%%',shadow=True, startangle=90)
ax2.axis('equal')
plt.tight_layout()
plt.savefig('question_2.png')

# Campaing and y variable was used. 
# Comparison for no and yes campaing and adding a new DataFrame

#=========================================================================================================%
# Third question
#=========================================================================================================%
new_df3 = pd.DataFrame({'campaign_no': data[data['y'] == 'no']['campaign'].describe()})
new_df3['campaign_yes'] = data[data['y'] == 'yes']['campaign'].describe()

# Explicar boxplot
f, ax = plt.subplots(figsize=(5,6))
sns.boxplot(x='y', y='campaign', data=data)
ax.set(ylabel='Number of Contacts During this Campaign', xlabel='Client Subscribed')
plt.savefig('question_3.png')

#=========================================================================================================%
# Fourth question
#=========================================================================================================%
new_df4 = pd.get_dummies(data[['poutcome', 'y']])
correlation = new_df4.corr(method='pearson')
mask = np.zeros_like(correlation)
mask[np.triu_indices_from(mask)] = True

with sns.axes_style('white'):
    plt.figure(figsize=(12,10))
    ax = sns.heatmap(correlation, mask=mask, vmax=.4, square=True,annot=True, cmap='YlGnBu')
    plt.savefig('question_4.png')

#=========================================================================================================%
# Fifth question
#=========================================================================================================%

default_y = data[(data['default'] == 'yes') ]['balance']
default_n = data[(data['default'] == 'no') ]['balance']

new_df5 = pd.DataFrame({'default_yes': data[(data['default'] == 'yes') ]['balance'].describe()})
new_df5['default_no'] = data[(data['default'] == 'no') ]['balance'].describe()

f, (ax1, ax2) = plt.subplots(1,2,figsize=(10,6))
sns.distplot(default_y, kde=True, bins=20, ax=ax1)
ax1.set(ylabel='Density', xlabel='Balance for Default')
sns.despine(left=True, bottom=True)

sns.distplot(default_n, kde=True, bins=20, ax=ax2)
ax2.set(ylabel='Density', xlabel='Balance for No Default')
plt.tight_layout()
plt.savefig('question_5.png')

#=========================================================================================================%
# Sixth question
#=========================================================================================================%

data.info()
Caracteristics = ['job', 'marital', 'education', 'default', 'loan']
new_df6 = data[data['housing'] == 'yes'][Caracteristics].describe()

fig = plt.figure(figsize=(6,4))
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
ax.bar(Caracteristics,new_df6.loc['freq'])
ax.set_ylabel('Frequency')
ax.set_xlabel('Characteristics')
plt.savefig('question_6.png')
