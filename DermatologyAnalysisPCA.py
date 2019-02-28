
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import decomposition

datasetColumns=['erythema', 'scaling','definite borders','itching','koebner phenomenon']
df=pd.read_csv('dermatologyData.csv',delimiter=',',usecols=['erythema', 'scaling','definite borders','itching','koebner phenomenon','polygonal papules','follicular papules','oral mucosal involvement','knee and elbow involvement','scalp involvement','family history (0 or 1)','melanin incontinence'],header=0)
X=df.iloc[:,0:10]
#print(df)
pca=decomposition.PCA(n_components=5)
pc=pca.fit_transform(X)
print(pc)
explained_var=pca.explained_variance_ratio_
print('*'*50)
print(explained_var)

pc_df=pd.DataFrame(data=pc,columns=datasetColumns)

df=pd.DataFrame({'var':pca.explained_variance_ratio_,'PC':datasetColumns})
sns.barplot(x='PC',y='var',data=df,color='c')
sns.lmplot(x=datasetColumns[0],y=datasetColumns[1],data=pc_df,fit_reg=False,legend=True,scatter_kws={'s':80})

plt.show()

