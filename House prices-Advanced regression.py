#!/usr/bin/env python
# coding: utf-8

# In[113]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[114]:


housepricetest=pd.read_csv("test.csv")


# In[115]:


housepricetrain=pd.read_csv("train.csv")


# In[116]:


print(housepricetest.shape)
print(housepricetrain.shape)


# In[117]:


housepricetrain.info()


# In[118]:


housepricetest.info()


# In[119]:


#CONCATINATION OF BOTH FILES -for preprocessing
#For concatination the columns must be same.
#For concatination temporarily add dependent variable to test data and fill column with 'test'


# In[120]:


housepricetest['SalePrice']='test'


# In[121]:


#Concatination of dataframes
combinedf=pd.concat([housepricetest,housepricetrain],axis=0)


# In[122]:


combinedf.info()


# In[123]:


#separate num and non num cols in preprocessing
numcols=combinedf.select_dtypes(include=np.number)
objcols=combinedf.select_dtypes(include=['object'])


# In[124]:


objcols.columns


# In[125]:


objcols.describe()


# In[126]:


notavailablecols=[ 'Alley','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                  'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond','PoolQC', 'Fence', 'MiscFeature']


# In[127]:


for col in notavailablecols:
    objcols[col]=objcols[col].fillna('Not Available')


# In[128]:


#imputing the missing values with most frequent/mode
for col in objcols.columns:
    objcols[col]=objcols[col].fillna(objcols[col].value_counts().idxmax())
    
#idxmax()-impute with index/classmate of maximum frequency


# In[129]:


#to find how many missing values
numcols.isnull().sum().sort_values(ascending=False)


# In[130]:


#seperate categorical variables from numeric varaiables like rating scale
#data related like year,month etc
numcols.columns


# In[131]:


catcols=numcols[['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MoSold',
       'YrSold', 'GarageYrBlt', 'MSSubClass']]


# In[132]:


catcols.columns


# In[133]:


numcols.columns


# In[134]:


#Median imputation for num cols
for col in numcols.columns:
    numcols[col]=numcols[col].fillna(numcols[col].median())


# In[135]:


catcols.GarageYrBlt=catcols.GarageYrBlt.fillna(catcols.GarageYrBlt.value_counts().idxmax())


# In[136]:


#BsmtFinSF1,MasVnrArea


# In[137]:


#Objcols imputation -MSZoning
objcols.isnull().sum().sort_values(ascending=False)


# In[138]:


#concat all the 3 dataframes
combineddf_EDA=pd.concat([numcols,objcols,catcols],axis=1)


# In[139]:


housedf_EDA=combineddf_EDA[combineddf_EDA.SalePrice!='test']


# In[140]:


#groupby()-mean and barchart with datalabels
#sale price and overallcond
#saleprice and utilities
#saleprice and mszoning
#saleprice and bldgtype
#saleprice and housestyle


# In[141]:


housedf_EDA.info()


# In[142]:


import seaborn as sns


# In[143]:


housedf_EDA.SalePrice.groupby(housedf_EDA.Utilities).mean()


# In[144]:


housedf_EDA.SalePrice.groupby(housedf_EDA.MSZoning).mean()


# In[145]:


housedf_EDA.SalePrice.groupby(housedf_EDA.BldgType).mean()


# In[146]:


housedf_EDA.SalePrice.groupby(housedf_EDA.HouseStyle).mean()


# In[147]:


ax=housedf_EDA.SalePrice.groupby(housedf_EDA.SaleCondition).mean().sort_values(
    ascending=False).plot(kind='bar',color='green')
for i in ax.containers:
    ax.bar_label(i,fontsize=6,color="red")


# In[148]:


ax=housedf_EDA.SalePrice.groupby(
    housedf_EDA.Utilities).mean().sort_values(
    ascending=False).plot(kind='bar',color='green')
for i in ax.containers:
    ax.bar_label(i,fontsize=6,color="red")


# In[149]:


ax=housedf_EDA.SalePrice.groupby(
    housedf_EDA.MSZoning).mean().sort_values(
    ascending=False).plot(kind='bar',color='green')
for i in ax.containers:
    ax.bar_label(i,fontsize=6,color="red")


# In[150]:


ax=housedf_EDA.SalePrice.groupby(
    housedf_EDA.BldgType).mean().sort_values(
    ascending=False).plot(kind='bar',color='brown')
for i in ax.containers:
    ax.bar_label(i,fontsize=6,color="green")


# In[151]:


ax=housedf_EDA.SalePrice.groupby(
    housedf_EDA.HouseStyle).mean().sort_values(
    ascending=False).plot(kind='bar')
for i in ax.containers:
    ax.bar_label(i,fontsize=6,color="red")


# In[152]:


#cross tabulation and stacked bar plots with datalabels
#neighbhourhood and bldgtype
#neighbourhood and housestyle
#neighbood and mszoing


# In[153]:


ax=pd.crosstab(housedf_EDA.Neighborhood,housedf_EDA.BldgType).plot(
    kind='bar',stacked=True)
for i in ax.containers:
    ax.bar_label(i,fontsize=6,color="Red")


# In[154]:


ax=pd.crosstab(housedf_EDA.Neighborhood,housedf_EDA.HouseStyle).plot(
    kind='bar',stacked=True)
for i in ax.containers:
    ax.bar_label(i,fontsize=6,color="Red")


# In[155]:


ax=pd.crosstab(housedf_EDA.Neighborhood,housedf_EDA.MSZoning).plot(
    kind='bar',stacked=True)
for i in ax.containers:
    ax.bar_label(i,fontsize=6,color="Red")


# In[156]:


numcols.head()


# In[157]:


#scaling of numcols
from sklearn.preprocessing import StandardScaler


# In[158]:


numcols_scaled=StandardScaler().fit_transform(numcols)


# In[159]:


numcols_scaled=pd.DataFrame(numcols_scaled,columns=numcols.columns)


# In[160]:


#check multicollinearity(do not include ID)
numcols.corr()


# In[161]:


numcols.cov()


# In[162]:


plt.figure(figsize=(30,15))
sns.heatmap(numcols_scaled.drop('Id',axis=1).corr(),annot=True,cmap="plasma")


# In[163]:


objcols.columns


# In[164]:


numcols_scaled.columns


# In[165]:


catcols.columns


# In[166]:


numcols_scaled=numcols_scaled.reset_index()


# In[167]:


objcols=objcols.reset_index()


# In[168]:


numcols_scaled['SalePrice']=objcols.SalePrice


# In[169]:


objcols=objcols.drop('SalePrice',axis=1)


# In[170]:


from sklearn.preprocessing import LabelEncoder


# In[171]:


objcols_encode=objcols.apply(LabelEncoder().fit_transform)


# In[172]:


catcols_encode=catcols.apply(LabelEncoder().fit_transform)


# In[173]:


numcols_scaled=numcols_scaled.reset_index()


# In[174]:


objcols_encode=objcols_encode.reset_index()


# In[175]:


catcols_encode=catcols_encode.reset_index()


# In[176]:


housedf_clean=pd.concat([numcols_scaled,objcols_encode,catcols_encode],axis=1)


# In[177]:


combinedf=combinedf.reset_index()


# In[178]:


housedf_clean['SalePrice']=combinedf.SalePrice


# In[179]:


housedf_train=housedf_clean[housedf_clean.SalePrice!='test']
housedf_test=housedf_clean[housedf_clean.SalePrice=='test']


# In[180]:


housedf_test=housedf_test.drop('SalePrice',axis=1)


# In[181]:


y=housedf_train.SalePrice


# In[182]:


X=housedf_train.drop('SalePrice',axis=1)


# In[183]:


X=X.drop(['level_0','index','Id'],axis=1)


# In[184]:


housedf_test=housedf_test.drop(['level_0','index','Id'],axis=1)


# In[185]:


from sklearn.linear_model import LinearRegression


# In[186]:


reg=LinearRegression()


# In[187]:


regmodel=reg.fit(X,y)


# In[188]:


regpredit=regmodel.predict(housedf_test)


# In[189]:


regmodel.score(X,y)


# In[190]:


from sklearn.tree import DecisionTreeRegressor


# In[191]:


tree=DecisionTreeRegressor(max_depth=8)


# In[192]:


treemodel=tree.fit(X,y)


# In[193]:


treemodel.score(X,y)


# In[194]:


from sklearn.model_selection import cross_val_score


# In[195]:


cross_val_score(tree,X,y)


# In[196]:


np.mean(cross_val_score(tree,X,y))


# In[197]:


tree_test_predict=treemodel.predict(housedf_test)


# In[198]:


pd.DataFrame(tree_test_predict).to_csv("house.csv")


# In[199]:


#RANDOM FOREST REGRESSION


# In[200]:


from sklearn.ensemble import RandomForestRegressor


# In[201]:


RF=RandomForestRegressor(n_estimators=100)


# In[202]:


RFmodel=RF.fit(X,y)


# In[203]:


RFmodel.score(X,y)


# In[204]:


cross_val_score(RF,X,y)


# In[205]:


np.mean([0.87759278, 0.84218557, 0.87943688, 0.87789715, 0.80240176])


# In[206]:


RFpredict=RFmodel.predict(X)


# In[207]:


RFresid=y-RFpredict


# In[208]:


np.sqrt(np.mean(RFresid**2))


# In[209]:


RF_test_pred=RFmodel.predict(housedf_test)


# In[210]:


pd.DataFrame(RF_test_pred).to_csv("randomforest.csv")


# In[211]:


from sklearn.ensemble import GradientBoostingRegressor


# In[212]:


GBM=GradientBoostingRegressor(n_estimators=100)


# In[213]:


gbm_model=GBM.fit(X,y)


# In[214]:


gbm_model.score(X,y)


# In[215]:


cross_val_score(GBM,X,y)


# In[216]:


np.mean([0.89853129, 0.83716772, 0.90206256, 0.88977605, 0.89204657])


# In[217]:


gbm_predict=gbm_model.predict(X)


# In[218]:


gbm_residue=y-gbm_predict


# In[219]:


np.sqrt(np.mean(gbm_residue**2))


# In[ ]:




