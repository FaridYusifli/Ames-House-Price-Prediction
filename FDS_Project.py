import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso

def fillColumn(lst,dataset,typee):
    if typee == "Num":
        for col in lst:
            dataset[col].fillna(0, inplace = True) 
    elif typee == "Str":
        for col in lst:
            dataset[col].fillna('None', inplace = True)
    elif typee == "":
        for col in lst:
            dataset[col].fillna(dataset[col].mode()[0],inplace=True)
    else:
        print("wrong parameter sended to <fillColumn> function")
        sys.exit()
        
def meanSum(*args):
    return sum(args)/len(args)

#read train and test set
train=pd.read_csv('dataset/train.csv')
test=pd.read_csv('dataset/test.csv')
ID_test = test['Id']

#after feature engineering and data visualization,delete some outliers
train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index, inplace = True)
train.drop(train[(train['TotalBsmtSF'] > 5000) & (train['SalePrice'] < 300000)].index, inplace = True)
train.drop(train[(train['MasVnrArea'] > 1500) & (train['SalePrice'] < 300000)].index, inplace = True)
train.drop(train[(train['BsmtFinSF1'] > 5000) & (train['SalePrice'] < 300000)].index, inplace = True)
train.drop(train[(train['LotFrontage'] > 250) & (train['SalePrice'] < 300000)].index, inplace = True)
train.drop(train[(train['OpenPorchSF'] > 400) & (train['SalePrice'] < 100000)].index, inplace = True)
train.drop(train[(train['OverallQual'] <5) & (train['SalePrice']>200000)].index, inplace=True)
train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index, inplace=True)

#merge test and train set
all_data = pd.concat([train,test]).reset_index(drop=True)

#drop useless columns (some are multicollinear,some has same values for each row)
all_data.drop(['TotRmsAbvGrd','GarageArea','Id','Utilities','GarageYrBlt','Alley','FireplaceQu','PoolQC','Fence',               'MiscFeature','3SsnPorch','MoSold','BsmtFinSF2','BsmtHalfBath','MiscVal'],axis=1,inplace = True)
all_data['Total'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF'] 
all_data['TimeRemodel'] = all_data['YrSold'].astype(int) - all_data['YearRemodAdd'].astype(int)
all_data['FullQuality'] = all_data['OverallQual'] + all_data['OverallCond']

#log transformation
y = np.log1p(train['SalePrice'])
all_data.drop(['SalePrice'], inplace=True,axis=1)
all_data.reset_index(drop=True,inplace=True)

#Filling 'LotFrontage' based on mean value in each neighborhood
group = all_data.groupby('Neighborhood')['LotFrontage'].mean()
for index in range(0,all_data.shape[0]):
    if np.isnan(all_data.loc[index,'LotFrontage']):
        all_data.loc[index,'LotFrontage'] = group[all_data.loc[index,'Neighborhood']]
        
# Na -> None
fillColumn(["GarageCond","GarageFinish","GarageQual","GarageType","BsmtFinType2","BsmtExposure","BsmtQual",            "BsmtFinType1","BsmtCond","MasVnrType"],all_data,"Str")
#Na -> 0
fillColumn(["MasVnrArea","Total","BsmtFullBath","BsmtFinSF1","BsmtUnfSF","TotalBsmtSF",            "GarageCars","Functional","KitchenQual"],all_data,"Num")

# if value is 'na' then replace it with most common value
fillColumn(['Electrical','MSZoning','SaleType','Exterior1st','Exterior2nd'],all_data,"")

all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)

# creating dummy variables (0,1) from categorical variables
all_data = pd.get_dummies(all_data,drop_first=True)


scaler=RobustScaler()

train = scaler.fit_transform(all_data[:train.shape[0]])
test = scaler.transform(all_data[train.shape[0]:].reset_index(drop=True))

Lasso_model_final= Lasso(alpha=0.001).fit(train, y)
GBS_model_final=GradientBoostingRegressor(loss='huber', n_estimators=4421,min_samples_split=210,min_samples_leaf=6, 
                                          learning_rate=0.05,max_depth=2,max_features=27,subsample=0.8).fit(train, y)

labels_lasso = np.expm1(Lasso_model_final.predict(test))
labels_GBS=np.expm1(GBS_model_final.predict(test))

pd.DataFrame({'Id': ID_test, 'SalePrice': meanSum(labels_lasso,labels_GBS)}).to_csv('submission.csv', index =False) 

