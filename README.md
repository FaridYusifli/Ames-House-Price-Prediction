# Ames-House-Price-Prediction
Predicting house prices ([Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)) using Machine Learning techniques.   


1. After data visualization between ‘SalePrice’ and other columns some rows deleted in 8 columns total ('GrLivArea', 'TotalBsmtSF', 'MasVnrArea', 'BsmtFinSF1', 'LotFrontage', 'OpenPorchSF', 'OverallQual', 'GrLivArea') which were the outliers.
2. After correlation matrix, heat map visualization and close look to the ‘suspicious’ predictors (columns), I detect there are multicollinear columns and columns which all values are same, which is absolutely useless.  So they deleted too ('TotRmsAbvGrd','GarageArea','Id','Utilities','GarageYrBlt','Alley','FireplaceQu','PoolQC','Fence', 'MiscFeature','3SsnPorch','MoSold','BsmtFinSF2','BsmtHalfBath','MiscVal')
3. Some columns merged in one newly created column.
*	Total (Total square feet of all area) = Total square feet of basement area + 1st floor square feet + 2nd floor square feet
*	‘TimeRemodel’ = ‘YrSold’ – ‘YearRemodAdd’,etc.
4. There are missing values in most of the columns, so filled them in several different fashion (in notebook it is described). If column is numeric, it was filled with just 0 or the mod of the other rows depending of the predictor, for string filled with “None” or group this particular column based on another column (groupby) and take the mean (e.g. “Neighborhood” vs “LotFrontage”),etc.
5. To handling categorical data dummy variable created (0 or 1)
6. To make “SalePrice” normally distributed, I log transform it
7. First I try to predict with several algorithms (6 or 7) and take the average of the all results but the result was very poor. Then stacking and Grid Search CV used, at this time the result was very good, but running time was poor. At the end I come up with new solution which running time very fast and result is acceptable. Just using the average of Gradient Boosting Regressor and Lasso. One of the key points here in GBR using ‘huber’ (which is combination of ‘ls’ and ‘lad’) as a loss function which improve result significantly   
**Best Result: 0.1145**    
**Last Result: 0.116**
