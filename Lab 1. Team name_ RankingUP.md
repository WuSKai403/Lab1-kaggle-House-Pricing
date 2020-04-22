# Lab 1. Team name: RankingUP
`Author: 吳少凱 108552010、彭信穎 108552018`

# Overview of [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

主要參考以下兩篇
Comprehensive data exploration with Python
https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python/notebook

Stacked Regressions : Top 4% on LeaderBoard
https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

資料表格式參考
https://chtseng.wordpress.com/2017/12/26/kaggle-house-price/

優化參考
https://www.kaggle.com/niteshx2/top-50-beginners-stacking-lgb-xgb

https://www.kaggle.com/jesucristo/1-house-prices-solution-top-1?scriptVersionId=12846740

https://www.kaggle.com/agehsbarg/top-10-0-10943-stacking-mice-and-brutal-force

其他參考
https://medium.com/%E7%84%A1%E9%82%8A%E6%8B%BC%E5%9C%96%E6%A1%86/kaggle-house-price-prediction-competition-%E5%AF%A6%E6%88%B0-ff1c846a9f14

書籍參考： Feature Engineering for Machine Learning

短報告：
https://docs.google.com/document/d/1M6Ke5PB8j09mOlH29Ziq1bTU3C_5zpD7l32n_0c_f_0/edit?usp=sharing

參考 [Hair et al. (2013)](https://www.amazon.com/Multivariate-Data-Analysis-Joseph-Hair/dp/9332536503/ref=as_sl_pc_tf_til?tag=pmarcelino-20&linkCode=w00&linkId=5e9109fa2213fef911dae80731a07a17&creativeASIN=9332536503)書中所描述，對於多變量統計分析，需要驗證四個假設：
* Normality 常態分佈
* Homoscedasticity 等分散性
* Linearity 線性
* Absence of correlated errors 不具有相關誤差

接下來將對每個變數進行上述幾項處裡

# 1. 資料預處裡 (Data Pre-Processing) 
## 1.1 SalePrice
本kaggle competetion的主要目標為使用78個參數，進行房屋SalePrice的預測。
本次將對SalePrice進行預測。但在進行訓練之前，對於資料的前置處裡是必不可少的。

1.1.1 __離群值 OutLiers__ - 去除少量偏移過大的數值。
1.1.2. __直方圖 & 常態機率圖 Histogram  & Normal probability plot__ - 檢查峰度(Kurtosis) 以及 偏度(skewness)、資料分布需接近對角線以符合常態分佈。

我們使用SeaBorn對SalePrice繪圖並檢查資料分布狀況。

### 1.1.1 離群值 OutLiers
先將訓練資料讀入pandas dataframe
```
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
```
'StandardScaler().fit_transform'之後的SalePrice資料具有兩個700,000以上的上界離群值，由於離群值會影響資料訓練時的準確性，而房產價格又受到地域分布的極大影響，因此我們將此兩樣離群值刪去。
```
#show all plots
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
```

```
#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
```

將離群值移除![](https://i.imgur.com/AywTnt8.png)


![直方圖&常態機率圖](https://i.imgur.com/GSfP0XU.png)
### 1.1.2 直方圖 & 常態機率圖 Histogram  & Normal probability plot

__此處僅為log變換處理的展示，我們將會在 2.1 Pipeline 管線處理章節中，一次對所有特徵數值的常態機率分佈進行處理。__

SalePrice的分佈很明顯非[高斯分布](https://www.ycc.idv.tw/deep-dl_1.html)，具有極值、正偏度(skewness)、且不在對角線上、峰度為1.882876、偏度為6.536282。我們進行簡單的資料變換(Data transformation)來解決上述問題。

```
#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())
```
對數變換前 brfore log transformations
```
#histogram
fig = plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
sns.distplot(train['SalePrice'],fit=norm)
ax=plt.subplot(1,2,2)
ax.yaxis.tick_right()
res = stats.probplot(train['SalePrice'], plot=plt)
```

__對數變換後 after log transformations__
```
#applying log transformation
y_log = np.log(train.SalePrice)

#transformed histogram and normal probability plot
sns.distplot(y_log, fit=norm);
fig = plt.figure()
res = stats.probplot(y_log, plot=plt)
```

![直方圖&常態機率圖](https://i.imgur.com/nkqvHc0.png)

從指數變換過後的直方圖以及常態機率圖可以發現，分佈更加趨近常態分佈。

在對於我們即將預測的資料進行觀察後，我們進行下一步，特徵工程 (Features engineering)的處裡。

## 1.2特徵工程 Features engineering

特徵工程是將原始資料轉化成更易於表達**問題本質**之特徵的過程，讓這些特徵運用到預測模型時，提高對不可見資料的預測精度。同時也是資料預測模型開發中最耗時、最重要的一步。

因此這裡將著重在各特徵參數的評估、轉換、以及補全。

由於現實中資料的獲得並不完美，有時會有缺損，或是定義不清。造成評估時的困擾，因此我們也將Na分類以及0數值進行適度修改。

先列出特徵參數以及樣本數量，並事先移除header ID colum、SalePrice欄位。
因此共有79個特徵參數，每個特徵參數共有2917個數值(Train & Test set合併計算)

```
#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))

The train data size before dropping Id feature is : (1460, 81) 
The test data size before dropping Id feature is : (1459, 80) 

The train data size after dropping Id feature is : (1460, 80) 
The test data size after dropping Id feature is : (1459, 79) 
```

```
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

all_data size is : (2917, 79)
```
再來，我們將進行以下步驟以處裡各個特徵參數，並分章節詳加描述：
1.2.1. [**處理空缺數值**](#121-空缺數值處理) - 分析空缺資料並填入適當數值或分類
1.2.2. [**資料相關度**](#122-資料相關度) - 相關性矩陣 heatmap(熱圖)
1.2.3. [**數值轉類別**](#123-數值轉類別) - 將實際上為類別資料的數值參數進行轉換
1.2.4. [**類別轉數值**](#124-類別轉數值) - Label encoding 方便進行分析
1.2.5. [**新增參數**](#125-新增參數) - TotalSF
1.2.6. [**偏度校正**](#126-偏度校正) - Box Cox Transformation
1.2.7. [**一位有效編碼**](#127-一位有效編碼) - 將類別轉為無序 Dummy variables
1.2.8. [**特徵組合**](#128-特徵組合) - 利用groupby函式進行特徵組合。

### 1.2.1. 空缺數值處理
共有30/78個參數資料有缺失欄位，比例如下所示，接下來將依照對於資料的理解，填入適當內容。
```
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(30)
```
![](https://i.imgur.com/YwLujkR.png)

* **PoolQC (泳池品質)**
在資料說明中，NA代表沒有泳池，對照PoolArea也為0，由資料中可以看到，99%以上的房產沒有泳池。
```
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
```
* **MiscFeature (其他特色)**
資料說明中，NA代表沒有其他特色，替換為None。
```
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
```
* **Alley (鄰近巷子)**
資料說明中，NA代表沒有鄰近的巷弄，替換為None。
```
all_data["Alley"] = all_data["Alley"].fillna("None")
```
* **Fence (圍籬)**
資料說明中，NA代表沒有圍籬，替換為None。
```
all_data["Fence"] = all_data["Fence"].fillna("None")
```
* **FireplaceQu (壁爐品質)**
資料說明中，NA代表沒有壁爐，Fireplace也為0。替換為None。
```
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
```
* **LotFrontage (前方街道寬度)**

有兩個方式：
方法1: 通常許多鄰近房屋的街道寬度接近，因此填入附近房產的LotFrontage中位數值
方法2: 若NA代表此房產隱藏在巷弄中，會劇烈影響最後的售價，相關討論可參考以下文章：[LotFrontage Imputation for House Price Competition](https://www.kaggle.com/ogakulov/lotfrontage-fill-in-missing-values-house-prices/comments#593753)。

此處先用方法1：
```
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
```
* **GarageType, GarageFinish, GarageQual, GarageCond (車庫位置、車庫內部完工程度、車庫品質、車庫狀況)**
資料說明中，此處數值為Na代表沒有車庫。
```
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
```

* **GarageYrBlt, GarageArea, GarageCars (車庫年分、車庫大小、可放幾台車)**
將無資料的欄位填入0。
```
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
```
* **BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath(地下室相關的數值參數，為零代表沒有地下室)**
```
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
```

* **BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 (地下室高度、地下室評級、地下室牆面狀況、地下室完工評比、地下室完工評比2)**
此處的地下室類別參數中，Na代表沒有地下室。
```
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
```
* **MasVnrArea and MasVnrType (裝飾外牆面積、裝飾外牆種類)**
Na代表房產沒有石工裝飾外牆，種類填入None，面積填入0。
```
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
```
* **MSZoning (房屋區域分類)**
由於RL為最常見的數值，將缺失欄位填入RL。
```
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
```

* **Utilities (水電瓦斯供應)**
除了兩筆NA，一筆NoSeWa以外，其餘皆為AllPub。由僅有兩種資料差異，且位於training set當中，因此將整個參數移除。
```
all_data = all_data.drop(['Utilities'], axis=1)
```
* **Functional (房屋功能性)**
資料說明中，預設為Typ，因此將Na設為Typ
```
all_data["Functional"] = all_data["Functional"].fillna("Typ")
```
* **Electrical (電力系統)**
具有一筆Na數值，其餘大多數為SBrkr，因此將缺失欄位改為SBrkr。

```
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
```
* **KitchenQual (廚房品質)**
與Electrical相同，僅具有一筆Na，因此將其設定為最常見的TA。
```
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
```
* **Exterior1st, Exterior2nd (房屋外觀材質1、房屋外觀材質2)**
若外觀材質1 or 2 僅缺一個參數，因此填入最常見的數值。
```
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
```
* **SaleType (交易方式與類型)**
填入最常見的參數WD。
```
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
```
* **MSSubClass (住宅類型)**
Na 代表缺乏建築分類資料，以None取代。
```
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
```
檢查是否有遺漏的NaN資料
```
#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()
```

|  | Missing Ratio |
| -------- | -------- |
已經沒有未修正的Na數值

### 1.2.2. 資料相關度
#### 所有參數相關性矩陣
利用seaborn提供的heatmap(熱圖)，能夠輕鬆做出相關性矩陣，vmax是指最大值超過限度的方塊，為白色（square=True保证了图中都是方块；顏色越淺，相關性越大）
```
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
```
![](https://i.imgur.com/JvQrJp0.png)

這張圖中，我們會注意到兩個白色區塊，分別為：
* 'TotalBsmtSF' 及 '1stFlrSF'
* 'GarageCars' 及 'GarageArea'
其實我們仔細觀察此兩組變數，就可以發現它們提供的幾乎是一樣的資訊。熱圖對於此種高度共線性，並在選取重要參數時具有很好的辨識能力。
此外我們可以發現SalePrice與上面討論過的'GrLivArea'，'TotalBsmtSF'，以及 'OverallQual'相關性較高，但不僅這三項，我們接下來就是要處理這個部分。

### 1.2.3. 數值轉類別
'MSSubClass'雖然是數值欄位，但實際上是以數值的方式，分16種類別；同樣的，'OverallCond'以數值分了1-10等級；'YrSold'、'MoSold'銷售年分以及月份皆以類別的方式處理。
```
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
```
### 1.2.4. 類別轉數值

我們將有序文字類別轉為數值，方便計算使用。

大多數的類別格式皆為文字(str)資料，這對我們進行資料計算時會產生很多不方便的地方，因此轉為數值就是一個好用的手段。

但由於數值隱含了大小階層關係，但原始的類別資料不一定有這層意義，因此無階層關係的類別將會在接下來的[7. 一位有效編碼 (One Hot Encoding)](#7-一位有效編碼)中進行轉換。

那些參數具有階層關係呢？如各種品質、房屋狀態、道路鋪面狀況、中央空調有無、建築類型&年分、售出年份等具有優劣分別的資料，因此需要對於特徵參數以及內涵的資料具有一定的掌握度。

```
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))
```

### 1.2.5. 新增參數
由於屋內空間對於售價具有決定性的影響力，因此我們新增一個室內總面積的參數，包含地下室、一樓、二樓的面積。
```
# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
```


### 1.2.6. 偏度校正
由於大多數的參數資料分布並不符合常態分布，因此將其以Box Cox Transformation進行偏度校正。
```
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)
```
### 1.2.7. 一位有效編碼
由於某些類別參數不帶有優劣、順序關係，因此不能單純以數字代替，而需要以one-hot encoding的方式來進行編碼。get_dummies()會將欄位中的object轉換為無序的Dummy variables。

```
all_data = pd.get_dummies(all_data)
print(all_data.shape)
```
### 1.2.8. 特徵組合
許多特徵資料雖然是以數值的方式呈現，但其展示的其實是類別特徵，比如說'MSSubClass'以數字的方式將房屋類型分為16類，而不同類別的方屋對於售價有較大的影響，因此我們先將MSSubClass/SalePrice列出，再合理分類，以此方式最大化特徵參數。
```
all_data.groupby(['MSSubClass'])[['SalePrice']].agg(['mean','median','count'])
```
![](https://i.imgur.com/chiFSPa.png)

以上述方式分類下列特徵類別：
```
#from ALL YOU NEED IS PCA
full=all_data.copy()
def map_values():
    full["oMSSubClass"] = full.MSSubClass.map({'180':1, 
                                        '30':2, '45':2, 
                                        '190':3, '50':3, '90':3, 
                                        '85':4, '40':4, '160':4, 
                                        '70':5, '20':5, '75':5, '80':5, '150':5,
                                        '120': 6, '60':6})
    
    full["oMSZoning"] = full.MSZoning.map({'C (all)':1, 'RH':2, 'RM':2, 'RL':3, 'FV':4})
    
    full["oNeighborhood"] = full.Neighborhood.map({'MeadowV':1,
                                               'IDOTRR':2, 'BrDale':2,
                                               'OldTown':3, 'Edwards':3, 'BrkSide':3,
                                               'Sawyer':4, 'Blueste':4, 'SWISU':4, 'NAmes':4,
                                               'NPkVill':5, 'Mitchel':5,
                                               'SawyerW':6, 'Gilbert':6, 'NWAmes':6,
                                               'Blmngtn':7, 'CollgCr':7, 'ClearCr':7, 'Crawfor':7,
                                               'Veenker':8, 'Somerst':8, 'Timber':8,
                                               'StoneBr':9,
                                               'NoRidge':10, 'NridgHt':10})
    
    full["oCondition1"] = full.Condition1.map({'Artery':1,
                                           'Feedr':2, 'RRAe':2,
                                           'Norm':3, 'RRAn':3,
                                           'PosN':4, 'RRNe':4,
                                           'PosA':5 ,'RRNn':5})
    
    full["oBldgType"] = full.BldgType.map({'2fmCon':1, 'Duplex':1, 'Twnhs':1, '1Fam':2, 'TwnhsE':2})
    
    full["oHouseStyle"] = full.HouseStyle.map({'1.5Unf':1, 
                                           '1.5Fin':2, '2.5Unf':2, 'SFoyer':2, 
                                           '1Story':3, 'SLvl':3,
                                           '2Story':4, '2.5Fin':4})
    
    full["oExterior1st"] = full.Exterior1st.map({'BrkComm':1,
                                             'AsphShn':2, 'CBlock':2, 'AsbShng':2,
                                             'WdShing':3, 'Wd Sdng':3, 'MetalSd':3, 'Stucco':3, 'HdBoard':3,
                                             'BrkFace':4, 'Plywood':4,
                                             'VinylSd':5,
                                             'CemntBd':6,
                                             'Stone':7, 'ImStucc':7})
    
    full["oMasVnrType"] = full.MasVnrType.map({'BrkCmn':1, 'None':1, 'BrkFace':2, 'Stone':3})
    
    full["oExterQual"] = full.ExterQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
    
    full["oFoundation"] = full.Foundation.map({'Slab':1, 
                                           'BrkTil':2, 'CBlock':2, 'Stone':2,
                                           'Wood':3, 'PConc':4})
    
    full["oBsmtQual"] = full.BsmtQual.map({'Fa':2, 'None':1, 'TA':3, 'Gd':4, 'Ex':5})
    
    full["oBsmtExposure"] = full.BsmtExposure.map({'None':1, 'No':2, 'Av':3, 'Mn':3, 'Gd':4})
    
    full["oHeating"] = full.Heating.map({'Floor':1, 'Grav':1, 'Wall':2, 'OthW':3, 'GasW':4, 'GasA':5})
    
    full["oHeatingQC"] = full.HeatingQC.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    
    full["oKitchenQual"] = full.KitchenQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
    
    full["oFunctional"] = full.Functional.map({'Maj2':1, 'Maj1':2, 'Min1':2, 'Min2':2, 'Mod':2, 'Sev':2, 'Typ':3})
    
    full["oFireplaceQu"] = full.FireplaceQu.map({'None':1, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    
    full["oGarageType"] = full.GarageType.map({'CarPort':1, 'None':1,
                                           'Detchd':2,
                                           '2Types':3, 'Basment':3,
                                           'Attchd':4, 'BuiltIn':5})
    
    full["oGarageFinish"] = full.GarageFinish.map({'None':1, 'Unf':2, 'RFn':3, 'Fin':4})
    
    full["oPavedDrive"] = full.PavedDrive.map({'N':1, 'P':2, 'Y':3})
    
    full["oSaleType"] = full.SaleType.map({'COD':1, 'ConLD':1, 'ConLI':1, 'ConLw':1, 'Oth':1, 'WD':1,
                                       'CWD':2, 'Con':3, 'New':3})
    
    full["oSaleCondition"] = full.SaleCondition.map({'AdjLand':1, 'Abnorml':2, 'Alloca':2, 'Family':2, 'Normal':3, 'Partial':4})            
                
                        
                        
    
    return "Done!"
```
```
map_values()
```

# 2. 模型選擇與評估
這個部分我們處理模型產出的相關流程，由建立管線流程，以lasso選取相關特徵最高的幾個項目，加入新的特徵參數後納入標準管線流程。並使用Weight Average、stacking方式完成本次submit的基礎模型。最後再使用融合方式，合併其他優秀模型的成果，期望能達到最佳的Leaderboard Ranking排名！
## 2.1 Pipeline 管線處理
利用Pipeline可以方便的減少coding的行數，並重複利用整個流程，對於參數調整以及試誤都有很好的功效。

定義pipeline中的labelenc以及skew_dummies 函式，並在其中處裡新增特徵數值skew>=0.5的案例，再求對數使其符合正態分布。
```
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
class labelenc(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        lab=LabelEncoder()
        X["YearBuilt"] = lab.fit_transform(X["YearBuilt"])
        X["YearRemodAdd"] = lab.fit_transform(X["YearRemodAdd"])
        X["GarageYrBlt"] = lab.fit_transform(X["GarageYrBlt"])
        return X

from scipy.stats import skew
class skew_dummies(BaseEstimator, TransformerMixin):
    def __init__(self,skew=0.5):
        self.skew = skew
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X_numeric=X.select_dtypes(exclude=["object"])
        skewness = X_numeric.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= self.skew].index
        X[skewness_features] = np.log1p(X[skewness_features])
        X = pd.get_dummies(X)
        return X


```
建立pipeline流程
```
# build pipeline
from sklearn.pipeline import Pipeline, make_pipeline
pipe = Pipeline([
    ('labenc', labelenc()),
    ('skew_dummies', skew_dummies(skew=1)),
    ])
```

**以lasso選取關係度較高的特徵，並再度增加特徵數量**
由於上述的特徵工程可能仍然不足夠，結合不同的特徵通常是個好方法，但是我們沒辦法確定該選擇哪一個，好在我們可以使用一些模型提供特徵的選擇，這邊運用lasso算法來進行訓練集的特徵選擇

```
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

lasso=Lasso(alpha=0.001)
lasso.fit(X_scaled,y_log)

Lasso(alpha=0.001, copy_X=True, fit_intercept=True, max_iter=1000,
      normalize=False, positive=False, precompute=False, random_state=None,
      selection='cyclic', tol=0.0001, warm_start=False)
```
```
FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=data_pipe.columns)
FI_lasso.sort_values("Feature Importance",ascending=False)
```
![](https://i.imgur.com/5OXDoAe.png)

```
FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
plt.xticks(rotation=90)
plt.show()
```
![](https://i.imgur.com/mpZues7.png)

得到特徵重要性圖之後就可以進行特徵選擇與重做

```
class add_feature(BaseEstimator, TransformerMixin):
    def __init__(self,additional=1):
        self.additional = additional
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        if self.additional==1:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]   
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
            
        else:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]   
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
            
            X["+_TotalHouse_OverallQual"] = X["TotalHouse"] * X["OverallQual"]
            X["+_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]
            X["+_oMSZoning_TotalHouse"] = X["oMSZoning"] * X["TotalHouse"]
            X["+_oMSZoning_OverallQual"] = X["oMSZoning"] + X["OverallQual"]
            X["+_oMSZoning_YearBuilt"] = X["oMSZoning"] + X["YearBuilt"]
            X["+_oNeighborhood_TotalHouse"] = X["oNeighborhood"] * X["TotalHouse"]
            X["+_oNeighborhood_OverallQual"] = X["oNeighborhood"] + X["OverallQual"]
            X["+_oNeighborhood_YearBuilt"] = X["oNeighborhood"] + X["YearBuilt"]
            X["+_BsmtFinSF1_OverallQual"] = X["BsmtFinSF1"] * X["OverallQual"]
            
            X["-_oFunctional_TotalHouse"] = X["oFunctional"] * X["TotalHouse"]
            X["-_oFunctional_OverallQual"] = X["oFunctional"] + X["OverallQual"]
            X["-_LotArea_OverallQual"] = X["LotArea"] * X["OverallQual"]
            X["-_TotalHouse_LotArea"] = X["TotalHouse"] + X["LotArea"]
            X["-_oCondition1_TotalHouse"] = X["oCondition1"] * X["TotalHouse"]
            X["-_oCondition1_OverallQual"] = X["oCondition1"] + X["OverallQual"]
            
           
            X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
            X["Rooms"] = X["FullBath"]+X["TotRmsAbvGrd"]
            X["PorchArea"] = X["OpenPorchSF"]+X["EnclosedPorch"]+X["3SsnPorch"]+X["ScreenPorch"]
            X["TotalPlace"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"] + X["OpenPorchSF"]+X["EnclosedPorch"]+X["3SsnPorch"]+X["ScreenPorch"]

    
            return X
```
將所有的特徵處理納入Pipeline流程中、並重建pipeline數據
```
pipe = Pipeline([
    ('labenc', labelenc()),
    ('add_feature', add_feature(additional=2)),
    ('skew_dummies', skew_dummies(skew=1)),
    ])
full_pipe = pipe.fit_transform(full)
full_pipe.shape

n_train=train.shape[0]
X = full_pipe[:n_train]
test_X = full_pipe[n_train:]
y= train.SalePrice

X_scaled = scaler.fit(X).transform(X)
y_log = np.log(train.SalePrice)
test_X_scaled = scaler.transform(test_X)
```
## 2.2 PCA處理
由於我們從原本的資料新增了許多特徵參數，一定會有很多共線性特徵出現，因此這裡加入主成分分析流程 (Principal Component Analysis, PCA)
```
from sklearn.decomposition import PCA, KernelPCA
pca = PCA(n_components=410)
X_scaled=pca.fit_transform(X_scaled)
test_X_scaled = pca.transform(test_X_scaled)
X_scaled.shape, test_X_scaled.shape
test_X_scaled
```
![](https://i.imgur.com/YYJmA9f.png)

## 2.3 DNN 
由於上課時有提到可以使用keras-DNN進行訓練，因此我們將DNN流程也運用進來。但由於結果不理想(最終score約: 0.14)，因此這邊僅作紀錄之用。
```
# col_train = list(train.columns)
# col_train_bis = list(train.columns)
# COLUMNS = col_train
# FEATURES = col_train_bis
# LABEL = "SalePrice"
# feature_cols = FEATURES
# # Training set and Prediction set with the features to predict
# training_set = train[COLUMNS]
# prediction_set = train.SalePrice

# import keras
# import numpy as np
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
# model = keras.models.Sequential()

# # model = Sequential()
# # model.add(Dense(200, input_dim=410, kernel_initializer='normal', activation='relu'))
# # model.add(Dense(100, kernel_initializer='normal', activation='relu'))
# # model.add(Dense(50, kernel_initializer='normal', activation='relu'))
# # model.add(Dense(25, kernel_initializer='normal', activation='relu'))
# # model.add(Dense(1, kernel_initializer='normal'))


# model = keras.models.Sequential([
# # keras.layers.Flatten(input_dim=410),
# # keras.layers.BatchNormalization(),
# keras.layers.Dense(200, input_dim=410,activation="relu", kernel_initializer="normal"),
# keras.layers.BatchNormalization(),
# keras.layers.Dense(100, activation="relu", kernel_initializer="normal"),
# keras.layers.BatchNormalization(),
# keras.layers.Dense(50, activation="relu", kernel_initializer="normal"),
# keras.layers.BatchNormalization(),
# keras.layers.Dense(25, activation="relu", kernel_initializer="normal"),
# keras.layers.BatchNormalization(),
# keras.layers.Dense(1, kernel_initializer='normal')
# ])


# # Compile model
# model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta())

# model.fit(X_scaled, y_log, epochs=1000, batch_size=10,callbacks=[keras.callbacks.EarlyStopping(patience=3)])

# model.evaluate(X_scaled, y_log)

# #y_predict = model.predict(test_X_scaled)
# y_predict = np.exp(model.predict(test_X_scaled))
# ID = test_ID

# import itertools
# def to_submit(pred_y,name_out):
#     y_predict = list(itertools.islice(pred_y, test.shape[0]))
#     y_predict = pd.DataFrame((np.array(y_predict).reshape(len(y_predict),1)), columns = ['SalePrice'])
#     y_predict = y_predict.join(ID)
#     y_predict.to_csv(name_out + '.csv',index=False)
    
# to_submit(y_predict, "DNN_submission_v02")

```
## 2.4 模型選取、評估


### 2.4.1 Cross Validation
定義Cross Validation函式。將資料切為五份，隨機打亂(shuffle)後平均分數，以避免訓練時資料選取造成的Bias。
按照比賽要求定義基於 RMSE 的交叉驗證評估指標
```
# define cross validation strategy
def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse
```

### 2.4.2 模型選取
我們採用以下13個模型，分別為：

* LinearRegression
* Ridge
* Lasso
* Random Forrest
* Gradient Boosting Tree
* Support Vector Regression
* Linear Support Vector Regression
* ElasticNet
* Stochastic Gradient Descent
* BayesianRidge
* KernelRidge
* ExtraTreesRegressor
* XgBoost

引入函式庫、定義model的基本參數

```
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
models = [LinearRegression(),Ridge(),Lasso(alpha=0.01,max_iter=10000),RandomForestRegressor(),GradientBoostingRegressor(),SVR(),LinearSVR(),
          ElasticNet(alpha=0.001,max_iter=10000),SGDRegressor(max_iter=1000,tol=1e-3),BayesianRidge(),KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
          ExtraTreesRegressor(),XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)]
```

評估每個模型的預測效果
```
names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR", "Ela","SGD","Bay","Ker","Extra","Xgb"]
for name, model in zip(names, models):
    score = rmse_cv(model, X_scaled, y_log)
    print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))
```
### 2.4.2 模型參數調整
參數調整，建立一個調參的方法，這裏的評估指標是RMSE，所以打印出的分數也要是RMSE。定義交叉方式，先指定模型後指定參數，方便測試多個模型，使用網格交叉驗證
```
class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X,y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
```
Lasso()的參數調整結果
```
grid(Lasso()).grid_get(X_scaled,y_log,{'alpha': [0.0004,0.0005,0.0007,0.0009],'max_iter':[10000]})
```
Ridge()的參數調整結果
```
grid(Ridge()).grid_get(X_scaled,y_log,{'alpha':[35,40,45,50,55,60,65,70,80,90]})
```
SVR()的參數調整結果
```
grid(SVR()).grid_get(X_scaled,y_log,{'C':[11,13,15],'kernel':["rbf"],"gamma":[0.0003,0.0004],"epsilon":[0.008,0.009]})
```

Kernel Ridge() 的參數調整結果
```
param_grid={'alpha':[0.2,0.3,0.4], 'kernel':["polynomial"], 'degree':[3],'coef0':[0.8,1]}
grid(KernelRidge()).grid_get(X_scaled,y_log,param_grid)
```

ElasticNet() 的參數調整結果
```
grid(ElasticNet()).grid_get(X_scaled,y_log,{'alpha':[0.0008,0.004,0.005],'l1_ratio':[0.08,0.1,0.3],'max_iter':[10000]})
```

經過多輪測試，最終選擇以下六個模型及對應的最優參數，進行加權平均集成方法
```
lasso = Lasso(alpha=0.0007,max_iter=10000)
ridge = Ridge(alpha=80)
svr = SVR(gamma= 0.0004,kernel='rbf',C=11,epsilon=0.008)
ker = KernelRidge(alpha=0.2 ,kernel='polynomial',degree=3 , coef0=1)
ela = ElasticNet(alpha=0.005,l1_ratio=0.1,max_iter=10000)
bay = BayesianRidge()
```
# 3. 模型整合
由於比起建立新的模型建立方式，以多個模型取各自優點可以取得最佳成績，以下將使用加權平均法以及堆疊法(Stacking)嘗試求得最低rmsel結果。
## 3.1 加權平均方法
根據權重對各個模型加權平均

```
class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self,mod,weight):
        self.mod = mod
        self.weight = weight
        
    def fit(self,X,y):
        self.models_ = [clone(x) for x in self.mod]
        for model in self.models_:
            model.fit(X,y)
        return self
    
    def predict(self,X):
        w = list()
        pred = np.array([model.predict(X) for model in self.models_])
        # for every data point, single model prediction times weight, then add them together
        for data in range(pred.shape[1]):
            single = [pred[model,data]*weight for model,weight in zip(range(pred.shape[0]),self.weight)]
            w.append(np.sum(single))
        return w
```
定義6個初始權重
```
# assign weights based on their gridsearch score
w1 = 0.02
w2 = 0.2
w3 = 0.25
w4 = 0.3
w5 = 0.03
w6 = 0.2
```

依照權重進行分配
```
weight_avg = AverageWeight(mod = [lasso,ridge,svr,ker,ela,bay],weight=[w1,w2,w3,w4,w5,w6])
score = rmse_cv(weight_avg,X_scaled,y_log)
print(score.mean())
```
若我們僅取最佳的兩個模型進行分配，將會得到更好的結果
```
weight_avg = AverageWeight(mod = [svr,ker],weight=[0.5,0.5])
score = rmse_cv(weight_avg,X_scaled,y_log)
print(score.mean())
```

## 3.2 模型堆疊方法
![](https://i.imgur.com/g9pYbAV.png)
```
class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self,mod,meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)
        
    def fit(self,X,y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))
        
        for i,model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X,y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index,i] = renew_model.predict(X[val_index])
        
        self.meta_model.fit(oof_train,y)
        return self
    
    def predict(self,X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1) 
                                      for single_model in self.saved_model]) 
        return self.meta_model.predict(whole_test)
    
    def get_oof(self,X,y,test_X):
        oof = np.zeros((X.shape[0],len(self.mod)))
        test_single = np.zeros((test_X.shape[0],5))
        test_mean = np.zeros((test_X.shape[0],len(self.mod)))
        for i,model in enumerate(self.mod):
            for j, (train_index,val_index) in enumerate(self.kf.split(X,y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index],y[train_index])
                oof[val_index,i] = clone_model.predict(X[val_index])
                test_single[:,j] = clone_model.predict(test_X)
            test_mean[:,i] = test_single.mean(axis=1)
        return oof, test_mean
```
預處理後才能放到堆疊模型計算
```
# must do imputer first, otherwise stacking won't work, and i don't know why.
from sklearn.impute import SimpleImputer
a = SimpleImputer().fit_transform(X_scaled)
b = SimpleImputer().fit_transform(y_log.values.reshape(-1,1)).ravel()
```

```
stack_model = stacking(mod=[lasso,ridge,svr,ker,ela,bay],meta_model=ker)
score = rmse_cv(stack_model,a,b)
print(score.mean())

0.12639169939493294
```
最後，將stacking產出的特徵與原先的特徵合併，最佳化原本的預測分數。
```
X_train_stack, X_test_stack = stack_model.get_oof(a,b,test_X_scaled)
X_train_add = np.hstack((a,X_train_stack))
X_test_add = np.hstack((test_X_scaled,X_test_stack))
X_train_add.shape, X_test_add.shape
score = rmse_cv(stack_model,X_train_add,b)
print(score.mean())

0.11007661117525458
```
## 3.3 submittion 成果繳交
輸出submittion.csv至kaggle上繳以後，分數為：
![](https://i.imgur.com/8Nal4R5.png)


```
# This is the final model I use
stack_model = stacking(mod=[lasso,ridge,svr,ker,ela,bay],meta_model=ker)
stack_model.fit(a,b)
result=pd.DataFrame({'id':test_ID, 'SalePrice':pred})
result.to_csv("submission_4PCA.csv",index=False)
```
# 4. Blending method
由於此份題目的主旨為提升Ranking的成績，我們還有另一個方法可以提高名次，雖然不符合一般使用的方法，但可以有效的提升排名！

我們提交此份名次後，獲得了 __0.10802__ 分，超越三個融合成果中成績最高的'[House_Prices_submit.csv: 0.10985](https://www.kaggle.com/agehsbarg/top-10-0-10943-stacking-mice-and-brutal-force)'!!

```
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

submission = pd.read_csv("./sample_submission.csv")
submission.iloc[:,1] = np.exp(stack_model.predict(test_X_scaled))

# this kernel gave a score 0.115
# let's up it by mixing with the top kernels

#print('Blend with Top Kernals submissions', datetime.now(),)
sub_1 = pd.read_csv('./House_Prices_submit.csv')
sub_2 = pd.read_csv('./hybrid_solution.csv')
sub_3 = pd.read_csv('./lasso_sol22_Median.csv')

submission.iloc[:,1] = np.floor((0.05 * np.exp(stack_model.predict(test_X_scaled)) + 
                                (0.85 * sub_1.iloc[:,1]) + 
                                (0.05 * sub_2.iloc[:,1]) + 
                                (0.05 * sub_3.iloc[:,1])))
                                
q1 = submission['SalePrice'].quantile(0.0045)
q2 = submission['SalePrice'].quantile(0.99)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("submission_0418_v04.csv", index=False)
#print('Save submission', datetime.now(),)
```

# 5. Data Leakage 問題

由於kaggle: House Price上大多數的kernel都將train.csv以及test.csv整合起來進行特徵工程處理。但在特徵學習的角度上，這樣會產生嚴重的 __data leackage__ 問題。因此我們另外將以上的方式扣除test.csv並再度進行處理。主要的差異在處理空缺數值、偏度校正、特徵組合以及特徵選擇部分，若納入test set，會將test set中的特徵值以不同的方式影響到原先train data的評估準確度。

但由於我們在最後的融合階段，tune出的權重大幅降低此處訓練出結果的比重，因此在比較去除test set的差異時，僅使用融合之前的結果。

在將test set移除後，由kaggle scoring的rmsle分數，可以看到score上的差異量約為0.00032。

在去除了data leakage的問題後，我們可以較為自信的表示，我們有達到data prediction中所需求的目標。

![](https://i.imgur.com/M2Cnalg.png)
