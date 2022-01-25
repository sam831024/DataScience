#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv(r'C:\Users\a0486121\Desktop\Data_Sc_ML\USA_Housing\USA_Housing.csv')

df.head()
df.info()

#Explore data with diagram 
sns.pairplot(df)


#利用distplot來看房價主要集中的區間
sns.distplot(df['Price'])

#利用df.corr()先做出個變數間的關係係數，再用heatmap作圖
sns.heatmap(df.corr(),annot=True)

#訓練線性模型
#X是想探索的自變數，Y是依變數
df.columns

#準備X & Y array
X = df[['Avg. Area Income','Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

#將資料分成訓練組及測試組
from sklearn.model_selection import train_test_split

#test_size代表測試組比例。random_state代表設定隨機種子，讓測試結果可被重複
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=101)

#載入線性迴歸，並訓練模型
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

#取得截距。如果公式是y=a+bx，a即是截距
print(lm.intercept_)

#取得迴歸係數，並用Data Frame顯示
lm.coef_
X_train.columns

#從迴歸係數中，看起來平均房齡及平均房間數影響房價甚大，但由於這是假資料，所以課程建議去練習真實的Boston資料集
cdf = pd.DataFrame(lm.coef_,X_train.columns,columns=['Coef'])
cdf

#預測
#使用測試組資料來預設結果
predictions = lm.predict(X_test)
predictions

#比較實際房價及預測房價的關係
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions))

#評估模型好壞
#載入迴歸常見的評估指標
from sklearn import metrics

#Mean Absolute Error(MAE)代表平均誤差，公式為所有實際值及預測值相減的絕對值平均
metrics.mean_absolute_error(y_test,predictions)

#Mean Squared Error(MSE)比起MSE可以拉開誤差差距，算是蠻常用的指標。公式為所有實際值及預測值相減的平方的平均
metrics.mean_squared_error(y_test,predictions)

#Root Mean Squared Error(RMSE)代表MSE的平分根。比起MSE更為常用，因為更容易解釋y
np.sqrt(metrics.mean_squared_error(y_test,predictions))

