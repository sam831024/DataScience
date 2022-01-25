import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.preprocessing import LabelEncoder #encode categoric data
from sklearn.preprocessing import Normalizer #normalize numeric data
from sklearn.model_selection import train_test_split #to split data
from sklearn.neighbors import KNeighborsClassifier #our lovely classifie

for dirname, _, filenames in os.walk(r'C:\Users\a0486121\Desktop\Adult income dataset_KNN'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv(r'C:\Users\a0486121\Desktop\Adult income dataset_KNN\adult.csv')
print (data.head())

def object_cols(df):
    return list(df.select_dtypes(include='object').columns)

def numerical_cols(df):
    return list(df.select_dtypes(exclude='object').columns)

obj_col = object_cols(data)
num_col = numerical_cols(data)
print ('object_cols:',obj_col,'num_cols:',num_col)

le = LabelEncoder()
norm = Normalizer()
#Label encoding
for col in obj_col:
    data[col] = le.fit_transform(data[col])
#Normalize
data[num_col] = norm.fit_transform(data[num_col])
print (data.head())
#K-NN Algorithm with sklearn
X = data.drop(['income'], axis = 1)
y = data['income']
# Dataset Setting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

knn = KNeighborsClassifier() #create a k-NN instance
fit_knn = knn.fit(X_train, y_train) #fitting model by using "fit" method
s_value = X_test[:1] #specific value to predict
print('Predicted class value is : {0}'.format(fit_knn.predict(s_value)))

distance_matrix = fit_knn.kneighbors_graph(s_value, mode = "distance") #distance value matrix

ind = distance_matrix.nonzero() # indexes of non-zero elements

k_nearest_classes = [y_train.iloc[val[1]] for val in zip(*distance_matrix.nonzero())]
print(fit_knn.predict_proba(s_value))
dis = distance_matrix[distance_matrix.nonzero()] # non-zero distances

k_near = pd.DataFrame()
for val in zip(*distance_matrix.nonzero()):
    k_near = k_near.append(X_train.iloc[val[1]], ignore_index = True)

print ('Success Rate:',fit_knn.score(X_test, y_test))
