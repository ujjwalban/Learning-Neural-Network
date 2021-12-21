

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt






dataset = pd.read_csv("C:\\Users\\New\\Churn_Modelling.csv")



X = dataset.iloc[:,3:13]
Y = dataset.iloc[:,13]

geography = pd.get_dummies(X["Geography"],drop_first = True)
gender = pd.get_dummies(X["Gender"],drop_first = True)

X = pd.concat([X,geography,gender], axis = 1)
X = X.drop(['Geography','Gender'],axis = 1)

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout

classifier = Sequential()


classifier.add(Dense(units = 6 , activation = 'relu' ,  input_dim = 11 ))
classifier.add(Dense(units = 6, activation = 'relu' ))
classifier.add(Dense(units= 1, activation = 'sigmoid'))
 




classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])
model_history = classifier.fit(X_train,Y_train , validation_split=0.33,batch_size = 10,epochs = 100)




y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import accuracy_score
score  = accuracy_score(y_pred,Y_test)



score





