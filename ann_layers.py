#6/18/18
#Building an artificial neural network for predicting if a customer will return and 
#buy another product or not 
#import libraries 
import theano
import tensorflow
import keras 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#import the dataset 
churn=pd.read_csv("Churn_Modelling.csv")
X=churn.iloc[:,3:13].values
y=churn.iloc[:,13].values 

#for encoding the categorical variables 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder 
labelencoder_X_1=LabelEncoder() 

#scale and one-hot encode categorical data 
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

#scale gender
labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:] 

#split train and test set 
from sklearn.cross_validation import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.28,random_state=0)

#scaling the features
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

#libraries for the ANN 
import keras
from keras.models import Sequential 
from keras.layers import Dense 

#initializing the ann (regularly connected neural network)
classifier=Sequential() 
#adding the input layers and the first hidden layer 
classifier.add(Dense(output_dim=6,init="uniform",activation='relu',input_dim=11)) #choose hidden layers by average of nodes in the input and output layers? or use k-fold cross-validation? (por artistas)
classifier.add(Dropout(p=0.1)) #10% of neurons are disabled 
#adding a new hidden layer 
classifier.add(Dense(output_dim=6,init="uniform",activation="relu"))
classifier.add(Dropout(p=0.2)) #20% of neurons are disabled 
#adding the output layer 
classifier.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))

#compliling the ann
#optimizer-find the optimal set of weights in the nn 
#loss w/n the sgd (adam) algorithm (optimize loss to find the ideal weights)
#binary-binary_cross, 2+ outcomes=cat_cross (for loss parameter)
#metrics-->accuracy
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fitting the ann to the training set 
classifier.fit(X_train,y_train,batch_size=32,nb_epoch=10)

#make predictions and evaluate 
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)
y_pred.shape 
#predicting a single obs 
new_pred=classifier.predict(scaler.transform(np.array([[0,1,34]])))

from sklearn.metrics import confusion_matrix 
cm=confusion_matrix(y_test,y_pred)

#ANN with k-fold validation 
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score 
def build_classifier(): #local variable 
    classifier=Sequential() 
    classifier.add(Dense(output_dim=6,init="uniform",activation='relu',input_dim=11)) #choose hidden layers by average of nodes in the input and output layers 
    classifier.add(Dense(output_dim=6,init="uniform",activation="relu"))
    classifier.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier 

classifier=KerasClassifier(build_fn=build_classifier,batch_size=3,nb_epoch=3)

#estimator-fit the data
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=4,n_jobs= -1)
mean=accuracies.mean()
variance=accuracies.std() 

###improving ann
#dropout regularization (prevents overfitting)
#high variance (when applying k-fold cv)--indicates high overfitting 
from keras.layers import Dropout 

#finetuning the ANN 
from sklearn.model_selection import GridSearchCV
def build_classifier(): 
    classifier=Sequential() 
    classifier.add(Dense(output_dim=6,init="uniform",activation='relu',input_dim=11)) #choose hidden layers by average of nodes in the input and output layers? or use k-fold cross-validation? (por artistas)
    classifier.add(Dense(output_dim=6,init="uniform",activation="relu"))
    classifier.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier 

classifier=KerasClassifier(build_fn=build_classifier) 
parameters={'batch_size':[25,32],'nb_epoch':[6,10], #create a parameter dictionary 
'optimizer':['adam','rmsprop']}
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,
scoring='accuracy',cv=10) #cv-number of folds 
grid_search=grid_search.fit(X_train,y_train) #fit grid search to the training set 
best_parameters=grid_search.best_params_ 
best_accuracy=grid_search.best_score_
