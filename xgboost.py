# Support Vector Classification
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values


#Encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder;
xlabelencoder=LabelEncoder();
x2labelencoder=LabelEncoder();
X[:,1]=xlabelencoder.fit_transform(X[:,1]);
X[:,2]=x2labelencoder.fit_transform(X[:,2]);
xonehotencoder=OneHotEncoder();
p=X[:,1];
dv=xonehotencoder.fit_transform(p.reshape(-1,1)).toarray();
dv=dv[:,1:]; #dummy variable trap
X=np.concatenate((X[:,[0]],dv[:],X[:,2:]),axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.svm import SVC;
classifier_svc = SVC(kernel='linear', random_state=0)
classifier_svc.fit(X_train, y_train)

y_pred=classifier_svc.predict(X_test);

#confusion metrics
from sklearn.metrics import confusion_matrix;
cm=confusion_matrix(y_test, y_pred);
print(cm)

#accuracies k-fold
from sklearn.model_selection import cross_val_score;
accuracy=cross_val_score(estimator=classifier_svc, X=X_train,y=y_train,cv=10);
mean=accuracy.mean();
variance=accuracy.std();
print('SVC--->',f'Accuracy: {mean}',f'Variance: {variance}');

#Xgboost
from xgboost import XGBClassifier;

classifier =XGBClassifier();
classifier.fit(X_train,y_train);
y_pred=classifier.predict(X_test);

#confusion metrics
from sklearn.metrics import confusion_matrix;
cm=confusion_matrix(y_test, y_pred);
print(cm)

#accuracies k-fold
from sklearn.model_selection import cross_val_score;
accuracy=cross_val_score(estimator=classifier, X=X_train,y=y_train,cv=10);
mean=accuracy.mean();
variance=accuracy.std();
print('XGBOOST--->',f'Accuracy: {mean}',f'Variance: {variance}');
