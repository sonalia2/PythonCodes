# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 19:50:32 2019

@author: 500035501
"""


import pandas as pd
from pandas import Series,DataFrame
import numpy as np

#to display all columns and rows in console
pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns',50)
pd.set_option('display.width',1000)
pd.set_option('display.precision',2)



###Credit Risk Analytics using SVM & other models in Python -using  Validate Data
###Use ‘Validate’ data  and select the best Model ( after complete Evaluation )
 

df = pd.read_csv('Python_Module_Day_15.2_Credit_Risk_Train_data.csv')
df.info()   
df2 = pd.read_csv('Python_Module_Day_15.4_Credit_Risk_Validate_data.csv')
df2.info()
df2.describe()
df2.describe().transpose() 
df2.describe(include='O')
df.info()
df.describe()
df.describe(include='O')

#Count missing values
df.isna().sum()
df2.isna().sum()

#Drop unwanted column
df2=df2.drop(['Loan_ID'],axis=1)
df=df.drop(['Loan_ID'],axis=1)
df['Gender'].value_counts(dropna=False)
df2['Gender'].value_counts(dropna=False)

from sklearn_pandas import CategoricalImputer
imputer=CategoricalImputer()
df2['Gender']=imputer.fit_transform(df2['Gender'])
df['Gender']=imputer.fit_transform(df['Gender'])

df2['Dependents']=imputer.fit_transform(df2['Dependents'])
df['Dependents']=imputer.fit_transform(df['Dependents'])

df2['Self_Employed']=imputer.fit_transform(df2['Self_Employed'])
df['Self_Employed']=imputer.fit_transform(df['Self_Employed'])

df2['Credit_History']=imputer.fit_transform(df2['Credit_History'])
df['Credit_History']=imputer.fit_transform(df['Credit_History'])

df['Married']=imputer.fit_transform(df['Married'])

df2.isna().sum()
df.isna().sum()


df2['LoanAmount'].fillna(df2['LoanAmount'].median(),inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(),inplace=True)
df2['Loan_Amount_Term'].fillna(df2['Loan_Amount_Term'].median(),inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(),inplace=True)
df2.isna().sum()
df.isna().sum()


## Mapping of 'Target' variable
df2['Loan_Status'] = df2['outcome'].map( {'N': 0, 'Y': 1} ).astype(int)
df['Loan_Status'] = df['Loan_Status'].map( {'N': 0, 'Y': 1} ).astype(int)

df2=df2.drop(['outcome'],axis=1)
df2.info()
df.info()

##  Creating hot-code or dummies 
data_dum2 = pd.get_dummies(df2)
data_dum = pd.get_dummies(df)

data_dum2.info()
data_dum.info()


##  Split data : validate data as test data
yTrain=data_dum['Loan_Status'] # Target variable
xTrain=data_dum.drop(['Loan_Status'],axis=1)

yTest=data_dum2['Loan_Status'] # Target variable
xTest=data_dum2.drop(['Loan_Status'],axis=1)

##scale data for SVM
from sklearn.preprocessing import StandardScaler

#initialise the scaler
scaler=StandardScaler()
#to scale data
scaler.fit(xTrain)
xTrainS=pd.DataFrame(scaler.transform(xTrain))
xTestS=pd.DataFrame(scaler.transform(xTest))
xTrainS.columns=list2
xTestS.columns=list2

#Support vector machines ...ensemble model...faster
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

svc=OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear',probability=True)))
svc.fit(xTrainS,yTrain)
Y_pred_SVM=svc.predict(xTestS)
from sklearn.metrics import confusion_matrix
confusion_matrix(yTest,Y_pred_SVM)
#Accuracy:-(58+289)/367----94%
#ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
svm_roc_auc=roc_auc_score(yTest,svc.predict(xTest))
svm_roc_auc
fpr,tpr,thresholds=roc_curve(yTest,svc.predict_proba(xTest)[:,1])
# auc 48%



###Logistiuc regression '
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(xTrain,yTrain)

#prediction
Y_pred_LG=logmodel.predict(xTest)

#confusion matrix
from sklearn.metrics import confusion_matrix 
confusion_matrix(yTest,Y_pred_LG)
#Accuracy:-93%

#ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
lg_roc_auc=roc_auc_score(yTest,logmodel.predict(xTest))
lg_roc_auc
#auc is 86%% (area under the curve)
lgfpr,lgtpr,thresholds=roc_curve(yTest,logmodel.predict_proba(xTest)[:,1])


###Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest=RandomForestClassifier(n_estimators=100,oob_score=True,random_state=123)
random_forest.fit(xTrain,yTrain)
Y_prediction=random_forest.predict(xTest)
random_forest.score(xTrain,yTrain) 
confusion_matrix(yTest,Y_prediction)
#(64+274)/367---> 0.9209809264305178%

RF_roc_auc=roc_auc_score(yTest,random_forest.predict(xTest))
RF_roc_auc
#auc 88%
RF_fpr,RF_tpr,RF_thresholds=roc_curve(yTest,random_forest.predict_proba(xTest)[:,1])

#Decision tree
from sklearn.tree import DecisionTreeClassifier
decision_tree=DecisionTreeClassifier()
decision_tree.fit(xTrain,yTrain)
Y_pred_DT=decision_tree.predict(xTest)
confusion_matrix(yTest,Y_pred_DT)
#accuracy (64+230)/367--> 0.8010899182561307

#ROC Curve and auc
dt_roc_auc=roc_auc_score(yTest,decision_tree.predict(xTest))
DTprob=decision_tree.predict_proba(xTest)
DT_fpr,DT_tpr,DT_thresholds=roc_curve(yTest,decision_tree.predict_proba(xTest)[:,1])
dt_roc_auc


##KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)  #bydefault it takes 5
knn.fit(xTrain,yTrain)
Y_pred_knn=knn.predict(xTest)
confusion_matrix(yTest,Y_pred_knn)


##### KNN tuning 'k'
error = []
# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(xTrain, yTrain)
    pred_i = knn.predict(xTest)
    error.append(np.mean(pred_i != yTest))
    
###### KNN tuning 'k'
plt.figure()
plt.plot(error,marker='o')
plt.xticks(np.arange(0, 40, step=1))
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

from sklearn.neighbors import KNeighborsClassifier

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib qt

knn=KNeighborsClassifier(n_neighbors=36)  #using above curve selcted 22
knn.fit(xTrain,yTrain)
Y_pred_knn=knn.predict(xTest)
confusion_matrix(yTest,Y_pred_knn)


knn_roc_auc = roc_auc_score(yTest,knn.predict(xTest))
knnfpr, knntpr, knnthresholds = roc_curve(yTest,knn.predict_proba(xTest)[:,1])

##Over sampling-Synthetic Minority Over-sampling Technique (SMOTE) 
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.datasets import make_imbalance
from imblearn.under_sampling import NearMiss

smt = SMOTE()
xSTrain,ySTrain = smt.fit_sample(xTrain, yTrain)
np.bincount(ySTrain)
np.bincount(yTrain)

#logistuc model with yTrainS
from sklearn.linear_model import LogisticRegression
logmodelS=LogisticRegression()
logmodelS.fit(xSTrain,ySTrain)

#prediction
Y_pred_LGS=logmodelS.predict(xTest)

#confusion matrix
from sklearn.metrics import confusion_matrix 
confusion_matrix(yTest,Y_pred_LGS) 

Slg_roc_auc = roc_auc_score(yTest,logmodel.predict(xTest))
Sfpr, Stpr, Sthresholds = roc_curve(yTest,logmodel.predict_proba(xTest)[:,1])

# Under sampling
nr = NearMiss()
xNTrain, yNTrain = nr.fit_sample(xTrain, yTrain)
np.bincount(yTrain)
np.bincount(yNTrain)

logmodel.fit(xNTrain,yNTrain)
NY_pred_LG = logmodel.predict(xTest)
# Confusion matrix
confusion_matrix(yTest, NY_pred_LG)


# AUC and FPR, TPR & Threshholds
# ROC curve
Nlg_roc_auc = roc_auc_score(yTest,logmodel.predict(xTest))
Nfpr, Ntpr, Nthresholds = roc_curve(yTest,logmodel.predict_proba(xTest)[:,1])



#### ROC & AUC curve
plt.figure()
plt.plot(fpr,tpr,label='SVM (area = %0.2f)' % svm_roc_auc)
plt.plot(lgfpr,lgtpr,label='LG (area = %0.2f)' % lg_roc_auc)
plt.plot(RF_fpr,RF_tpr,label='RF (area = %0.2f)' % RF_roc_auc)
plt.plot(DT_fpr,DT_tpr,label='DT (area = %0.2f)' % dt_roc_auc)
plt.plot(knnfpr,knntpr,label='KNN (area = %0.2f)' % knn_roc_auc)
plt.plot(Sfpr,Stpr,color='black', label='SLG-Smote (area = %0.2f)' % Slg_roc_auc)
plt.plot(Nfpr,Ntpr,label='NLG-NearMiss (area = %0.2f)' % Nlg_roc_auc)
plt.plot([0, 1],[0, 1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC’, fontsize='xx-large')
plt.legend(loc='lower right',fontsize='xx-large')   
plt.show()


#Model evaluation table

#pip isntall pTable
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

x = PrettyTable()
x.field_names = ["Model","Accuracy","Precision","Recall","F1-score","AUC"]
print(x)

## Model Evaluation Table
x.add_row(["Logistic Regression",
           format(round(accuracy_score(yTest, Y_pred_LG),2),"0.2f"),
           format(round(precision_score(yTest, Y_pred_LG),2),"0.2f"),
           format(round(recall_score(yTest, Y_pred_LG),2),"0.2f"),
           format(round(f1_score(yTest, Y_pred_LG),2),"0.2f"),
           format(round(lg_roc_auc,2),"0.2f")])
print(x)


x.add_row(["Random-Forest",
           format(round(accuracy_score(yTest, Y_prediction),2),"0.2f"),
           format(round(precision_score(yTest, Y_prediction),2),"0.2f"),
           format(round(recall_score(yTest, Y_prediction),2),"0.2f"),
           format(round(f1_score(yTest, Y_prediction),2),"0.2f"),
           format(round(RF_roc_auc,2),"0.2f")])
print(x)

x.add_row(["SVM",
           format(round(accuracy_score(yTest, Y_pred_SVM),2),"0.2f"),
           format(round(precision_score(yTest, Y_pred_SVM),2),"0.2f"),
           format(round(recall_score(yTest, Y_pred_SVM),2),"0.2f"),
           format(round(f1_score(yTest, Y_pred_SVM),2),"0.2f"),
           format(round(svm_roc_auc,2),"0.2f")])
print(x)

x.add_row(["Decision Tree",
           format(round(accuracy_score(yTest, Y_pred_DT),2),"0.2f"),
           format(round(precision_score(yTest, Y_pred_DT),2),"0.2f"),
           format(round(recall_score(yTest, Y_pred_DT),2),"0.2f"),
           format(round(f1_score(yTest, Y_pred_DT),2),"0.2f"),
           format(round(dt_roc_auc,2),"0.2f")])
print(x)

x.add_row(["KNN",
           format(round(accuracy_score(yTest, Y_pred_knn),2),"0.2f"),
           format(round(precision_score(yTest, Y_pred_knn),2),"0.2f"),
           format(round(recall_score(yTest, Y_pred_knn),2),"0.2f"),
           format(round(f1_score(yTest, Y_pred_knn),2),"0.2f"),
           format(round(knn_roc_auc,2),"0.2f")])
print(x)

x.add_row(["SMOTE-LG",
           format(round(accuracy_score(yTest, Y_pred_LGS),2),"0.2f"),
           format(round(precision_score(yTest, Y_pred_LGS),2),"0.2f"),
           format(round(recall_score(yTest, Y_pred_LGS),2),"0.2f"),
           format(round(f1_score(yTest, Y_pred_LGS),2),"0.2f"),
           format(round(Slg_roc_auc,2),"0.2f")])
print(x)

x.add_row(["Under-sampling-LG",
           format(round(accuracy_score(yTest, NY_pred_LG),2),"0.2f"),
           format(round(precision_score(yTest, NY_pred_LG),2),"0.2f"),
           format(round(recall_score(yTest, NY_pred_LG),2),"0.2f"),
           format(round(f1_score(yTest, NY_pred_LG),2),"0.2f"),
           format(round(Nlg_roc_auc,2),"0.2f")])
print(x)


## Model Evaluation Table
## write table to file
with open('model_evaluationValidation.txt', 'w') as w:
    w.write(str(x))

##Random forest is the best model 
    
