# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 17:28:18 2019

@author: 500035501
"""

#Credit risk analytics using svm
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier

#to display all columns and rows in console
pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns',50)
pd.set_option('display.width',1000)
pd.set_option('display.precision',2)


df=pd.read_csv('Python_Module_Day_15.2_Credit_Risk_Train_data.csv')
df.describe()
df.describe().transpose() 
df.info()
df.describe(include='O')

#Count missing values
df.isna().sum()

#Drop unwanted column
df=df.drop(['Loan_ID'],axis=1)

#impute 'catagorical varibles' ..impute  gender
df['Gender'].value_counts(dropna=False)  #gives na clunts for gender seperately

from sklearn_pandas import CategoricalImputer
imputer=CategoricalImputer()
df['Gender']=imputer.fit_transform(df['Gender'])

df['Married'].value_counts(dropna=False)
df['Married']=imputer.fit_transform(df['Married'])
df['Dependents'].value_counts(dropna=False)
df['Dependents']=imputer.fit_transform(df['Dependents'])
df['Self_Employed'].value_counts(dropna=False)
df['Self_Employed']=imputer.fit_transform(df['Self_Employed'])
df['Credit_History'].value_counts(dropna=False)
df['Credit_History']=imputer.fit_transform(df['Credit_History'])
df.isna().sum()

#only numeric data impute
#impute loamAmount
df['LoanAmount'].isna().sum()
df['LoanAmount'].describe()
df['LoanAmount'].plot(kind='hist')
#replace na with median as there is positive skewness
df['LoanAmount'].fillna(df['LoanAmount'].median(),inplace=True)
df['LoanAmount'].describe()
df['LoanAmount'].isna().sum()

#impute Loan_Amount_Term
df['Loan_Amount_Term'].isna().sum()
df['Loan_Amount_Term'].describe()
df['Loan_Amount_Term'].plot(kind='hist')
#replace na with median as there is negative skewness
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(),inplace=True)
df['Loan_Amount_Term'].describe()
df['Loan_Amount_Term'].isna().sum()

#EDA-data visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib qt       #plot on new window

sns.barplot(x='Loan_Status',y='LoanAmount',hue='Property_Area',data=df,ci=None)

#Histogram
sns.distplot(df['ApplicantIncome'])

#Boxplot
sns.boxplot(y='LoanAmount',x='Loan_Status',data=df)
sns.boxplot(y='LoanAmount',x='Loan_Status',hue='Gender',data=df)

sns.boxplot(x='Married',y='ApplicantIncome',hue='Loan_Status',data=df)
sns.boxplot(data=df,orient="h")

#use swarmplolt() to show the dataponits on top of the boxes
sns.swarmplot(x='LoanAmount',y='Loan_Status',hue='Gender',data=df)

#scatter plot with regression line
#regression line and a 95% confidence internal
sns.regplot(y='LoanAmount',x='ApplicantIncome',data=df)
#seaborn implot is a 2d scatterplot with an optional regression line and hue
sns.lmplot(x='LoanAmount',y='ApplicantIncome',data=df,hue='Gender')
#seaborn implot is a 2d scatterplot with an optional regression line and hue and column
sns.lmplot(x='LoanAmount',y='ApplicantIncome',col='Married',data=df,hue='Gender')

#fit and plot a univariate and bivariate
#piechart
p1=df['Gender'].value_counts()
p1
plt.pie(p1,labels=p1.index,autopct='%2.1f%%')
plt.title('Gender proportion of cases')

p1=df['Property_Area'].value_counts()
p1
plt.pie(p1,labels=p1.index,autopct='%2.1f%%')
plt.title('Property_Area proportion of cases')


p1=df['Loan_Status'].value_counts()
p1
plt.pie(p1,labels=p1.index,autopct='%2.1f%%')
plt.title('loan status proportion of cases')

#Cross table and pivot table
pd.crosstab(df.Gender,df.Loan_Status)

#values:mean by default
df.pivot_table('ApplicantIncome',index=['Gender','Dependents'],columns='Property_Area')

#cross table nad pivot table
t1=pd.pivot_table(df,values=['LoanAmount'],
                  index=['Property_Area'],
                  aggfunc={'LoanAmount':[min,max,np.mean,np.std]})

t1

#mapping of target variable
df['Loan_Status']=df['Loan_Status'].map({'N':0,'Y':1}).astype(int)

#creating hot code or dummies
data_dum=pd.get_dummies(df)
data_dum.info()

#split data into tets and train
from sklearn.model_selection import train_test_split
y=data_dum['Loan_Status']  #target variable
x=data_dum.drop('Loan_Status',axis=1)  #independent variable

#split data into test and train
xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.2,random_state=123)

list2=xTest.columns

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
#Accuracy:-(17+79)/123
# 0.7804878048780488 

#ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
svm_roc_auc=roc_auc_score(yTest,svc.predict(xTest))
svm_roc_auc
fpr,tpr,thresholds=roc_curve(yTest,svc.predict_proba(xTest)[:,1])
# auc 50%

#plot ROC and AUC
plt.figure()
plt.plot(fpr,tpr,label='Logistic Regression(area=%0.2f)'%lg_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC')
plt.legend(loc="lower right")
plt.show()

###Logistiuc regression '

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(xTrain,yTrain)

#prediction
Y_pred_LG=logmodel.predict(xTest)

#confusion matrix
from sklearn.metrics import confusion_matrix 
confusion_matrix(yTest,Y_pred_LG)
#Accuracy:-78%

#ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
lg_roc_auc=roc_auc_score(yTest,logmodel.predict(xTest))
lg_roc_auc
#auc is 69%% (area under the curve)
lgfpr,lgtpr,thresholds=roc_curve(yTest,logmodel.predict_proba(xTest)[:,1])

###Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest=RandomForestClassifier(n_estimators=100,oob_score=True,random_state=123)
random_forest.fit(xTrain,yTrain)
Y_prediction=random_forest.predict(xTest)
random_forest.score(xTrain,yTrain) #gives oob score i,.e 97 %
confusion_matrix(yTest,Y_prediction)
#(19+77)/123 i.e 78%

RF_roc_auc=roc_auc_score(yTest,random_forest.predict(xTest))
RF_roc_auc
#auc 70%
RF_fpr,RF_tpr,RF_thresholds=roc_curve(yTest,random_forest.predict_proba(xTest)[:,1])


#Decision tree
from sklearn.tree import DecisionTreeClassifier
decision_tree=DecisionTreeClassifier()
decision_tree.fit(xTrain,yTrain)
Y_pred_DT=decision_tree.predict(xTest)
confusion_matrix(yTest,Y_pred_DT)
#accuracy (25+66)/123 -->73%

#ROC Curve and auc
dt_roc_auc=roc_auc_score(yTest,decision_tree.predict(xTest))
DTprob=decision_tree.predict_proba(xTest)
DT_fpr,DT_tpr,DT_thresholds=roc_curve(yTest,decision_tree.predict_proba(xTest)[:,1])
dt_roc_auc
#auc 70%

##ROC and AUC curve

plt.figure()
plt.plot(fpr,tpr,lable='SVM(area=%0.2f)'%svm_roc_auc)
plt.plot(lgfpr,lgtpr,lable='LG(area=%0.2f)'%lg_roc_auc)
plt.plot(RF_fpr,RF_tpr,lable='RF(area=%0.2f)'%RF_roc_auc)
plt.plot(DT_fpr,DT_tpr,lable='DT(area=%0.2f)'%dt_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.legend(loc="lower right",fontsize="xx-large")
plt.show()


##KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)  #bydefault it takes 5
knn.fit(xTrain,yTrain)
Y_pred_knn=knn.predict(xTest)
confusion_matrix(yTest,Y_pred_knn)
## Accuracy--(7+64)/123-->57%

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
knn=KNeighborsClassifier(n_neighbors=22)  #using above curve selcted 22
knn.fit(xTrain,yTrain)
Y_pred_knn=knn.predict(xTest)
confusion_matrix(yTest,Y_pred_knn)
#(3+73)/123
#Out[224]: 0.6178861788617886

knn_roc_auc = roc_auc_score(yTest,knn.predict(xTest))
knnfpr, knntpr, knnthresholds = roc_curve(yTest,knn.predict_proba(xTest)[:,1])
##AUC 49%

##Over sampling-Synthetic Minority Over-sampling Technique (SMOTE) 
#when target variable is imbalanced (y=69% vs n=31%)
p1=df['Loan_Status'].value_counts()
p1
plt.pie(p1,labels=p1.index,autopct='%2.1f%%')
plt.title('loan status proportion of cases')

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
confusion_matrix(yTest,Y_pred_LGS) #accuracy-81%

Slg_roc_auc = roc_auc_score(yTest,logmodel.predict(xTest))
Sfpr, Stpr, Sthresholds = roc_curve(yTest,logmodel.predict_proba(xTest)[:,1])
#auc 69%

# Under sampling
nr = NearMiss()
xNTrain, yNTrain = nr.fit_sample(xTrain, yTrain)
np.bincount(yTrain)
np.bincount(yNTrain)

logmodel.fit(xNTrain,yNTrain)
NY_pred_LG = logmodel.predict(xTest)
# Confusion matrix
confusion_matrix(yTest, NY_pred_LG)
##acuuracy--->(29+49)/123  --0.6341463414634146

# AUC and FPR, TPR & Threshholds
# ROC curve
Nlg_roc_auc = roc_auc_score(yTest,logmodel.predict(xTest))
Nfpr, Ntpr, Nthresholds = roc_curve(yTest,logmodel.predict_proba(xTest)[:,1])
#AUC-64%


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
with open('model_evaluation.txt', 'w') as w:
    w.write(str(x))

##SMOTE LG is the best model 
    
