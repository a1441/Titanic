# Titanic
Code for the GDS video Titanic
### 1. Importing bla bla bla libraries

import pandas as pd
import numpy as np
from math import sqrt
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

### 2. Data Prep

#Importing the train/test csv and concatenating them

train = pd.read_csv('train.csv') #1-891
test = pd.read_csv('test.csv') #892 - 1309

data = pd.concat([train,test])
test_id= test['PassengerId'] #saving IDs for the output file

### 2.1 Data exploration

data.info() #we have missing values that need to be filled/cleaned

data

data.describe()

### 2.2. Exploring the unique values per category

data.astype('object').describe(include='all').loc['unique', :]

### 2.3. Exploring missing values per category

data.isnull().sum()

### 3. Data cleaning

#Clearning missing values
data['Age'] = data['Age'].fillna(data['Age'].median()) #fill median for Age
data['Fare'] = data['Fare'].fillna(data['Fare'].median()) #same for Fare
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0]) #mode for categorical data

data

data.isnull().sum()

### 4. Feature engineering

data['FamilySize'] = 1 + data['SibSp'] + data['Parch'] #1 for the person in the data row and adding his sublings/parents
data['OnboardAlone'] = data['FamilySize'].apply(lambda x: 0 if x > 1 else 1) #dummy for home alone

for name in data:
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.') 
    
    #regex FTW srsly + actually extracting data from this Feature

data #seems to work okay

### 4.1. Binning numerical data

sns.distplot(data['Fare'])

data['FareBin'] = (pd.qcut(data['Fare'], 5)).astype(str) #fuck ranges srsly
data['FareBin'].unique()

data['FareBin']=data['FareBin'].apply(lambda x: {'(-0.001, 7.854]': 1, '(7.854, 10.5]': 2,
                                                '(10.5, 21.558]': 3, '(21.558, 41.579]': 4,
                                                 '(41.579, 512.329]':5
                                                }.get(x,x))

sns.distplot(data['Age'])

data['AgeBin'] = (pd.qcut(data['Age'], 4)).astype(str)
data['AgeBin'].unique()

data['AgeBin']=data['AgeBin'].apply(lambda x: {'(0.169, 22.0]': 1, '(22.0, 28.0]': 2,
                                                '(28.0, 35.0]': 3, '(35.0, 80.0]': 4
                                                }.get(x,x))

data

data['Title'].value_counts() #do we need so many titles?

sns.countplot(x='Title',hue='Survived',data=data) #not rly

data['Title']=data['Title'].apply(lambda x: {'Mr': 1, 'Miss': 2,'Mrs': 3, 'Master': 4, 'Rev':5,
                                               'Dr':5, 'Col' :5, 'Mlle' :5, 'Ms' : 5, 'Major':5,
                                               'Dona':5, 'Lady':5, 'Sir':5, 'Jonkheer' :5, 'Mme':5,
                                               'Countess':5, 'Capt':5, 'Don':5 }.get(x,x))
data #bin them together to 4 bins

sns.countplot(x='Title',hue='Survived',data=data) 

data['Title']=data['Title'].apply(lambda x: {5 : 4}.get(x,x)) 
#5 doesn't seem to have much value in its own - off to the bin

sns.countplot(x='Title',hue='Survived',data=data)

data['Sex']=data['Sex'].apply(lambda x: {'male': 0, 'female': 1 }.get(x,x)) 
#no need to dummy it down, 0 is male, 1 is female

### 4.2. Dropping data that's processed/not useful

columns2drop = ['PassengerId', 'Cabin', 'Ticket','Name','Age','Fare']
data.drop(columns2drop, axis=1, inplace=True)

### 4.3. Converting categorical data to dummies 

data = pd.get_dummies(data)

data #looks good mofo

### 5.0. Preparing for the model

#respliting processed data
train = data[:891] #1-891 
test = data[891:]  #892 - 1309

train

test

### 5.1 Understanding what the model does

#Shout out to DataCamp
#https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python?

5.1.1 Splitting Data - stealing shit from DataCamp

 "To understand model performance, dividing the dataset into a training set and a test set is a good strategy.

 Let's split dataset by using function train_test_split(). You need to pass 3 parameters features, target, and test_set   size. Additionally, you can use random_state to select records randomly." 

Y = train['Survived'] #train
Y_test_kaggle= test['Survived'] #test
X = train.drop(columns=['Survived']) #train
X_test_kaggle= test.drop(columns=['Survived']) #test



### 6.0. Model me like one of your french girls

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

Here, the Dataset is broken into two parts in a ratio of 75:25. It means 75% data will be used for model training and 25% for model testing.



### 6.1. I've done a lot of nasty stuff with models ;)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

What is a Logistic Regression? Beats me, let's steal some explanations:

"It is a special case of linear regression where the target variable is categorical in nature. It uses a log of odds as the dependent variable. Logistic Regression predicts the probability of occurrence of a binary event utilizing a logit function. - Estimation is done through maximum likelihood."

![image.png](attachment:image.png)

The MLE is a "likelihood" maximization method.

Maximizing the likelihood function determines the parameters that are most likely to produce the observed data. 

From a statistical point of view, MLE sets the mean and variance as parameters in determining the specific parametric values for a given model. 

This set of parameters can be used for predicting the data needed in a normal distribution.

### 6.2. The actuall modeling part

model.fit(X_train, y_train) #model me daddy

y_pred=model.predict(X_test) #predicting the y's that we actually know

y_pred

### 6.3. Evaluating the model

"Model Evaluation using Confusion Matrix

A confusion matrix is a table that is used to evaluate the performance of a classification model. You can also visualize the performance of an algorithm. The fundamental of a confusion matrix is the number of correct and incorrect predictions are summed up class-wise."

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix #le confusing matrix

The dimension of this matrix is 2*2 because this model is binary classification. 

You have two classes 0 and 1. 

Diagonal values represent accurate predictions, while non-diagonal elements are inaccurate predictions. In the output, 111 and 28 are actual predictions, and 28 and 20 are incorrect predictions.

### 6.4. Let's make it sexy

Visualizing Confusion Matrix using Heatmap
Let's visualize the results of the model in the form of a confusion matrix using matplotlib and seaborn.

Here, you will visualize the confusion matrix using Heatmap.

# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#shit I've stolen from smarter than me people

class_names=[0,1] # classifying correct/incorect

fig, ax = plt.subplots() # fitting subplots together

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# creating the actual heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g') #importing the array into the plot
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

The confusion matrix ain't that bad.

Let's evaluate the model using model evaluation metrics such as accuracy, precision, and recall.

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

Well, we got a classification rate of 78%, considered as shitty accuracy.

Precision is about being precise, i.e., how accurate your model is. In other words, you can say, when a model makes a prediction, how often it is correct. In my prediction case, when the Logistic Regression model predicted a person is going to survive from the crash, then that person survives 76.1% of the time.

Recall: If there are people who have survived in the test set then the Logistic Regression model can identify them 76% of the time.

![image.png](attachment:image.png)

### 6.5. More fancy evaluations

ROC Curve
Receiver Operating Characteristic(ROC) curve is a plot of the true positive rate against the false positive rate. It shows the tradeoff between sensitivity and specificity.

y_pred_proba = model.predict_proba(X_test)[::,1] #le stealing shit again
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

AUC score for the case is 0.86. AUC score 1 represents perfect classifier, and 0.5 represents a worthless classifier.

### 7.0. Submitting the model

y_pred=model.predict(X_test_kaggle) #predicting unknown data

Submission = pd.DataFrame(y_pred) #DF from predicted values
Submission.columns = ['Survived'] #renaming the column name
Submission['Survived'] = Submission['Survived'].astype(int) #converting from float to int
 
test_id


my_submission = pd.concat([test_id, Submission ], axis=1) #concat the ID & predictions

my_submission

my_submission.to_csv('MakeMillionsFromYoutube6.csv', index=False) #let's check what we did

