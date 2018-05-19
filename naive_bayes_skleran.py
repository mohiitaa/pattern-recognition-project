# Import the libraries
from __future__ import division
import numpy as np    #For playing with matrices and vectors
import pandas as pd   #To use .csv files
import seaborn as sns  #For plots
import random
import scipy as sp
from numpy.linalg import pinv

##################################################################

# Import the training and test data


train = pd.read_csv("/home/mohiitaa/Downloads/train.csv")             ### Change the directory to your system path
test    = pd.read_csv("/home/mohiitaa/Downloads/test.csv")

a = train.head(5)  #Prints the first 5 datasamples. Note "head"
# print a
##################################################################

# Delete the features which are not required

train= train.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
test = test.drop(['Name'], axis=1)

full = [train, test]
b = train.head()
# print b

###################################################################

# Map gender from string to a numeric value 

for dataset in full:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

####################################################################

#Filling all N/A values with 0

freq_port = train.Embarked.dropna().mode()[0]

for dataset in full:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
# Transforming a categorical feature into numeric one    
for dataset in full:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#####################################################################

# Modify the age 

guess_ages = np.zeros((2,3))
#Dividing age into bands using PClass and Sex
for dataset in full:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)


#####################################################################

# Group age categories

train['AgeBand'] = pd.cut(train['Age'], 5)
train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
# Dividing age into bands 
for dataset in full:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train.head()


#####################################################################    

# Drop Ageband

train= train.drop(['AgeBand'], axis=1)
full = [train, test]
b = train.head()
# print b

#####################################################################

train['FareBand'] = pd.qcut(train['Fare'], 4)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

#####################################################################

test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
# test.head()

for dataset in full:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train = train.drop(['FareBand'], axis=1)
full = [train, test]
    
# train.head()
#####################################################################
# Create training and test data

X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop(["PassengerId", "Ticket", "Cabin"], axis=1).copy()

#####################################################################

#Look at the first few data samples from the training data
xtrain = X_train.head()
print xtrain
X_tr = X_train.values
Y_tr = Y_train.values
####################################################################

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_tr, Y_tr).predict(X_train)
print "Prediction " + str(gnb.predict(X_tr))
print "Actual     " + str(Y_tr)
print "Accuracy   " + str(gnb.score(X_tr, Y_tr)*100) + "%"
