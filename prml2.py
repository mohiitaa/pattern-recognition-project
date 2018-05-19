##################################################################
#### Data Cleaning 
# Author - Mohita Chowdhury
##################################################################
# Import the libraries
from __future__ import division
import numpy as np    #For playing with matrices and vectors
import pandas as pd   #To use .csv files
import seaborn as sns  #For plots
import random

##################################################################

# Import the training and test data


train = pd.read_csv("/home/mohiitaa/Downloads/train.csv")             # Change the directory to your system path
test    = pd.read_csv("/home/mohiitaa/Downloads/test.csv")

a = train.head(5)
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

#####################################################################

n_examples = train.shape[0]
# print n_rows
n_features = train.shape[1]
n_features = n_features - 1
print n_features


####################################################################
# ##initialising weights

# # Choose an epsilon to initialise weight vector between [-eps, eps]
# eps = 0.0000

# w = random.sample(range(-10, 10), n_features + 1)

# weights = [x * eps for x in w]

# # Converting list to numpy array

# weights = np.asarray(weights)
# weights = weights.transpose()
# print weights

X_tr = X_train.values
Y_tr = Y_train.values

####################################################################
# Make a prediction with weights
def perceptron_sgd(X_tr, Y_tr):
    w = np.zeros(len(X_tr[0]))
    eta = 1
    epochs = 3

    for t in range(epochs):
        for i, x in enumerate(X_tr):
            if (np.dot(X_tr[i], w)*Y_tr[i]) <= 0:
                w = w + eta*X_tr[i]*Y_tr[i]

    return w

weights = perceptron_sgd(X_tr,Y_tr)
print(weights)

# detect = np.zeros(len(X_tr))
# for i in range(len(X_tr)):
#     detect[i] = np.dot(X_tr[i],weights);

# for i in range(len(detect)):
#     if detect[i] <= 0:
#         detect[i] = 0
#     else:
#         detect[i] = 1
# print detect


# print Y_tr
# detect = np.zeros(len(X_tr))
# for i in range(len(X_tr)):
#     detect[i] = np.dot(X_tr[i,:],weights);
# for i in range(len(detect)):
#     if detect[i] <= 0:
#         detect[i] = 0
#     else:
#         detect[i] = 1
# print detect

# err =Y_tr - detect

# count_correct = 0
# # count_y_survival = 0
# # count_detect = 0
# for i in range(len(err)):
#     if err[i] == 0:
#         count_correct =  count_correct + 1

# accuracy = (count_correct *100 )/len(Y_tr)  
# print accuracy

##################################################################
###Bayes Classifier with ML estimation
# count_y_survival = 0

# for i in range(len(Y_train)):
#     if Y_train[i] == 1:
#         count_y_survival =  count_y_survival + 1

# p1 = count_y_survival/len(Y_train);
# p0 = (len(Y_train) - count_y_survival)/len(Y_train)

# # print p1,p0

# mu0 = 0
# mu1 = 0

# sig0 = 0
# sig1 = 0

# for i in range(len(Y_train)):
#     if Y_train[i] == 0:
#         class_0_sum = class_0_sum + X_train[i]
#     else:
#         class_1_sum = class_1_sum + X_train[i]


# mu0 = class_0_sum/(len(Y_train)-count_y_survival)
# mu1 = class_1_sum/count_y_survival

# sig0 =         
