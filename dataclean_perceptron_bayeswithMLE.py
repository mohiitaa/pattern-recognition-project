##################################################################
#### Data Cleaning, Perceptron Learning and Gaussian MLE
# Author - Mohita Chowdhury
##################################################################
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
test  = pd.read_csv("/home/mohiitaa/Downloads/test.csv")

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
print "The dataset looks like:-------------------"
print xtrain

#####################################################################

#### PERCEPTRON LEARNING IMPLEMENTATION #############################




#####################################################################
n_examples = train.shape[0] #Finding number of rows of train DataFrame jusing pandas command
# print n_rows
n_features = train.shape[1] #Finding number of columns of train DataFrame jusing pandas command
n_features = n_features - 1 #To remove the serial number column
# print n_features


####################################################################
##initialising weights
# Random intialising of weights is considered a good practice. eps = 0.0001
# Here initialising to zero gives better accuracy

# Choose an epsilon to initialise weight vector between [-eps, eps]


eps = 0.0000    #Initialising weights to 0 for now

w = random.sample(range(-10, 10), n_features)

weights = [x * eps for x in w]

# Converting list to numpy array

weights = np.asarray(weights)
weights = weights.transpose()
# print weights

####################################################################
epochs = 3  #Number of passes to be made

alpha = 1   #Learning rate | kept to "1" for perfect percetron

#Converting dataframe into numpy array
X_tr = X_train.values
Y_tr = Y_train.values

prediction = np.zeros(len(X_tr[0]))  #Array that stores classes predicted by percetron


for t in range(epochs):
    for i in range(len(X_tr[0])):

    #     if (np.dot(X_tr[i], weights)) <= 0:
    #         weights = weights + alpha * X_tr[i];
    #     else:
    #         weights = weights - alpha * X_tr[i];
        prediction[i] = np.dot(X_tr[i], weights);
        if prediction[i] <= 0:
            prediction[i] = 0 
        else:
            prediction[i] = 1

        if prediction[i] == Y_tr[i]:
            weights[i] = weights[i] + 0
        else:
            if prediction[i] - Y_tr[i] <= 0:
                weights = weights + alpha * X_tr[i];
            else:
                weights = weights - alpha * X_tr[i];
    t = t+1            



print "The learned weights are:",weights
detect = np.zeros(len(X_tr))  #Detect stores the sign(prediction)
for i in range(len(X_tr)):
    detect[i] = np.dot(X_tr[i,:],weights);

for i in range(len(detect)):
    if detect[i] <= 0:
        detect[i] = 0
    else:
        detect[i] = 1

# print detect
# print Y_train
Y_train = Y_train.values

####################################################################
## Calculating Accuracy
err =Y_train - detect

count_correct = 0
for i in range(len(err)):
    if err[i] == 0:
        count_correct =  count_correct + 1

accuracy = (count_correct *100 )/len(Y_train)     

print "Accuracy of Perceptron Learning on the given dataset is :",accuracy, "%"


##################################################################

##################################################################

#Bayes Classifier with ML estimation

###################################################################
## Calculating Priors

count_y_survival = 0

for i in range(len(Y_train)):
    if Y_tr[i] == 1:
        count_y_survival =  count_y_survival + 1

p1 = count_y_survival/len(Y_train);
p0 = (len(Y_train) - count_y_survival)/len(Y_train)

# print p1,p0
###################################################################
# Calculating Means for Classes 0 and 1

mu0 = np.zeros(len(X_tr[0]))
mu1 = np.zeros(len(X_tr[0]))

X_tr_0 = []
X_tr_1 = []
j = 0
for i in range(len(X_tr)):
    if Y_tr[i] == 0:
        X_tr_0.append(X_tr[i,:])
    else:
        X_tr_1.append(X_tr[i,:])
 
#Converting list to numpy array        
X_tr_0 = np.array(X_tr_0)

X_tr_1 = np.array(X_tr_1)

# print X_tr_0
# print X_tr_1


for i in range(len(X_tr_0[0])):
    mu0[i] = np.mean(X_tr_0[:,i]);
    # sig0[i] = np.std(X_tr_0[:,i]);

for i in range(len(X_tr_1[0])):
    mu1[i] = np.mean(X_tr_1[:,i]);
    # sig1[i] = np.std(X_tr_1[:,i]);


# print "Mean of class 0 :",mu0
# print "Mean of class 1 :",mu1 

#############################################################
# Calculating Standard Deviation or Covariance Matrix

sig0 = np.zeros((len(X_tr_0),len(X_tr_0)))
sig1 = np.zeros((len(X_tr_1),len(X_tr_1)))

# print len(X_tr_0)
# print len(X_tr_1)

sig0 = np.cov(X_tr_0.transpose())
sig1 = np.cov(X_tr_1.transpose())

# print "Standard Deviation of Class 0:",sig0
# print "Standard Deviation of Class 1:",sig1

# print len(sig0)
# print len(sig1)

########################################################################

## Bayes Classifier Implementation

pred = np.zeros(len(Y_tr))

for i in range(len(X_tr)):
    b = np.matmul((X_tr[i] - mu0),pinv(sig0))
    a = (X_tr[i] - mu0).transpose()
    q0 = p0 * (np.linalg.det(sig0)**(-0.5))*np.exp((-0.5)*np.matmul(b,a))
    # print "Posterior probability of class 0",q0

    b1 = np.matmul((X_tr[i] - mu1),pinv(sig1))
    a1 = (X_tr[i] - mu1).transpose()
    q1 = p1 * (np.linalg.det(sig1)**(-0.5))*np.exp((-0.5)*np.matmul(b1,a1))
    # print "Posterior probability of class 1",q1

    if q0/q1 < 1:
        pred[i] = 1
    else:
        pred[i] = 0

########################################################################
## Calculating Accuracy
count_correct_bayes = 0
err_bayes =Y_train - pred
for i in range(len(err_bayes)):
    if err_bayes[i] == 0:
        count_correct_bayes =  count_correct_bayes + 1

accuracy_bayes = (count_correct_bayes *100 )/len(Y_train)

print "Accuracy using Bayes Classifier and Gaussian MLE: ",accuracy_bayes,"%"