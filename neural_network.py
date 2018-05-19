######### NEURAL NETWORK IMPLEMENTATION #########################
# Authors - Mohita Chowdhury

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
n_examples = train.shape[0] #Finding number of rows of train DataFrame jusing pandas command
# print n_rows
n_features = train.shape[1] #Finding number of columns of train DataFrame jusing pandas command
n_features = n_features - 1 #To remove the serial number column
####################################################################
## Defining parameters
##  L  - #Layers
## n_L - #Nodes in a layer
##  s  - index to iterate over all training samples
##  j  - index to iterate over all nodes in a layer

L = 3
n_L = np.zeros(L)
n_L = [n_features, 3, 1]  #Defining the number of nodes in each layer
####################################################################
##initialising weights
# Random intialising of weights is considered a good practice. eps = 0.0001
# Here initialising to zero gives better accuracy

epsilon = 0.12
# Choose an epsilon to initialise weight vector between [-eps, eps]
def initialiseWeights(L_in, L_out,eps):
    W = np.random.rand(L_out,1+L_in)*2*eps - eps*np.ones((L_out,1+L_in))
    return W

weights = []
for l in range(L-1):
    weights.append(initialiseWeights(n_L[l],n_L[l+1],epsilon))


# weights_1 = initialiseWeights(n_L[0], n_L[1], epsilon)
# weights_2 = initialiseWeights(n_L[1], n_L[2], epsilon)
# weights_1 = initialiseWeights(n_L[0], n_L[1], epsilon)


# print weights
weights = np.asarray(weights)
# print weights[0][0,7]
# print weights[0][1]
# print weights[0][:,1]
# print weights[0]
# print weights.shape
# print weights_1

######################################################################
#Defining sigmoid function as the activation function
def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))
######################################################################
# for i in range(3):
#     print i
predict = [[]]
net = [[]]
# for s in range(len(X_tr)):
#     for l in range(L):
        # predict.append(initialiseWeights(L,n_L[l],0))
temp  = []
def initialiseOutputs(numberNodes):
    y = np.zeros(numberNodes)
    return y
##################################################################
###  FORWARD PROPAGATION #########################################
##################################################################
for s in range(len(X_tr)):

    y_nn = []
    n_ip = []
    for l in range(L):
        y_nn.append(initialiseOutputs(n_L[l])) 

    for l in range(L):
        n_ip.append(initialiseOutputs(n_L[l])) 
    # X0 = np.ones((n_examples,1))
    y_nn[0] = X_tr[s]
    n_ip[0] = X_tr[s]
    # y_nn[0] = np.asarray(y_nn[0])
    # print y_nn[0] 
    # print len(y_nn),len(y_nn[0])
    for l in range(L-1):
        # print "Computation for Layer ------------------", l
        y_nn[l] = np.hstack((1,y_nn[l]))
        # print y_nn[l]
        for j in range(n_L[l+1]):   #[7 3 1]
           for i in range(n_L[l]):
                # print weights[l][j]
                # print "Length of width vector",len(weights[l][j])
                # print "Length of y vector",len(y_nn[l])
                # print "ith node:",i
                # print "jth node:",j
                # print "lth Layers:",l
                n_ip[l+1][j] = np.dot(weights[l][j],y_nn[l])
                y_nn[l+1][j] = sigmoid(n_ip[l+1][j],derivative = False)
           # print "y_nn for node",j,"and layer",l,"is:",y_nn[l+1][j]
        # y_nn[l+1] = np.hstack((1,y_nn[l+1]))       
        # print "Y_nn of layer",l,"is:",y_nn[l+1]
        # predict[l][s] = y_nn[l]
    # print y_nn[0] 
    temp_n = []
    temp_i_n = []
    for l in range(L):
        temp_i_n = np.asarray(n_ip[l])
        temp_n.append(temp_i_n)
    # print temp    
    # temp = np.asarray(temp)
    net.append(temp_n)

    temp = []
    temp_i = []
    for l in range(L):
        temp_i = np.asarray(y_nn[l])
        temp.append(temp_i)
    # print temp    
    # temp = np.asarray(temp)
    predict.append(temp)
    # print y_nn[2]
# print predict[0] 
# print net[0]
# print X_tr[0]
# print predict[len(X_tr)]
# print net[len(X_tr)]
# print len(X_tr)
# print predict[2][1][3]
# print predict[2][1]
# print predict[2]
# count_correct_nn = 0
# err_nn =Y_train - predict
# for i in range(len(err_nn)):
#     if err_nn[i] == 0:
#         count_correct_nn =  count_correct_nn + 1

# accuracy_nn = (count_correct_nn *100 )/len(Y_train)

# print "Accuracy using NN Classifier: ",accuracy_nn,"%"
##################################################################
###  BACKPROPAGATION #############################################
##################################################################
delta = []
for l in range(L):
    delta.append(initialiseOutputs(n_L[l]))

for s in range(len(X_tr)):
    for l in range(L-1,-1,-1):
        # print l
        if l == L-1:
            delta[l][0] = (predict[s+1][l][0]-Y_tr[s])*sigmoid(net[s+1][l][0],derivative=True) 
            # print "Delta:",delta[L-1][0]
            # print "Actual",Y_tr[s]
            # print "Predicted",predict[s]
            # print "Derv of activation",sigmoid(net[s][L-1][0],derivative=True)

        else:
            for j in range(n_L[l]):
                delta[l][j] = (np.dot(delta[l+1],weights[l][:,j]))*sigmoid(net[s+1][l][0],derivative=True) 

### Printing node errors

# print delta[L-3]    
l_rate = 0.1

for s in range(len(X_tr)):
    for l in range(L-1):
        for i in range(n_L[l]):
            for j in range(n_L[l+1]):
                weights[l][j,i] = weights[l][j,i] - l_rate*delta[l+1][j]*predict[s+1][l][i]


for s in range(len(X_tr)):

    y_nn = []
    n_ip = []
    for l in range(L):
        y_nn.append(initialiseOutputs(n_L[l])) 

    for l in range(L):
        n_ip.append(initialiseOutputs(n_L[l])) 
    # X0 = np.ones((n_examples,1))
    y_nn[0] = X_tr[s]
    n_ip[0] = X_tr[s]
    # y_nn[0] = np.asarray(y_nn[0])
    # print y_nn[0] 
    # print len(y_nn),len(y_nn[0])
    for l in range(L-1):
        # print "Computation for Layer ------------------", l
        y_nn[l] = np.hstack((1,y_nn[l]))
        # print y_nn[l]
        for j in range(n_L[l+1]):   #[7 3 1]
           for i in range(n_L[l]):
                # print weights[l][j]
                # print "Length of width vector",len(weights[l][j])
                # print "Length of y vector",len(y_nn[l])
                # print "ith node:",i
                # print "jth node:",j
                # print "lth Layers:",l
                n_ip[l+1][j] = np.dot(weights[l][j],y_nn[l])
                y_nn[l+1][j] = sigmoid(n_ip[l+1][j],derivative = False)
           # print "y_nn for node",j,"and layer",l,"is:",y_nn[l+1][j]
        # y_nn[l+1] = np.hstack((1,y_nn[l+1]))       
        # print "Y_nn of layer",l,"is:",y_nn[l+1]
        # predict[l][s] = y_nn[l]
    # print y_nn[0] 
    temp_n = []
    temp_i_n = []
    for l in range(L):
        temp_i_n = np.asarray(n_ip[l])
        temp_n.append(temp_i_n)
    # print temp    
    # temp = np.asarray(temp)
    net.append(temp_n)

    temp = []
    temp_i = []
    for l in range(L):
        temp_i = np.asarray(y_nn[l])
        temp.append(temp_i)
    # print temp    
    # temp = np.asarray(temp)
    predict.append(temp)


# for i in range(len(X_tr)):
#     print predict[i+1][L-1]


y_out = np.zeros(len(X_tr))
# for i in range(len(X_tr)):
#     print predict[i+1][L-1]
sum = 0
for i in range(len(X_tr)):
    sum += predict[i+1][L-1]

mean_predict = sum/n_examples    
for i in range(len(X_tr)):
    if (predict[i+1][L-1] > mean_predict):
        y_out[i] = 1
    else:
        y_out[i] = 0

print y_out
count_correct_nn = 0
err_nn =Y_tr - y_out
for i in range(len(err_nn)):
    if err_nn[i] == 0:
        count_correct_nn =  count_correct_nn + 1

accuracy_nn = (count_correct_nn *100 )/len(Y_tr)

print "Accuracy using NN Classifier: ",accuracy_nn,"%"            