import pandas as pd
#I am importin pandas

from sklearn.tree import DecisionTreeClassifier
# I am importing decision tree from scikit learn package

from sklearn.model_selection import train_test_split
# I am importing test split for spliting my data 80% for training and 20% for testing

from sklearn.metrics import accuracy_score
#I import this for my accuracy score generating for my training data

music = pd.read_csv('music.csv')
#imorting music csv

X = music.drop(columns='genre')
#Dropping genre column as that is a split for my input and output

Y = music['genre']
#adding Y column as ouput expecting what type of music we need to predict

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
# here 0.2 means 20% I am adding for testing

model = DecisionTreeClassifier()
#adding a model and assigning our decision tree classifier

model.fit(X_train,Y_train)
#adding my input and output, where X_train is input and Y_train are main 80% training data is expected output

pre = model.predict(X_test)
#added the 20% test input for prediction

score = accuracy_score(Y_test, pre)
#gives our accuracy score by taking X_test as input from pre varaible and Y test as ouptut for our score

score
#To generate  the accurate score

