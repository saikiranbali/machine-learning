import pandas as pd
#I am importin pandas

from sklearn.tree import DecisionTreeClassifier
# I am importing decision tree from scikit learn package

music = pd.read_csv('music.csv')
#imorting music csv

X = music.drop(columns='genre')
#Dropping genre column as that is a split for my input and output

Y = music['genre']
#adding Y column as ouput expecting what type of music we need to predict

model = DecisionTreeClassifier()
#adding a model and assigning our decision tree classifier

model.fit(X,Y)
#adding my input and output, where X is input and Y is expected output

pre = model.predict([[21,1],[22,0]])
#adding new columns ad 21 age and 1 as male
#adding new columns as 22 age and 0 as female to get some sample predictions
pre
