import pandas as pd
#I am importin pandas

from sklearn.tree import DecisionTreeClassifier
# I am importing decision tree from scikit learn package


import joblib
#importing joblib to dump and load our models so that we dont need to create again and train our models. we can use existing pre-trained model

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

joblib.dump(model,'music-suggestion.joblib')
#creating a joblib file as already trained model and I can directly load this for my future projects and use this as AI
