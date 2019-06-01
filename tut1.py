from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# [height, weight, shoe size]
X =[[181,80,44], [177,70,43], [160,60,38], [154, 54, 37], [166,65,40], 
    [190,90,47], [175,64,39], [177,70,40], [159,55,37], [171,75,42],[181,85,43] ]

Y = ['male', 'female', 'female', 'female', 'male', 'male', 
      'male', 'female', 'male', 'female', 'male']


#classifiers
clf_tree = tree.DecisionTreeClassifier()
clf_svc = svm.SVC()
clf_KNN = KNeighborsClassifier()
clf_NB = GaussianNB()


#training the models
clf_tree = clf_tree.fit(X,Y)
clf_svc = clf_svc.fit(X,Y)
clf_KNN = clf_KNN.fit(X,Y)
clf_NB = clf_NB.fit(X,Y)


prediction_tree = clf_tree.predict (X)
prediction_svc = clf_svc.predict (X)
prediction_KNN = clf_KNN.predict(X)
prediction_NB = clf_NB.predict (X)

result= accuracy_score(Y, prediction_tree)
result1 = accuracy_score (Y, prediction_svc)
result2 = accuracy_score(Y, prediction_KNN)
result3 = accuracy_score (Y,prediction_NB)

print (result)
print (result1)
print (result2)
print (result3)




