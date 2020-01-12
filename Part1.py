#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import warnings
warnings.filterwarnings('ignore')


# In[48]:

#Tout d'abord on a fait une pré-étude et on a trouvé que les classes données dans chaque dataset, sont bien répartis,
#C'est à dire 50% 50% si on a 2 classes ou presque
#Vu que le nombre d'observation est petits, on a décidé donc de faire la validation des scores avec k-fold validation,
# et cela afin de profiter d'un maximum de notre dataset complet


# In[3]:


import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[4]:


aggregation = pd.read_table("aggregation.txt",header = None) ##  7 classes
flame = pd.read_table("flame.txt",header = None)  ## 2 classes
spiral = pd.read_table("spiral.txt", header = None)  ## 3 classes




# In[4]:
pd.set_option('expand_frame_repr', False)
aggregation.shape
flame.shape



# In[5]:


X_aggr = aggregation.iloc[:,0:2].values
Y_aggr = aggregation.iloc[:,2].values

X_flame = flame.iloc[:,0:2].values 
Y_flame = flame.iloc[:,2].values

X_spiral = spiral.iloc[:,0:2].values
Y_spiral = spiral.iloc[:,2].values


# In[4]:
sc1 = StandardScaler()
sc2 = StandardScaler()
sc3 = StandardScaler()

X_aggr = sc1.fit_transform(X_aggr)
X_flame = sc2.fit_transform(X_flame)
X_spiral = sc3.fit_transform(X_spiral)


# In[]


"""
X_aggr, X_aggr_test, Y_aggr, Y_aggr_test = train_test_split(X_aggr, Y_aggr, test_size = 0.2, random_state = 0)
X_flame, X_flame_test, Y_flame, Y_flame_test = train_test_split(X_flame, Y_flame, test_size = 0.2, random_state = 0)
X_spiral, X_spiral_test, Y_spiral, Y_spiral_test = train_test_split(X_spiral, Y_spiral, test_size = 0.2, random_state = 0)
"""



# In[100]:


# Aggregation
num_folds = 80 ##  
seed = 7
scoring = 'accuracy'
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(3)))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', SVC()))
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds)
    cv_results = cross_val_score(model, X_aggr, Y_aggr, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: Accuracy : %f écart-type de : %f" % (name, cv_results.mean(),cv_results.std())
    print(msg)
classifier = RandomForestClassifier()
classifier.fit(X_aggr, Y_aggr)

# In[89]:


#spiral 
num_folds = 110
seed = 7
scoring = 'accuracy'
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(3)))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier(10)))
models.append(('SVM', SVC()))
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_spiral, Y_spiral, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: Accuracy : %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
classifier = KNeighborsClassifier(3)
classifier.fit(X_spiral, Y_spiral)


# In[76]:


#flame
num_folds = 180 # in this case knn wins because he is the fastest and the bester with n_neigbors = 1
seed = 7
scoring = 'accuracy'
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(1)))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_flame, Y_flame, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: Accuracy : %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
classifier = KNeighborsClassifier(1)
classifier.fit(X_flame, Y_flame)

# In[86]:

# Visualising the flame data set result's

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

from matplotlib.colors import ListedColormap
X_set, y_set = X_flame, Y_flame
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.65, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier KNN with neighbour = 1 (Flame data set)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# In[112]:


# Visualising the aggregation data set result's

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

from matplotlib.colors import ListedColormap
X_set, y_set = X_aggr, Y_aggr
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.50, cmap = ListedColormap(('red', 'green','blue','black','maroon','purple','olive')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue','black','maroon','purple','olive'))(i), label = j)
plt.title('Classifier Random Forest (Aggregation Data Set)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# In[98]:


# Visualising the spiral data set result's

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

from matplotlib.colors import ListedColormap
X_set, y_set = X_spiral, Y_spiral
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.50, cmap = ListedColormap(('red', 'green','blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Classifier KNN = 3 ( Spiral Data Set )')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()









# In[98]:







