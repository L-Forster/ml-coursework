import joblib
import numpy as np
import sklearn.datasets
import os
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA, SparsePCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, \
    BaggingClassifier, StackingClassifier
import pandas as pd


os.environ["LOKY_MAX_CPU_COUNT"] = "-1"
seed = 1
np.random.seed(seed)
covtype = fetch_covtype()
X = np.array(covtype['data'])
y = np.array(covtype['target'])
scalar = StandardScaler()
X = scalar.fit_transform(X)



def split(X, y):
    return sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=seed)


def train_logistic_regression(data, y):
    log = LogisticRegression(random_state=seed, max_iter=300)
    cv_scores = cross_val_score(log, data, y, cv=5, scoring='accuracy')
    print(cv_scores)
    return log.fit(data, y)



def optimise_decision_tree(data, y):
    param_space = {
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 20),
        'max_features': Categorical([None, 'sqrt', 'log2'] + list(range(1, data.shape[1] + 1))),
        'ccp_alpha': Categorical([0.0, 0.1, 0.01]),  # pruning
        'max_depth': Categorical([None] + list(range(1, 20))),
        'splitter': Categorical(['best'])
    }
    model = DecisionTreeClassifier(random_state=seed)
    bayes_search = BayesSearchCV(estimator=model, search_spaces=param_space, cv=5, random_state=seed)
    bayes_search.fit(data, y)

    print("Best parameters:", bayes_search.best_params_)

    return bayes_search.best_params_


# #

def train_decision_tree(data, y):
    model = DecisionTreeClassifier(random_state=seed)
    cv_scores = cross_val_score(model, data, y, cv=5, scoring='accuracy')
    print(cv_scores)
    print(model.get_params())
    return model.fit(data, y)


def optimise_random_forest(data, y):
    param_space = {
        'n_estimators': Integer(75, 125),
        'max_depth': Categorical([None] + list(range(1, 10))),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Categorical([None, 'sqrt', 'log2']),
        'criterion': Categorical(['gini', 'entropy']),
        'bootstrap': Categorical([True, False])
    }

    model = RandomForestClassifier(random_state=seed)
    bayes_search = BayesSearchCV(estimator=model, search_spaces=param_space, cv=3, random_state=seed, n_jobs=-1, n_iter=5)
    print("3")
    bayes_search.fit(data, y)

    print("Best parameters:", bayes_search.best_params_)

    return bayes_search.best_params_


def train_random_forest(data, y):
    model = RandomForestClassifier(n_estimators=10, random_state=seed)
    scores = cross_val_score(model, data, y, cv=5, scoring='accuracy')
    model.fit(data, y)
    print(np.mean(scores))
    return model.fit(data, y)


def train_adaboost(data, y):
    model = AdaBoostClassifier(n_estimators=10, random_state=seed)
    scores = cross_val_score(model, data, y, cv=5, scoring='accuracy')
    print(np.mean(scores))
    return model.fit(data, y)


def train_grad_boost(data, y):
    model = GradientBoostingClassifier(n_estimators=10, random_state=seed)
    scores = cross_val_score(model, data, y, cv=5, scoring='accuracy')
    print(np.mean(scores))
    return model.fit(data, y)


def train_bagging(data, y):
    model = BaggingClassifier(random_state=seed)
    scores = cross_val_score(model, data, y, cv=5, scoring='accuracy')
    print(np.mean(scores))
    return model.fit(data, y)





def train_stacking(data, y, tree_params, forest_params):
    estimators = [
        ('bagging', BaggingClassifier(random_state=seed, n_jobs=-1)),
        ('dt', DecisionTreeClassifier(random_state=seed, **tree_params)),
        ('rf', RandomForestClassifier(random_state=seed, n_jobs=-1,**forest_params))
    ]

    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=seed, n_jobs=-1),
        n_jobs=-1
    )
    print("Splitting")
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=seed)
    print("fitting")
    model.fit(X_train, y_train)
    print("Predicting")
    y_pred = model.predict(X_val)
    print("Computing Accuracy")
    accuracy = accuracy_score(y_val, y_pred)
    print(accuracy)
    return model


def calc_error(model, X, y):
    return model.score(X, y)


# print("Default: ")
# run_all(X, y)
#
print("Scaled: ")
scalar = StandardScaler()
scaled_X = scalar.fit_transform(X)
# x_train, x_test, y_train, y_test = split(scaled_X, y)
# run_all(scaled_X, y)
#



print("Applying PCA...")
pca = PCA(n_components="mle", random_state=seed)
pca_X = pca.fit_transform(scaled_X)
# run_all(pca_X, y)
x_train, x_test, y_train, y_test = split(pca_X, y)
print(x_train.shape, x_test.shape)


#
# ensemble = train_random_forest(x_train, y_train)
# print("RandomForest")
# print(ensemble.score(x_test,y_test))
#
# ensemble = train_bagging(x_train, y_train)
# print("Bagging")
# print(ensemble.score(x_test,y_test))

log = train_logistic_regression(x_train, y_train)
print("Logistic Regression")
print(log.score(x_test,y_test))

# decision_tree = train_decision_tree(x_train, y_train)
# print("Decision Tree")
# print(decision_tree.score(x_test,y_test))
#
# ensemble = train_adaboost(x_train, y_train)
# print("Adaboost")
# print(ensemble.score(x_test, y_test))

## takes too long
# ensemble = train_grad_boost(x_train, y_train)
# print("Gradient Boosting")
# print(ensemble.score(x_test,y_test))
#




'''
VALIDATION RESULTS
Random Forest - 0.9361221481755188
Bagging - 0.938262812544241
Logistic Regression - 0.7245234
Decision Tree: 0.903911066
Adaboost - 0.6090437051647489

I chose the highest performing ones to create a stacking ensemble




TEST RESULTS:

Random Forest - 0.9425832379542697
Logistic Regression - 0.7235871707270897
Decision Tree - 0.9119127733363166
Adaboost - 0.604201268469833
Bagging - 0.9614295672228772





----
Logistic regression - 0.7235613538376806
Decision tree - 0.9399327039749404


Using the ensemble methods taught in lectures:
Ensemble Accuracies:

RandomForest - 0.9361221481755188

Adaboost - 0.6492345292290216
Gradient Boosting - 0.7124600913917885

Stacking - 0.6977530700584322 - combining randomForest and decisionTree
Stacking - 0.9628064679913599 - randomForest, decisionTree, and bagging
The bottom stacking is the best

'''
# Decision Tree and Random Forest have much higher scores, so are going to be optimised
# Fine-tuning best model (Random Forest)


def tree_params(best_params):
    params = {**best_params}
    model = DecisionTreeClassifier(**params, random_state=seed)
    return model

def forest_params(best_params):
    params = {**best_params}
    model = RandomForestClassifier(**params, random_state=seed)
    return model


def optimise_models(X, y):

    dt_params = optimise_decision_tree(x_train, y_train)
    model = tree_params(dt_params)
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print("Decision Tree: ", score)
    '''


    # Random Forest
    Best parameters: OrderedDict([('bootstrap', False), ('criterion', 'gini'), ('max_depth', None), ('max_features', 30), ('min_samples_leaf', 5), ('min_samples_split', 6), ('n_estimators', 112)])

    '''

    params = optimise_random_forest(x_test, y_test)
    print("1")
    # params = {'bootstrap': False, 'criterion': 'gini', 'max_depth': None, 'max_features': 30, 'min_samples_leaf': 5, 'min_samples_split': 6, 'n_estimators': 112}

    model = forest_params(params)
    print("2")
    score = cross_val_score(model, X, y, cv=3, scoring='accuracy')
    print("Ensemble: ", score)
    return params, dt_params


# forest_params, dt_params = optimise_models(x_train, y_train)



'''
Run stacking ensemble meth with DecisionTree & RandomForest (with optimal hyperparameters) and bagging (with default hyper params)


'''
# random forest
forest_params = {'bootstrap': False, 'criterion': 'gini', 'max_depth': None, 'max_features': 30, 'min_samples_leaf': 5,
          'min_samples_split': 6, 'n_estimators': 112}

dt_params = {'max_depth': None, 'max_features': 23, 'min_samples_leaf': 1, 'min_samples_split': 2, 'splitter': 'best', 'ccp_alpha': 0.0}
# Best parameters: OrderedDict([('ccp_alpha', 0.0), ('max_depth', None), ('max_features', 23), ('min_samples_leaf', 1), ('min_samples_split', 2), ('splitter', 'best')])
# Decision Tree:  [0.90343366 0.90138982 0.902046   0.90028184 0.9014533 ]
print("training...")
ensemble = train_stacking(x_train, y_train, dt_params, forest_params)
print("Stacking")
print(ensemble.score(x_test,y_test))