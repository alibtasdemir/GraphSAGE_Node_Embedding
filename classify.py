import pandas as pd
import pickle
import os

import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
import os


def restore_model(model_name):
    return pickle.load(open(model_name, 'rb'))


def save_model(clf, filename='model', folder='classifier_models'):
    filename = filename + '.sav'
    if not os.path.exists(folder):
        os.makedirs(folder)
    to_path = os.path.join(folder, filename)
    pickle.dump(clf, open(to_path, 'wb'))


def do_prediction(Xin, yin, modType):
    Xin, Xtest, yin, ytest = train_test_split(Xin, yin, test_size=0.33, random_state=9)
    nfolds = 5
    cv = StratifiedKFold(n_splits=nfolds, shuffle=True)

    # Model type chosen as SVM. We perform a grid search to find optimal parameters.
    if modType in ('SVM', 'svm'):
        clasf = svm.LinearSVC()
        cvclasf = GridSearchCV(clasf, param_grid={
            'penalty': ['l1', 'l2'],
            'loss': ['hinge', 'squared-hinge'],
            'tol': [0.01, 0.001, 0.0005],
            'C': [0.1, 0.5, 1]
        }, verbose=0, refit=True,
                               cv=cv,
                               # scoring='roc_auc',
                               scoring='f1_macro',
                               n_jobs=1)

    # Logistic Regression
    elif modType in ('lr', 'log', 'logistic-regression'):
        clasf = LogisticRegression(max_iter=400)
        cvclasf = GridSearchCV(clasf, param_grid={
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'C': [0.1, 0.5, 1],
            'penalty': ['l1', 'l2'],
            'tol': [0.01, 0.001, 0.0005],
        }, verbose=0, refit=True,
                               cv=cv,
                               scoring='f1_macro',
                               n_jobs=1)

    # Random Forest
    elif modType in ('rf', 'random-forest'):
        clasf = RandomForestClassifier()
        cvclasf = GridSearchCV(clasf, param_grid={
            'n_estimators': [5, 10, 25, 50, 100],
            'criterion': ['entropy', 'gini'],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }, verbose=0, refit=True,
                               cv=cv,
                               scoring='f1_macro',
                               n_jobs=1)

    elif modType in ('sgd', 'stochastic', 'gradient-descent'):
        clasf = SGDClassifier()
        cvclasf = GridSearchCV(clasf, param_grid={
            'penalty': ['l1', 'l2'],
            'tol': [0.01, 0.001, 0.0005],
        }, verbose=0, refit=True,
                               cv=cv,
                               scoring='f1_macro',
                               n_jobs=1)

    # Fit GridSearch
    cvclasf.fit(Xin, yin)
    # Take the best classifier
    bclasf = cvclasf.best_estimator_
    # Print out best parameters chosen for given model.
    if not modType in ('naive', 'gnb'):
        print("%s %d-fold best CV params: %s" % (modType, nfolds, cvclasf.best_params_))

    return bclasf


def test_model(X_train, X_test, y_train, y_test, modelType):
    clasf = do_prediction(X_train, y_train, modelType)

    y_pred = clasf.predict(X_test)

    training_score, test_score = clasf.score(X_train, y_train), clasf.score(X_test, y_test)
    print('Accuracy of {} classifier on training set: {:.4f}'.format(modelType, training_score))
    print('Accuracy of {} classifier on test set: {:.4f}'.format(modelType, test_score))
    print(classification_report(y_test, y_pred))

    """
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True)
    plt.show()
    """
    save_model(clasf, filename=modelType)
    return training_score, test_score



if __name__ == "__main__":
    data = pd.read_csv("embeddings.csv")
    X, y = data.drop("Clique", axis=1), data["Clique"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # test_model(X_train, X_test, y_train, y_test, 'svm')
        # print("#" * 100)
        test_model(X_train, X_test, y_train, y_test, 'log')
        print("#" * 100)
        test_model(X_train, X_test, y_train, y_test, 'rf')
        # print("#" * 100)
        # test_model(X_train, X_test, y_train, y_test, 'sgd')
