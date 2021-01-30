import timeit
import numpy as np

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, \
    mean_squared_log_error, median_absolute_error, r2_score, \
    accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, \
    ExtraTreesClassifier, ExtraTreesRegressor, \
    AdaBoostRegressor, AdaBoostClassifier
from sklearn.svm import LinearSVR, SVR, LinearSVC, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, \
    KFold, ShuffleSplit, StratifiedShuffleSplit, TimeSeriesSplit


def build_clf_list(clf_select):
    ls_ref, ls_clf, ls_param = [], [], []

    ######################################################################
    # DATA PRE-PROCESSORS
    
    ######################################################################
    # sklearn.preprocessing.Normalizer
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html
    # Normalize samples individually to unit norm

    if "normp_0" in clf_select:
        ref = "normp_0"
        clf = "Normalizer()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.preprocessing.StandardScaler
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    # Standardize features by removing the mean and scaling to unit variance

    if "stdsp_0" in clf_select:
        ref = "stdsp_0"
        clf = "StandardScaler()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.preprocessing.MinMaxScaler
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    # Transforms features by scaling each feature to a given range

    if "mmsp_0" in clf_select:
        ref = "mmsp_0"
        clf = "MinMaxScaler()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))


    ######################################################################
    # REGRESSION BASED LEARNERS

    ######################################################################
    # sklearn.linear_model.LinearRegression
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    # Ordinary least squares Linear Regression

    if "olsr_0" in clf_select:
        ref = "olsr_0"
        clf = "LinearRegression()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "olsr_1" in clf_select:
        ref = "olsr_1"
        clf = "LinearRegression()"
        param = {
            'fit_intercept': [True, False],  # default=True
            'normalize': [True, False],  # default=False
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.linear_model.Lasso
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    # Linear Model trained with L1 prior as regularizer (aka the Lasso)

    if "lasr_0" in clf_select:
        ref = "lasr_0"
        clf = "Lasso()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "lasr_1" in clf_select:
        ref = "lasr_1"
        clf = "Lasso()"
        param = {
            'alpha': [.7, .9, 1.0, 1.1, 1.3],  # default=1.0
            'fit_intercept': [True, False],  # default=True
            'normalize': [True, False],  # default=False
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.linear_model.Ridge
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    # Linear least squares with l2 regularization

    if "ridr_0" in clf_select:
        ref = "ridr_0"
        clf = "Ridge()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "ridr_1" in clf_select:
        ref = "ridr_1"
        clf = "Ridge()"
        param = {
            'fit_intercept': [True, False],  # default=True
            'normalize': [True, False],  # default=False
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.linear_model.ElasticNet
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
    # Linear regression with combined L1 and L2 priors as regularizer

    if "enetr_0" in clf_select:
        ref = "enetr_0"
        clf = "ElasticNet()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "enetr_1" in clf_select:
        ref = "enetr_1"
        clf = "ElasticNet()"
        param = {
            'l1_ratio': [0.3, 0.4, 0.5, 0.6, 0.7],  # default=0.5
            'normalize': [True, False],  # default=False
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.tree.DecisionTreeRegressor
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    # A decision tree regressor

    if "dtr_0" in clf_select:
        ref = "dtr_0"
        clf = "DecisionTreeRegressor()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "dtr_1" in clf_select:
        ref = "dtr_1"
        clf = "DecisionTreeRegressor()"
        param = {
            'splitter': ['best', 'random'],  # default='best'
            'max_depth': [None, 2, 4, 6, 8],  # default=None
            'min_samples_split': [2, 3, 4, 5, 6]  # default=2
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.tree.ExtraTreeRegressor
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html
    # An extremely randomized tree regressor

    if "etr_0" in clf_select:
        ref = "etr_0"
        clf = "ExtraTreeRegressor()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "etr_1" in clf_select:
        ref = "etr_1"
        clf = "ExtraTreeRegressor()"
        param = {
            'splitter': ['best', 'random'],  # default='best'
            'max_depth': [None, 2, 4, 6, 8],  # default=None
            'min_samples_split': [2, 3, 4, 5, 6]  # default=2
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.ensemble.RandomForestRegressor
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    # A random forest regressor

    if "rfr_0" in clf_select:
        ref = "rfr_0"
        clf = "RandomForestRegressor()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "rfr_1" in clf_select:
        ref = "rfr_1"
        clf = "RandomForestRegressor()"
        param = {
            'n_estimators': [8, 9, 10, 11, 12],  # default=10
            'min_samples_split': [2, 3, 4, 5, 6],  # default=2
            'min_samples_leaf': [1, 2, 3],  # default=1
            'max_features': ['auto', 'sqrt', 'log2', None]  # default='auto'
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.ensemble.AdaBoostRegressor
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
    # An AdaBoost regressor (w Decision Tree Regressor)

    if "adabr_0" in clf_select:
        ref = "adabr_0"
        clf = "AdaBoostRegressor(DecisionTreeRegressor())"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "adabr_1" in clf_select:
        ref = "adabr_1"
        clf = "AdaBoostRegressor(DecisionTreeRegressor())"
        param = {
            'n_estimators': [30, 40, 50, 60, 70],  # default=50
            'learning_rate': [0.001, 0.01, 0.1, 1.0],  # default=1.0
            'loss': ['linear', 'square', 'exponential']  # default=linear
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.neighbors.KNeighborsRegressor
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    # Regression based on k-nearest neighbors

    if "knr_0" in clf_select:
        ref = "knr_0"
        clf = "KNeighborsRegressor()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "knr_1" in clf_select:
        ref = "knr_1"
        clf = "KNeighborsRegressor()"
        param = {
            'n_neighbors': [3, 4, 5, 6, 7],  # default=5
            'weights': ['uniform', 'distance'],  # default='uniform'
            'algorithm': ['auto', 'ball_tree', 'ks_tree', 'brute'],  # default='auto'
            'leaf_size': [20, 25, 30, 35, 40]  # default=30
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.svm.LinearSVR
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
    # Linear Support Vector Regression

    if "lsvr_0" in clf_select:
        ref = "lsvr_0"
        clf = "LinearSVR()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "lsvr_1" in clf_select:
        ref = "lsvr_1"
        clf = "LinearSVR()"
        param = {
            'epsilon': [0.001, 0.01, 0.1, 1.0, 10],  # default=0.1
            'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01],  # default=1e-4
            'fit_intercept': [True, False]  # default=True
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.svm.SVR
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    # Epsilon-Support Vector Regression

    if "svr_0" in clf_select:
        ref = "svr_0"
        clf = "SVR()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "svr_1" in clf_select:
        ref = "svr_1"
        clf = "SVR()"
        param = {
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed'],  # default='rbf'
            'epsilon': [0.001, 0.01, 0.1, 1.0, 10],  # default=0.1
            'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01],  # default=1e-4
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.neural_network.MLPRegressor
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    # Multi-layer Perceptron regressor

    if "mlpr_0" in clf_select:
        ref = "mlpr_0"
        clf = "MLPRegressor()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "mlpr_1" in clf_select:
        ref = "mlpr_1"
        clf = "MLPRegressor()"
        param = {
            'hidden_layer_sizes': [(10,), (50,), (100,)],  # default=(100,)
            'alpha': [0.0001, 0.001],  # default=0.0001
            'tol': [0.00001, 0.0001]  # default=1e-4
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))


    ######################################################################
    # CLASSIFICATION BASED LEARNERS

    ######################################################################
    # sklearn.linear_model.LogisticRegression
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # Logistic Regression (aka logit, MaxEnt) classifier

    if "logrc_0" in clf_select:
        ref = "logrc_0"
        clf = "LogisticRegression()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "logrc_1" in clf_select:
        ref = "logrc_1"
        clf = "LogisticRegression()"
        param = {
            'tol': [0.00001, 0.0001, 0.001],  # default=1e-4
            'C': [0.8, 0.9, 1.0, 1.1, 1.2]  # default=1.0
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.tree.DecisionTreeClassifier
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    # A decision tree classifier

    if "dtc_0" in clf_select:
        ref = "dtc_0"
        clf = "DecisionTreeClassifier()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "dtc_1" in clf_select:
        ref = "dtc_1"
        clf = "DecisionTreeClassifier()"
        param = {
            'criterion': ['gini', 'entropy'],  # default='gini'
            'splitter': ['random', 'best'],  # default='best'
            'max_depth': [None, 2, 4, 6, 8],  # default=None
            'max_features': ['auto', None]  # default=None
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.ensemble.ExtraTreesClassifier
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html
    # An extra-trees classifier

    if "etc_0" in clf_select:
        ref = "etc_0"
        clf = "ExtraTreeClassifier()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "etc_1" in clf_select:
        ref = "etc_1"
        clf = "ExtraTreeClassifier()"
        param = {
            'splitter': ['best', 'random'],  # default='random'
            'max_depth': [None, 2, 4, 6, 8],  # default=None
            'min_samples_split': [2, 3, 4, 5, 6]  # default=2
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.ensemble.RandomForestClassifier
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # A random forest classifier

    if "rfc_0" in clf_select:
        ref = "rfc_0"
        clf = "RandomForestClassifier()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "rfc_1" in clf_select:
        ref = "rfc_1"
        clf = "RandomForestClassifier()"
        param = {
            'n_estimators': [8, 10, 12],  # default=10
            'min_samples_split': [2, 3, 4, 5, 6],  # default=2
            'min_samples_leaf': [1, 2, 3],  # default=1
            'max_features': ['auto', 'sqrt', 'log2', None]  # default='auto'
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.ensemble.AdaBoostClassifier
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    # An AdaBoost classifier (w Decision Tree Classifier)

    if "adabc_0" in clf_select:
        ref = "adabc_0"
        clf = "AdaBoostClassifier(DecisionTreeClassifier())"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "adabc_1" in clf_select:
        ref = "adabc_1"
        clf = "AdaBoostClassifier(DecisionTreeClassifier())"
        param = {
            'n_estimators': [30, 40, 50, 60, 70],  # default=50
            'learning_rate': [0.001, 0.01, 0.1, 1.0],  # default=1.0
            'loss': ['linear', 'square', 'exponential']  # default=linear
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.neighbors.KNeighborsClassifier
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    # Classifier implementing the k-nearest neighbors vote

    if "knc_0" in clf_select:
        ref = "knc_0"
        clf = "KNeighborsClassifier()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "knc_1" in clf_select:
        ref = "knc_1"
        clf = "KNeighborsClassifier()"
        param = {
            'n_neighbors': [2, 3, 4, 5, 6],  # default=5
            'leaf_size': [10, 20, 30, 40, 50]  # default=30
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.svm.LinearSVC
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    # Linear Support Vector Classification

    if "lsvc_0" in clf_select:
        ref = "lsvc_0"
        clf = "LinearSVC()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "lsvc_1" in clf_select:
        ref = "lsvc_1"
        clf = "LinearSVC()"
        param = {
            'kernel': ['rbf', 'linear', 'poly'],  # default='rbf'
            'C': [0.001, 0.01, 0.1, 1.0],  # default=1.0
            'gamma': [0.0001, 0.001, 0.01, 0.1, 'auto'],  # default='auto'
            'tol': [0.00001, 0.0001, 0.001],  # default=1e-3
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.svm.SVC
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    # C-Support Vector Classification

    if "svc_0" in clf_select:
        ref = "svc_0"
        clf = "SVC()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "svc_1" in clf_select:
        ref = "svc_1"
        clf = "SVC()"
        param = {
            'kernel': ['rbf', 'linear', 'poly'],  # default='rbf'
            'C': [0.001, 0.01, 0.1, 1.0],  # default=1.0
            'gamma': [0.0001, 0.001, 0.01, 0.1, 'auto'],  # default='auto'
            'tol': [0.00001, 0.0001, 0.001],  # default=1e-3
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    ######################################################################
    # sklearn.neural_network.MLPClassifier
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    # Multi-layer Perceptron classifier

    if "mlpc_0" in clf_select:
        ref = "mlpc_0"
        clf = "MLPClassifier()"
        param = {}

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    if "mlpc_1" in clf_select:
        ref = "mlpc_1"
        clf = "MLPClassifier()"
        param = {
            'solver': ['lbfgs', 'sgd', 'adam'],  # default='adam'
            'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],  # default=0.0001
            'tol': [0.000001, 0.00001, 0.0001, 0.001]  # default=1e-4
        }

        ls_ref.append((ref))
        ls_clf.append((clf))
        ls_param.append((param))

    return ls_ref, ls_clf, ls_param


def eval_clf(y_true, y_pred, prob_type):
    dic_score = {}

    if prob_type == "regression":
        dic_score = {
            'explained_variance': explained_variance_score(y_true, y_pred),
            'mean_absolute_error': mean_absolute_error(y_true, y_pred),
            'mean_squared_error': mean_squared_error(y_true, y_pred),
            'mean_squared_log_error': mean_squared_log_error(y_true, abs(y_pred)),
            'median_absolute_error': median_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

    elif prob_type == "classification":
        dic_score = {
            'accuracy': accuracy_score(y_true, y_pred.round()),
            'f1': f1_score(y_true, y_pred.round()),
            'recall': recall_score(y_true, y_pred.round()),
            'precision': precision_score(y_true, y_pred.round()),
            'roc_auc_score:': roc_auc_score(y_true, y_pred.round())
        }

    return dic_score


def get_pred(clf, X, prob_type):
    y_pred = None

    if prob_type == "regression":
        y_pred = clf.predict(X)

    elif prob_type == "classification":
        y_pred = clf.predict_proba(X)[:, 1]

    return y_pred


def build_pipe(ref, clf, param):
    pipe_ref = []
    pipe_param = {}

    pipe_ref.append((ref, eval(clf)))

    for key, val in param.items():
        pipe_param[ref + '__' + key] = val

    return pipe_ref, pipe_param


def build_pipe_list(ref, clf, param):
    pipe_refs = []
    pipe_params = {}

    for ref, clf, param in zip(ref, clf, param):
        pipe_ref, pipe_param = build_pipe(ref, clf, param)

        pipe_refs.append((pipe_ref[0][0], pipe_ref[0][1]))
        pipe_params.update(pipe_param)

    return pipe_refs, pipe_params


def get_splitter(split, n_splits):
    splitter = None

    if split == "kfold":
        splitter = KFold(n_splits=n_splits)

    elif split == "shuffle":
        splitter = ShuffleSplit(train_size=None, n_splits=n_splits)

    elif split == "stratifiedshuffle":
        splitter = StratifiedShuffleSplit(train_size=None, n_splits=n_splits)

    elif split == "timeseries":
        splitter = TimeSeriesSplit(max_train_size=None, n_splits=n_splits)

    return splitter


def get_cv_optimizer(pipe, pipe_params, splitter, score, cv, n_jobs):
    cv_optimizer = None

    if cv == "gridsearch":
        cv_optimizer = GridSearchCV(pipe, pipe_params,
                                    cv=splitter, scoring=score, n_jobs=n_jobs)

    elif cv == "randomizedsearch":
        cv_optimizer = RandomizedSearchCV(pipe, pipe_params,
                                          cv=splitter, scoring=score, n_jobs=n_jobs)

    return cv_optimizer


def execute_pipe(clf_select, dic_X, dic_y, split_type, n_splits, cv_type, score_type, prob_type):
    np.random.seed(0)

    ref, clf, param = build_clf_list(clf_select)

    pipe_refs, dic_pipe_params = build_pipe_list(ref, clf, param)

    pipe = Pipeline(pipe_refs)

    splitter = get_splitter(split=split_type, n_splits=n_splits)

    start = timeit.default_timer()

    cv_optimizer = get_cv_optimizer(pipe, dic_pipe_params, splitter,
                                    score=score_type, cv=cv_type, n_jobs=1)

    cv_optimizer.fit(dic_X['X_train'], dic_y['y_true_train'])

    stop = timeit.default_timer()
    time = (stop - start)

    best_clf = cv_optimizer.best_estimator_

    dic_best = {
        'best_clf': best_clf,
        'best_param': cv_optimizer.best_params_,
        'best_score': cv_optimizer.best_score_
    }

    y_test_pred = get_pred(best_clf, X=dic_X['X_test'], prob_type=prob_type)

    dic_score = eval_clf(dic_y['y_true_test'], y_test_pred, prob_type=prob_type)

    return pipe_refs, dic_pipe_params, dic_best, dic_score, time