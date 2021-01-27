import numpy as np
import pandas as pd
from sklearn import metrics


def get_classification_metrics(y_true, y_pred):
    print('accuracy_score:', np.round(metrics.accuracy_score(y_true, y_pred), 4))
    print('precision_score:', np.round(metrics.precision_score(y_true, y_pred, average='weighted'), 4))
    print('recall_score:', np.round(metrics.recall_score(y_true, y_pred, average='weighted'), 4))
    print('f1_score:', np.round(metrics.f1_score(y_true, y_pred, average='weighted'), 4))
    
    
def get_regression_metrics(y_true, y_pred):
    print('mean_squared_error', np.round(metrics.mean_squared_error(y_true, y_pred), 4))
    print('explained_variance_score', np.round(metrics.explained_variance_score(y_true, y_pred), 4))
    print('mean_absolute_error', np.round(metrics.mean_absolute_error(y_true, y_pred), 4))
    print('mean_squared_error', np.round(metrics.mean_squared_error(y_true, y_pred), 4))
    print('median_absolute_error', np.round(metrics.median_absolute_error(y_true, y_pred), 4))
    print('r2_score', np.round(metrics.r2_score(y_true, y_pred), 4))
    
    
def get_confusion_matrix(y_true, y_pred, norm=False):    
    y_true = pd.Series(y_true, name='Actual')
    y_pred = pd.Series(y_pred, name='Predicted')
    
    df_cm = pd.crosstab(y_true, y_pred, margins=True)
    
    if norm:
        df_cm = df_cm / df_cm.sum(axis=1)
    
    return df_cm
    
    
def get_learning_curve(clf, y_true_all, X_all, metric):
    np.random.seed(0)

    plt.clf()

    fig = plt.figure(figsize=(18, 6))

    training_sizes = np.round(np.linspace(0.1, 0.9, 5), 10)

    df_results = pd.DataFrame(columns=['s', 'score', 'set'])

    for s in training_sizes:
        
        for i in range(0, 5):
            X_train, y_true_train, X_test, y_true_test = train_test_split(X_all, y_true_all, test_size=1-s)

            clf = clf.fit(X_train, y_true_train)

            train_s = eval_clf(clf, X_train, y_true_train, metric)
            test_s = eval_clf(clf, X_test, y_true_test, metric)

            df_results.loc[len(df_results.index)] = [s, test_s, 'test']
            df_results.loc[len(df_results.index)] = [s, train_s, 'train']

    ax = sns.pointplot(x='s', y='score', hue='set', data=df_results)

    ax.legend()
    ax.set_xlabel('training set size (%)')
    ax.set_ylabel(metric)
    
    return plt