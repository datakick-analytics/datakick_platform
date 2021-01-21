import numpy as np
from sklearn import metrics


def get_classification_metrics(y_true, y_pred):
    print('accuracy_score:', np.round(metrics.accuracy_score(y_true, y_pred), 4))
    print('precision_score:', np.round(metrics.precision_score(y_true, y_pred, average='weighted'), 4))
    print('recall_score:', np.round(metrics.recall_score(y_true, y_pred, average='weighted'), 4))
    print('f1_score:', np.round(metrics.f1_score(y_true, y_pred, average='weighted'), 4))
    
    
def get_regression_metrics(y_true, y_pred):
    print('mean_squared_error', np.round(mean_squared_error(y_true, y_pred), 4))
    print('explained_variance_score', np.round(explained_variance_score(y_true, y_pred), 4))
    print('mean_absolute_error', np.round(mean_absolute_error(y_true, y_pred), 4))
    print('mean_squared_error', np.round(mean_squared_error(y_true, y_pred), 4))
    print('median_absolute_error', np.round(median_absolute_error(y_true, y_pred), 4))
    print('r2_score', np.round(r2_score(y_true, y_pred), 4))