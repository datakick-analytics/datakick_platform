import pandas as pd
from sklearn import preprocessing


def fit_transform_ohe(df, col_name):
    le = preprocessing.LabelEncoder()
    
    df = df[col_name].copy()
    le_labels = le.fit_transform(df)
    df[col_name + '_label'] = le_labels
    
    ohe = preprocessing.OneHotEncoder()
    
    feature_arr = ohe.fit_transform(df[col_name + '_label'].reshape(-1, 1)).toarray()
    feature_labels = [col_name + '_' + str(cls_label) for cls_label in le.classes_]
    df_feature = pd.DataFrame(feature_arr, columns=feature_labels)
    
    return le, ohe, df_feature


def transform_ohe(df, le, ohe, col_name):
    df = df[col_name].copy()
    col_labels = le.transform(df)
    df[col_name + '_label'] = col_labels
    
    feature_arr = ohe.fit_transform(df[col_name + '_label'].reshape(-1, 1)).toarray()
    feature_labels = [col_name + '_' + str(cls_label) for cls_label in le.classes_]
    df_feature = pd.DataFrame(feature_arr, columns=feature_labels)
    
    return df_feature