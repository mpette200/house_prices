import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_validate, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMRegressor

raw_df = pd.read_csv('data/train_house_price.csv')
raw_test = pd.read_csv('data/test_house_price.csv')
X_train = raw_df.iloc[:, :-1].copy()
y_train = raw_df.iloc[:, -1].copy()

def get_features(X_train):
    '''Ignore columns with many nulls and split into text or numerical features'''
    cols_w_nulls = X_train.columns[X_train.isnull().mean() > 0.1]
    features_txt = [i for i in X_train.dtypes[X_train.dtypes=='object'].index if i not in cols_w_nulls]
    features_num = [i for i in X_train.dtypes[(X_train.dtypes=='int') | (X_train.dtypes=='float')].index
                   if i not in cols_w_nulls]
    features_num.remove('Id')
    return features_num, features_txt

class Target_mean_encoder(TransformerMixin, BaseEstimator):
    '''Do target mean encoding of dataframe. Category replaced with the mean of that category'''
    def __init__(self):
        self.computed_means = None
    def fit(self, X, y):
        self.computed_means = {
            column_name: dict(y.groupby(by=category).mean().iteritems())
            for column_name, category in X.iteritems()
        }
        return self
    def transform(self, X):
        df_out = X.copy()
        for column_name, means in self.computed_means.items():
            for category, mean in means.items():
                df_out[column_name].mask(df_out[column_name]==category, mean, inplace=True)
        # cope with values not in the fitted data
        df_out.replace(r'.*', np.nan, inplace=True, regex=True)
        df_out.fillna(value=df_out.mean(axis=0), inplace=True)
        return df_out.astype('f8')

class Ohe_nulls(TransformerMixin, BaseEstimator):
    '''One hot encoding that can cope with nulls'''
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore')
    def fit(self, X, y):
        df_tmp = X.fillna(value='null_ohe')
        self.encoder.fit(df_tmp, y)
        return self
    def transform(self, X):
        df_tmp = X.fillna(value='null_ohe_tx')
        return self.encoder.transform(df_tmp)

def create_pipe(model, feature_tuple, ohe=False):
    '''Create scikit-learn pipeline'''
    features_num, features_txt = feature_tuple
    transformed = FeatureUnion([
        ('square', FunctionTransformer(np.square, validate=True)),
        ('sqrt', FunctionTransformer(np.sqrt, validate=True))
    ])
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer()),
        ('transform', transformed),
        ('scaler', MinMaxScaler())
    ])
    txt_pipeline = Pipeline([
        ('encoder', Target_mean_encoder()),
        ('imputer', SimpleImputer()),
        ('scaler', MinMaxScaler())
    ])
    ohe_pipe = FeatureUnion([
        ('tm_enc', txt_pipeline),
        ('ohe', Ohe_nulls())
    ])
    if ohe:
        txt_encoder = ohe_pipe
    else:
        txt_encoder = txt_pipeline
    ctx = ColumnTransformer([
        ('numbers', numerical_pipeline, features_num),
        ('txt', txt_encoder, features_txt)
    ])
    model = Pipeline([
        ('columns', ctx),
        ('model', model)])
    return model

def test_model(model, X_train, y_train):
    '''Do cross-validation of model'''
    cv = KFold(n_splits=5, shuffle=True, random_state=192)
    cv_results = cross_validate(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, return_train_score=True)
    return pd.DataFrame(cv_results)

def tuning(model, X_train, y_train, params):
    '''Do hyperparameter tuning'''
    cv = KFold(n_splits=5, shuffle=True, random_state=192)
    grid = GridSearchCV(model, params, scoring='neg_mean_absolute_error', cv=cv, return_train_score=True)
    grid.fit(X_train, y_train)
    return pd.DataFrame(grid.cv_results_, dtype='float')

lgb = create_pipe(LGBMRegressor(max_depth=6), get_features(X_train), ohe=False)
params = {'model__num_leaves': range(2, 15)}
grid_results3 = tuning(lgb, X_train, y_train, params)
print(grid_results3.sort_values(by='mean_test_score', ascending=False))
grid_results3[['param_model__num_leaves', 'mean_train_score', 'mean_test_score']].plot(x='param_model__num_leaves', style='o', ms=10)
plt.grid()

lgb_model = create_pipe(LGBMRegressor(num_leaves=7, max_depth=6), get_features(X_train), ohe=False)
df_results = test_model(lgb_model, X_train, y_train)
print(df_results)

X_test = raw_test.copy()

p_model = create_pipe(LGBMRegressor(num_leaves=7, max_depth=6), get_features(X_train), ohe=False)
p_model.fit(X_train, y_train)
y_pred = p_model.predict(X_test)

def do_scatter(x, y, trend=True):
    '''scatter plot with optional trendline'''
    name = x.name
    if not trend:
        # categorical
        categories = y_train.groupby(X_train[name]).mean().sort_values().index
        x = OrdinalEncoder([categories]).fit_transform(x.to_frame())
    sns.regplot(x, y, ci=None, color='C1', fit_reg=trend, scatter_kws={'alpha': .3})
    plt.xlabel(name)
    plt.ylabel('Price (£)')
    if not trend:
        # categorical
        plt.xticks(ticks=range(len(categories)), labels=categories, rotation=90)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

def do_subplots(column_name, trend=True):
    fig, ax = plt.subplots(1, 2, sharex=trend, sharey=True, figsize=(12, 4))
    plt.sca(ax[0])
    do_scatter(X_train[column_name], y_train, trend=trend)
    plt.title('Training Set')
    plt.sca(ax[1])
    do_scatter(X_test[column_name], y_pred, trend=trend)
    plt.title('Test Set Predictions')

do_subplots('LotArea')
do_subplots('GrLivArea')
do_subplots('TotRmsAbvGrd')
do_subplots('TotalBsmtSF')
do_subplots('Neighborhood', trend=False)
print(y_pred[X_test['Neighborhood']=='StoneBr'].max())
print(y_train[X_train['Neighborhood']=='StoneBr'].max())
do_subplots('HouseStyle', trend=False)
do_subplots('YearBuilt')
