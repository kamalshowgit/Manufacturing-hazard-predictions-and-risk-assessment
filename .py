import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xgboost as xgb

datafile_train = "Hazard_train.csv"
datafile_test = "Hazard_test_share.csv"
bd_train = pd.read_csv(datafile_train)
bd_test = pd.read_csv(datafile_test)
bd_train.head()

cat_cols = bd_train.select_dtypes(['object']).columns
cat_cols

bd_train['data'] = 'train'
bd_test['data'] = 'test'
all_data = pd.concat([bd_train, bd_test], axis=0, sort=False)
all_data.shape

for col in cat_cols:
    k = all_data[col].value_counts()
    cats = k[k >= 100].index[:-1]

    for cat in cats:
        name = col + '_' + cat
        all_data[name] = (all_data[col] == cat).astype(int)

    del all_data[col]

all_data.shape
all_data.head()

x_train = all_data.drop(['Id', 'Hazard', 'data'], axis=1)[all_data['data'] == 'train']
y_train = all_data['Hazard'][all_data['data'] == 'train']
x_train.shape, y_train.shape

x_test = all_data.drop(['Id', 'Hazard', 'data'], axis=1)[all_data['data'] == 'test']

dtrain = xgb.DMatrix(x_train, label=y_train)
params = {
    'objective': 'count:poisson',
    'eval_metric': 'poisson-nloglik'
}
model = xgb.train(params, dtrain)

dtest = xgb.DMatrix(x_test)
y_pred = model.predict(dtest)

submissions = pd.DataFrame({'Id': bd_test['Id'], 'Hazard': y_pred})
submissions.to_csv('submission.csv', index=False)
