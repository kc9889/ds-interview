import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from usa_income_analysis import IncomeModeler

df = pd.read_csv('data/usa_income_data.csv')

train, test = train_test_split(df, train_size = .80)

m = IncomeModeler(train)
m.prepare_raw_data(target_col='income_class')

lr = LogisticRegression()
lr_params = {
    'penalty': ('l1', 'l2'), 
    'C': [0.1, 0.5, 0.75, 1, 2, 5],
    'solver': ['liblinear']
}
lr_best_params = m.grid_search(estimator=lr, param_grid=lr_params, scoring='f1')


rfc = RandomForestClassifier()
rfc_params = {
    'n_estimators': [5, 10, 15, 20, 25],
    'max_features': [5, 10, 30, 60],
    'max_depth': [5, 10, 15, 20]
}

rfr_best_params = m.grid_search(estimator=rfc, param_grid=rfc_params, scoring='f1')
# {'max_features': 10, 'n_estimators': 25, 'max_depth': 20}   

new_rfc = RandomForestClassifier(max_features=10, n_estimators=25, max_depth=20)

m.fit(new_rfc)

m.predict(test, 'income_class')

