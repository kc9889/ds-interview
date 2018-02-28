import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, CategoricalEncoder, LabelEncoder
# Testing out the new CategoricalEncoder from the upcoming 0.20 scikit-learn release (still in dev)
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

class IncomeModeler:
    '''This class allows the user to load data, convert it into a model 
       ready format, perform gridsearches to find a well-performing model, 
       and perform data preparation and predictions on a second dataset using 
       the trained model
    '''
    
    
    def __init__(self, data):
        self.raw_data = data
        self.scaler = None 
        self.encoder = None
        self.le = None
        
    
    def prepare_raw_data(self, target_col):
        features, targets = self.prepare_data(self.raw_data, target_col, training=True)
        train_data, test_data, train_target, test_target = train_test_split(features, targets, train_size=0.67)
        self.full_features, self.full_targets = features, targets
        self.train_data = train_data
        self.test_data = test_data
        self.train_target = train_target
        self.test_target = test_target
        
        
    def prepare_data(self, data, target_col, training):
        '''Prepares the raw data into a form suitable for ingestion
           into a model. Returns prepared features and targets
           
           If training is true, then it saves the categorical encoder,
           scaler and labelbinarizer for reuse in later datasets.
           
           If training is false, applies those saved transformers to 
           the new dataset.
        '''
        data = self.raw_data
        
        # The education_num is a field of integers that correspond directly with
        # the values in education column. Removing since it's redundant.
        if 'education_num' in data.columns:
            data.drop('education_num', axis=1, inplace=True)
        
        # Missing values denoted ?
        # Removing here for the sake of time instead of imputing them since 
        # they make up less than 1% of dataset
        data = data[data['workclass'] != ' ?']
        
        # Split out features and targets 
        features = data[data.columns[data.columns != target_col]]
        targets = data[target_col]
        
        # Split out numeric columns and categorical columns 
        numeric_cols = features.columns[features.dtypes == np.int64]
        cat_cols = features.columns[features.dtypes != np.int64]
        
        # Normalize the numeric features for use in logistic regression 
        # with regularization. Create new scaler if training, apply 
        # existing scaler if not
        if training:
            scaler = StandardScaler()
            numeric_features = scaler.fit_transform(features[numeric_cols])
            self.scaler = scaler
        else:
            numeric_features = self.scaler.transform(features[numeric_cols])
        
        # One hot encode all categorical columns. Create new encoder
        # if training, apply existing encoder if not
        if training:
            encoder = CategoricalEncoder(handle_unknown='ignore')
            cat_features = encoder.fit_transform(features[cat_cols])
            self.encoder = encoder
        else:
            cat_features = self.encoder.transform(features[cat_cols])
    
        # Label encode the target column. Create new label encoder
        # if training, apply existing encoder if not
        if training:
            le = LabelEncoder()
            new_targets = le.fit_transform(targets)
            self.le = le
        else:
            new_targets = self.le.transform(targets)

        X = np.concatenate((np.array(numeric_features), np.array(cat_features.todense())), axis=1)
        Y = new_targets
        
        return(X, Y)
    
    def grid_search(self, estimator, param_grid, scoring, cv=3):
        '''Runs a grid search on the inputted estimator and parameters.
           Reports on the best parameters and train vs test scores.
           Returns the best parameters.
        '''
        gs = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, cv=cv)
        gs.fit(self.train_data, self.train_target)
        
        best_est = gs.best_estimator_
        best_params = gs.best_params_
        train_score = gs.best_score_
        
        print('The best parameters were: %s' % str(best_params))
        print('The best training score was: %s' % str(train_score))
        
        predictions = best_est.predict(self.test_data)
        
        if scoring == 'f1':
            print('The test score was: %s' % str(f1_score(predictions, self.test_target)))
        elif scoring == 'accuracy':
            print('The test accuracy score was: %s' % accuracy_score(predictions, self.test_target))

        return(best_params)
    
    def fit(self, model):
        '''Trains the input model on the full dataset
        '''
        self.model = model.fit(self.full_features, self.full_targets)
        

    def predict(self, data, target_col):
        '''
        '''
        test_data, test_targets = self.prepare_data(data=data, target_col=target_col, training=False)
        predictions = self.model.predict(test_data)
        
        print('The final F1 score was: %s' % f1_score(predictions, test_targets))
        print('The final precision score was: %s' % precision_score(predictions, test_targets))
        print('The final recall score was: %s' % recall_score(predictions, test_targets))
        print('The final accuracy score was: %s' % accuracy_score(predictions, test_targets))
        
        


















