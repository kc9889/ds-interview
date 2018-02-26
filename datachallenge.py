import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import numpy as np
import xgboost
import matplotlib.pyplot as plt
import sys

"""
Instructions.
Please run the script with 1 or 2 arguments.
The first argument should be the file path of the CSV file.
The second (optional) argument should be y or n, indicating whether you want to tune the hyperparameters (yes), or use the parameters that were empirically found to show good performance (no).
By default, not adding the second argument will perform hyperparameter tuning.
"""

"""
In our dataset, we have numerical features (ex. age, hours per week, etc.), but we also have categorical features (ex. education, race, etc.).
A dataset of mixed feature types naturally lends itself to tree based models, so I decided to go with XGBoost, which makes use of boosted trees.
First, I preprocessed the dataset to remove rows containing missing values. 
The values appeared to be missing at random, so removing the data points should not bias the model, although losing data points may weaken its ability to generalize.
Then, I used 5 fold cross validation to tune the hyperparameters maximum tree depth, learning rate, and the minimum loss parameter.
The models were evaluated based on their f1 scores because the dataset is slightly unbalanced, with the '<=50k' class making up about 75% of the data.
The best fit model was evaluated and its prediction accuracy was found to be 86.75% with a standard deviation of 0.21% using a learning rate of 0.05, a max depth of 4, and a gamma of 4.
Further analysis showed that the features 'fnlwgt', age, capital gains/losses, occupation, and hours per week were among the most significant predictors for income class.
While I am unsure of what 'fnlwgt' signifies, it is fairly clear how the rest of the results affect income.
Age is correlated with work experience, capital gains and losses are directly tied to income, income typically increases with number of hours worked per week, and different jobs have different income ranges.
Additional tests can be done to tune hyperparameters beyond the scope that I have tested here.
For example, tuning the L1/L2 regularization parameters or expanding the range of the gamma parameter may improve the model's performance further.
"""

class TrainingData:
	def __init__(self, csv):
		"""
		Read csv file into a pandas dataframe for further processing.
		It is assumed that the last column is the target variable.
		"""
		self.df = pd.read_csv(csv)
		self.feature_names = self.df.columns.values[0:-1]

	def preprocess(self):
		"""
		The given file contains missing values in the form of ' ?'.
		Imputation of these missing values can possibly be considered through techniques such as k-NN.
		However, since the missing values appear to be randomly dispersed, they are simply dropped.
		"""
		self.df = self.df.replace(' ?', np.nan).dropna()
		for feature in self.df.columns:
			if self.df[feature].dtype == 'object':
				self.df[feature] = pd.Categorical(self.df[feature]).codes

	def get_training_data(self):
		self.preprocess()
		return self.df[self.feature_names], self.df['income_class']

class Classifier:
	def __init__(self, predictors, labels, max_depth=[4,6,8], eta=[0.05,0.1,0.3], gamma=[0,2,4], num=5):
		"""
		Among the several hyperparameters, only max_depth, learning_rate, and the gamma regularization parameter have been chosen for tuning.
		Perhaps tuning of additional hyperparameters may increase the model's performance.
		Stratified k-fold cross validation is used in training the model.
		Stratification is useful in binary classification because one class may dominate the majority of the data.
		Other techniques to overcome such scenarios include undersampling and oversampling.
		In this case, the '<=50k' income class make up about 75% of the data.
		While the data set is not severely unbalanced, stratified k-fold cross validation is used to ensure that both classes in the training sets are well represented.
		"""
		self.cv_params = {'max_depth':max_depth, 
						  'learning_rate':eta,
						  'gamma':gamma}
		self.ind_params = {'n_estimators':1000, 
						   'seed':0, 
						   'subsample':0.8, 
						   'colsample_bytree':0.8, 
						   'objective':'binary:logistic', 
						   'min_child_weight':1}
		self.kfold = sklearn.model_selection.StratifiedKFold(n_splits=num, random_state=1)
		self.predictors = predictors
		self.labels = labels

	def train_model(self, score_type='accuracy', fpath=None, supress=False):
		"""	
		Grid search is used to find the best combination of hyperparameters.
		Accuracy is the default scoring metric. However, I shall opt to use f1 score to account for the slightly unbalanced dataset.
		"""
		self.model = GridSearchCV(xgboost.XGBClassifier(**self.ind_params), self.cv_params, scoring=score_type, cv=self.kfold, n_jobs=-1)
		self.model.fit(self.predictors, self.labels)
		
		print(self.model.best_params_)
		self.best_model = xgboost.XGBClassifier(**self.model.best_params_, **self.ind_params)
		self.print_results()

		self.best_model = self.best_model.fit(self.predictors, self.labels)
		if not supress:
			self.show_fscores()

	def show_fscores(self):
		"""
		Plot the F scores of features to see which ones have the highest relative importance in the model.
		A variable's F score is a metric that sums up how many times the feature was split on in the boosted decision trees.
		I am not sure what 'fnlwgt' signifies, but it seems to be the most important feature.
		Age, capital gains/losses, occupation, and hours per week follow suit.
		This makes sense because the older you are, the more experience you are likely to have, so the more you should be paid.
		If you work long hours, you should be expected to be compensated more.
		Occupation also affects your income range.
		And, of course, capital gains or losses directly impact income.
		"""
		xgboost.plot_importance(self.best_model)
		plt.tight_layout()
		plt.show()

	def print_results(self):
		results = sklearn.model_selection.cross_val_score(self.best_model, self.predictors, self.labels, cv=self.kfold)
		print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

	def predict(self, data):
		"""
		Data should be in the same format as the training set.
		"""
		predictions = list(self.best_model.predict(data))
		for i in range(len(predictions)):
			if predictions[i] == 0:
				predictions[i] = '<=50K'
			else:
				predictions[i] = '>50K'
		probs = self.best_model.predict_proba(data)
		return (predictions, probs)

	def use_best_params(self, max_depth=4, eta=0.05, gamma=4, supress=False):
		"""
		Apply hyperparameters that were found to have the best performance based on the training set.
		"""
		self.cv_params = {'max_depth':max_depth, 
						  'learning_rate':eta,
						  'gamma':gamma}
		self.best_model = xgboost.XGBClassifier(**self.cv_params, **self.ind_params)
		self.print_results()

		self.best_model = self.best_model.fit(self.predictors, self.labels)
		if not supress:
			self.show_fscores()


def main(csv, use_best='n'):	
	income_data = TrainingData(csv)
	predictors, labels = income_data.get_training_data()

	model = Classifier(predictors, labels)
	"""
	The default scoring metric is accuracy. 
	However, using the f1 score provides slightly better performance.
	This could be because f1 scores balance precision and recall in unbalanced data.
	"""
	if use_best.lower() == 'y':
		model.use_best_params()
	else:
		best_model = model.train_model(score_type='f1')


if __name__ == '__main__':
	if len(sys.argv) == 2:
		main(sys.argv[1])
	elif len(sys.argv) == 3:
		if sys.argv[2].lower() not in ['y', 'n']:
			print('Please designate \'y\' for yes or \'n\' for no for the second parameter')
			sys.exit()
		main(sys.argv[1], sys.argv[2])
	else:
		print('Please run the program with one argument to specify file path of the CSV file\n' +\
			  'Or add a second argument, y or n, to specify whether to use a pre-determined set of hyperparameters or to train the model')
		sys.exit()

