# Progress DS Coding Test - USA Income Analysis
### Matthew Burke

## General Approach 

I approached this problem by creating a class that would prepare the data, mimic the fit/predict paradigm of sklearn and also contain grid searching capabilities so we can reuse the same dataset for all purposes. This is in the `usa_income_analysis.py` file. 

The `run.py` file creates an instance of that class, loads the dataset, finds an optimal model and calculates the final scoring on a holdout dataset so we can have an idea of future performance.

## Model Search / Parameter Optimization

The two main classifiers I tested were logistic regression (which I chose due to the smaller size of the dataset) and the relatively small feature count, and a random forest classifier because it's robust, resistant to overfitting and it makes no assumptions on the collinearity of the variables (which is a concern for this dataset).
The main accuracy measure i was measuring it on was the F1 score due to the class imbalance in the target variable (~22000 vs ~7000 in each category) as well as not knowing whether we care more about precision or recall.


### Logistic Regression Grid Search Results
```
The best parameters were: {'penalty': 'l2', 'C': 2, 'solver': 'liblinear'}
The best training score was: 0.6645
The test score was: 0.6646
```

### Random Forest Grid Search Results
```
The best parameters were: {'max_features': 10, 'n_estimators': 25, 'max_depth': 20}
The best training score was: 0.6685
The test score was: 0.6903
```

The results are interesting in that I did not expect the random forest to have a higher F1 score on the test dataset than on the train dataset.. but at least it's generalizing well on the first pass.

### Final Test Results

Since it did perform better, I used the optimal parameters for testing on the holdout set we set apart in the beginning of our run script. 

Because it is data I didn't use in the grid search as well as general training, I believe we can expect similar results on unseen data since this was unseen data.

```
The final F1 score was: 0.8623
The final precision score was: 0.7962
The final recall score was: 0.9405
The final accuracy score was: 0.9352
```

### Notes

Please note that I did use the dev 0.20 release of scikit-learn so I could use the `CategoricalEncoder` that doesn't exist in 0.19 yet. 

### Future steps

I didn't do a ton of data exploration and correlation between covariates. I identified the one column (`education_num`) which was a duplicate of another column, but there may be other data issues I missed.

There could be potential feature engineering to do on these by joining additional demographic data in, but as it's a standalone dataset, there wasn't much I could do.

Additionally, I'd like to convert the scaler/encoder methods into a proper sklearn pipeline, but I didn't decide early enough in the process of structuring the class that I wanted to do it that way. It's a cleaner way of doing it, and I would definitely do it next time.

