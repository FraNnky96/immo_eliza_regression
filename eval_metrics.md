### Eval metrics.

### Code snippet :

# Hyperparameter grid for GridSearchCV.

        param_grid = {
            "iterations": [500, 1000],
            "depth": [6, 8],
            "learning_rate": [0.1, 0.2],
            "early_stopping_rounds": [50],
        }
        cat_features = [0, 2, 7, 16, 17, 18]

# Create a CatBoostRegressor object.

        regressor = CatBoostRegressor(loss_function="RMSE", cat_features=cat_features)

# Perform GridSearchCV.

        grid_search = GridSearchCV(
            estimator=regressor, param_grid=param_grid, cv=2, n_jobs=-1, verbose=1
        )

# Fit the model with the best parameters.

        grid_search.fit(self.X_train, self.y_train)
        print("Best parameters:", grid_search.best_params_)

# Get the best model from GridSearchCV.

        self.regressor = grid_search.best_estimator_

### Table with value on train and test of the model on MAE, RMSE, R2, MAPE, sMAPE.

| MAE test | MAE train | RMSE test | RMSE train | R2 test | R2 train | MAPE test | MAPE train | sMAPE test | sMAPE train |
|----------|-----------|-----------|------------|---------|----------|-----------|------------|------------|-------------|
| 111385.46| 87577.89  | 214893.07 | 152649.60  | 76.53%  | 90.86 %  |  22.90 %  |  18.75 %   |   21.09 %  |   17.65 %   |


### List of features :

        For the features i used the basic ones from the dataset (Locality,Zip code,Property type,Bedrooms,Living area,Surface of the plot,Facades,Building condition,Fireplace,Equipped kitchen,Garden,Garden surface,Terrace,Terrace surface,Furnished,Swimming pool,Province,Property,Region)

        I struggled a lot with understanding how machine learning is working and spend too much time on it.

        Didn't had time left for scraping or features engenieering

### Accuracy Computing procedure.

        Split (train/test) = 80 / 20 % 
        Random state = 0

### Efficiency.

        Taking a bit more than 11 minutes to run completely from main.py

### Presentation of the final dataset .

        The final dataset is shaped with 12394 rows and 20 columns.

        I didn' t merge any dataset to the final dataset nor scrape spend too much time understanding logic of ML.

        I scaled the data with RobustScaler() .

        Made some clean some outliers , extras value, and default in the locality column

        Made some cross validation in the model


### Summary :

This model was built using CatBoostRegressor with a GridSearchCV to tune the hyperparameters. The evaluation metrics show that the model performs well on both training and test datasets. The model’s R² values indicate that it can explain a significant portion of the variance in the target variable, and its error metrics (MAE, RMSE, MAPE, and sMAPE) suggest that the model is making reasonable predictions. However, there is still room for improvement, particularly in reducing the error margins on the test set and on the dataset itself.