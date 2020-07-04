import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor


def linear_regression(x_train, y_train, x_val, y_val, cols_keep):
    lin_reg = LinearRegression()
    x_train = pd.DataFrame(x_train, index=x_train.index, columns=cols_keep)
    x_val = pd.DataFrame(x_val, index=x_val.index, columns=cols_keep)
    lin_reg.fit(x_train, y_train)
    train_predict = lin_reg.predict(x_train)
    val_predict = lin_reg.predict(x_val)
    print("Linear Regression Model:")
    print("Train RMSE = %.5f" % np.sqrt(mean_squared_error(y_train, train_predict)))
    print("Validation RMSE = %.5f" % np.sqrt(mean_squared_error(y_val, val_predict)))
    print("Training score = %.5f" % lin_reg.score(x_train, y_train))
    print("Validation score = %.5f" % lin_reg.score(x_val, y_val))
    return lin_reg


def ridge_regression(x_train, y_train, x_val, y_val, cols_keep):
    rid_reg = Ridge()
    x_train = pd.DataFrame(x_train, index=x_train.index, columns=cols_keep)
    x_val = pd.DataFrame(x_val, index=x_val.index, columns=cols_keep)
    rid_reg.fit(x_train, y_train)
    train_predict = rid_reg.predict(x_train)
    val_predict = rid_reg.predict(x_val)
    print("Ridge Regression Model:")
    print("Train RMSE = %.5f" % np.sqrt(mean_squared_error(y_train, train_predict)))
    print("Validation RMSE = %.5f" % np.sqrt(mean_squared_error(y_val, val_predict)))
    print("Training score = %.5f" % rid_reg.score(x_train, y_train))
    print("Validation score = %.5f" % rid_reg.score(x_val, y_val))
    return rid_reg


def lasso_regression(x_train, y_train, x_val, y_val, cols_keep):
    las_reg = Lasso()
    x_train = pd.DataFrame(x_train, index=x_train.index, columns=cols_keep)
    x_val = pd.DataFrame(x_val, index=x_val.index, columns=cols_keep)
    las_reg.fit(x_train, y_train)
    train_predict = las_reg.predict(x_train)
    val_predict = las_reg.predict(x_val)
    print("Lasso Regression Model:")
    print("Train RMSE = %.5f" % np.sqrt(mean_squared_error(y_train, train_predict)))
    print("Validation RMSE = %.5f" % np.sqrt(mean_squared_error(y_val, val_predict)))
    print("Training score = %.5f" % las_reg.score(x_train, y_train))
    print("Validation score = %.5f" % las_reg.score(x_val, y_val))
    return las_reg


def random_forest(x_train, y_train, x_val, y_val, cols_keep):
    rf = RandomForestRegressor(max_features=7)
    x_train = pd.DataFrame(x_train, index=x_train.index, columns=cols_keep)
    x_val = pd.DataFrame(x_val, index=x_val.index, columns=cols_keep)
    rf.fit(x_train, y_train)
    train_predict = rf.predict(x_train)
    val_predict = rf.predict(x_val)
    print("Random Forest Model:")
    print("Train RMSE = %.5f" % np.sqrt(mean_squared_error(y_train, train_predict)))
    print("Validation RMSE = %.5f" % np.sqrt(mean_squared_error(y_val, val_predict)))
    print("Training score = %.5f" % rf.score(x_train, y_train))
    print("Validation score = %.5f" % rf.score(x_val, y_val))
    return rf


def ridge_cv(x_train, y_train, x_val, y_val, cols_keep):
    x_train = pd.DataFrame(x_train, index=x_train.index, columns=cols_keep)
    x_val = pd.DataFrame(x_val, index=x_val.index, columns=cols_keep)
    x = pd.concat([x_train, x_val])
    y = pd.concat([y_train, y_val])
    parameter_grid = {'alpha':np.logspace(-3, 2, 50)}
    grid_search = GridSearchCV(Ridge(), parameter_grid, cv=5)
    grid_search.fit(x, y)
    print("Ridge Regression model after cross validation:")
    best_alpha = grid_search.best_params_
    print("Lambda for Ridge Regression is: %.5f" % best_alpha['alpha'])
    print("Best score Ridge Regression is: %.5f" % grid_search.best_score_)
    rid_reg_cv = Ridge(alpha=best_alpha['alpha'])
    rid_reg_cv.fit(x, y)
    return x, y, rid_reg_cv


def lasso_cv(x_train, y_train, x_val, y_val, cols_keep):
    x_train = pd.DataFrame(x_train, index=x_train.index, columns=cols_keep)
    x_val = pd.DataFrame(x_val, index=x_val.index, columns=cols_keep)
    x = pd.concat([x_train, x_val])
    y = pd.concat([y_train, y_val])
    parameter_grid = {'alpha':np.logspace(-3, 2, 50)}
    grid_search = GridSearchCV(Lasso(), parameter_grid, cv=5)
    grid_search.fit(x, y)
    print("Lasso Regression model after cross validation:")
    best_alpha = grid_search.best_params_
    print("Lambda for Lasso Regression is: %.5f" % best_alpha['alpha'])
    print("Best score of Lasso Regression is: %.5f" % grid_search.best_score_)
    las_reg_cv = Lasso(alpha=best_alpha['alpha'])
    las_reg_cv.fit(x, y)
    return las_reg_cv


def random_forest_cv(x_train, y_train, x_val, y_val, cols_keep):
    x_train = pd.DataFrame(x_train, index=x_train.index, columns=cols_keep)
    x_val = pd.DataFrame(x_val, index=x_val.index, columns=cols_keep)
    x = pd.concat([x_train, x_val])
    y = pd.concat([y_train, y_val])
    parameter_grid = {'n_estimators': [100, 250, 350, 500], 'max_depth': [5, 25, 50, 75]}
    grid_search = GridSearchCV(RandomForestRegressor(max_features=7), parameter_grid, cv=5)
    grid_search.fit(x, y)
    print("Random Forest model after cross validation:")
    best_parameters = grid_search.best_params_
    print("best parameters", best_parameters)
    print("Best score of Random Forest is: %.5f" % grid_search.best_score_)
    rf_cv = RandomForestRegressor(max_features=7, n_estimators=best_parameters['n_estimators'],
                                  max_depth=best_parameters['max_depth'])
    rf_cv.fit(x, y)
    return rf_cv
