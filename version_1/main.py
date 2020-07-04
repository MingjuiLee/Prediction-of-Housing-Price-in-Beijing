import utils 
import model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def main():
    data = utils.read_file('./data/new.csv')
    print("There are %d samples in raw data set" % len(data))
    print("Raw input data set information")
    utils.data_info(data)
    utils.missing_info(data, "raw_missing")

    # handle format and garbled text issues in raw data
    data = utils.time_format(data)
    data = utils.garbled_drawing(data)
    data = utils.garbled_floor(data)
    data = utils.garbled_living(data)
    data = utils.garbled_bath(data)
    data = utils.garbled_construct(data)
    data = utils.strange_building(data)

    # drop columns that provide no help
    data = utils.drop_columns(data, ['url', 'id', 'price', 'DOM'])
    print("Raw data set information after transferring format and drop columns")
    utils.data_info(data)
    utils.missing_info(data, "raw_missing_2")

    # the rawdata contains more than 300000 data points, try to use 10% of rawdata in this project
    x_raw, y_raw, data, y = utils.data_splitting(data, data['totalPrice'], 0.1)
    data.to_csv('small.csv')
    print("smaller data set", np.shape(data), np.shape(y))
    print("y_info")
    print(y.describe())
    plt.hist(y)
    plt.xlabel("totalPrice")
    plt.ylabel("counts")
    plt.savefig('y.png')
    plt.close()

    # split D into D'' and D_Test
    x_doubleprime, y_doubleprime, x_test, y_test = utils.data_splitting(data, data['totalPrice'], 0.2)
    print("D'' shape", np.shape(x_doubleprime), np.shape(y_doubleprime))
    print("D_test shape", np.shape(x_test), np.shape(y_test))

    # split D'' into D' and D_pt
    x_prime, y_prime, x_pt, y_pt = utils.data_splitting(x_doubleprime, x_doubleprime['totalPrice'], 0.1)
    print("D_pt shape", np.shape(x_pt), np.shape(y_pt))
    print("D_prime shape", np.shape(x_prime), np.shape(y_prime))

    # Use pre-training set to look at data and conduct initial test
    print("Pre-training data set preprocessing:")
    utils.pre_training(x_pt)

    # D' data set preprocessing
    print("D' preprocessing:")
    x_train, y_train, x_val, y_val, cols_keep, imputation = utils.preprocessing(x_prime)
    print("D' after preprocessing:")
    print("D_train shape after preprocessing", np.shape(x_train), np.shape(y_train))
    print("D_val shape after preprocessing", np.shape(x_val), np.shape(y_val))

    # Linear Regression
    lin_reg = model.linear_regression(x_train, y_train, x_val, y_val, cols_keep)

    # Ridge Regression
    rid_reg = model.ridge_regression(x_train, y_train, x_val, y_val, cols_keep)

    # Lasso Regression
    las_reg = model.lasso_regression(x_train, y_train, x_val, y_val, cols_keep)

    # Random Forest
    rf = model.random_forest(x_train, y_train, x_val, y_val, cols_keep)

    # model tuning
    x, y, rid_reg_cv = model.ridge_cv(x_train, y_train, x_val, y_val, cols_keep)
    las_reg_cv = model.lasso_cv(x_train, y_train, x_val, y_val, cols_keep)
    rf_cv = model. random_forest_cv(x_train, y_train, x_val, y_val, cols_keep)

    # Final_result
    x_cv = pd.concat([x, y], axis=1)
    with_missing_cv = utils.missing_info(x_cv, "cv_missing")
    with_missing_test = utils.missing_info(x_test, "test_missing")
    continuous = ["Lng", "Lat", "square", "ladderRatio", "communityAverage"]
    discrete = ["Cid", "tradeTime", "followers", "livingRoom", "drawingRoom", "kitchen", "bathRoom", "floor",
                "buildingType", "constructionTime", "renovationCondition", "buildingStructure", "elevator",
                "fiveYearsProperty", "subway", "district"]
    x_cv, y_cv, x_test, y_test = utils.method_2_prime(x_cv, with_missing_cv, x_test, with_missing_test, continuous, discrete)
    x_test = pd.DataFrame(x_test, index=x_test.index, columns=cols_keep)

    predict_lin_red = lin_reg.predict(x_test)
    print("Linear Regression:")
    print("RMSE on test data = %.5f" % np.sqrt(mean_squared_error(y_test, predict_lin_red)))
    predict_rid_reg = rid_reg.predict(x_test)
    predict_rid_reg_cv = rid_reg_cv.predict(x_test)
    print("Ridge Regression:")
    print("RMSE on test data = %.5f" % np.sqrt(mean_squared_error(y_test, predict_rid_reg)))
    print("After cross validation, RMSE on test data = %.5f" % np.sqrt(mean_squared_error(y_test, predict_rid_reg_cv)))
    predict_las_reg = las_reg.predict(x_test)
    predict_las_reg_cv = las_reg_cv.predict(x_test)
    print("Lasso Regression:")
    print("RMSE on test data = %.5f" % np.sqrt(mean_squared_error(y_test, predict_las_reg)))
    print("After cross validation, RMSE on test data = %.5f" % np.sqrt(mean_squared_error(y_test, predict_las_reg_cv)))
    predict_rf = rf.predict(x_test)
    predict_rf_cv = rf_cv.predict(x_test)
    print("Random Forest:")
    print("RMSE on test data = %.5f" % np.sqrt(mean_squared_error(y_test, predict_rf)))
    print("After cross validation, RMSE on test data = %.5f" % np.sqrt(mean_squared_error(y_test, predict_rf_cv)))

    # plot
    plt.scatter(x_test['square'], y_test, c='blue', marker='o', label='real test')
    plt.scatter(x_test['square'], predict_rf_cv, c='red', marker='x', label='predict test')
    plt.xlabel('square')
    plt.legend(loc='upper right')
    plt.savefig("feature_square.png")
    plt.close()

    plt.scatter(x_test['livingRoom'], y_test, c='blue', marker='o', label='real test')
    plt.scatter(x_test['livingRoom'], predict_rf_cv, c='red', marker='x', label='predict test')
    plt.xlabel('livingRoom')
    plt.legend(loc='upper right')
    plt.savefig("feature_living.png")
    plt.close()


if __name__ == "__main__":
    main()