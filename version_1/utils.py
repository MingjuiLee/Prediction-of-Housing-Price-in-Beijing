import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def read_file(filename):
    data = pd.read_csv(filename, sep=',', encoding='iso-8859-1', low_memory=False)
    print('File reading is completed, total %d samples in the raw data' % (len(data)))
    return data


def data_info(data):
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    pd.set_option('precision', 5)
    data.info()
    print(data.describe())
    print(data.head())
    columns = list(data)
    print("\nColumns in raw data: ", columns)
    for column in columns:
        print("\nColumn name: %s" % column)
        print("index", columns.index(column))
        print(data[column].value_counts())
        print(data[column].unique())


def missing_info(data, saved_name):
    missing_info = data.isnull().sum()
    print("\n%s missing data information:" % saved_name)
    print(missing_info[missing_info > 0].sort_values(ascending=False))
    plt.figure(figsize=(20, 15))
    plt.title('Missing value counts')
    sns.barplot(x=missing_info.index, y=missing_info.values)
    plt.xticks(rotation=90)
    plt.savefig(saved_name)
    plt.close()

    with_missing = []
    for column in data:
        missing = data[column].isnull().sum()
        if missing != 0:
            with_missing.append(column)
    print("with_missing", with_missing)
    return with_missing

def time_format(data):
    time_list = []
    for i in range(len(data['tradeTime'])):
        time_struct = time.strptime(data['tradeTime'][i], "%Y-%m-%d")
        str_time = time.strftime("%Y%m%d", time_struct)
        str_time = int(str_time)
        time_list.append(str_time)
    data['tradeTime'] = time_list
    return data


def garbled_floor(data):
    floor_new = []
    for i in range(len(data["floor"])):
        if re.findall(r"\d+", data["floor"][i]):
            flr = re.findall(r"\d+", data["floor"][i])
            floor_new.append(int(flr[0]))
        else:
            floor_new.append(-1)
    data["floor"] = floor_new
    return data


def garbled_drawing(data):
    drawing_new = []
    for i in range(len(data['drawingRoom'])):
        if re.findall(r"\d+", data['drawingRoom'][i]):
            drawing = re.findall(r"\d+", data['drawingRoom'][i])
            drawing_new.append(int(drawing[0]))
    data['drawingRoom'] = drawing_new
    return data


def garbled_living(data):
    data['livingRoom'] = data['livingRoom'].replace({'#NAME?': np.nan})
    data["livingRoom"] = data["livingRoom"].astype("float")
    return data


def garbled_bath(data):
    data['bathRoom'] = data['bathRoom'].replace({'Î´Öª': np.nan})
    data["bathRoom"] = data["bathRoom"].astype("float")
    return data


def garbled_construct(data):
    data['constructionTime'] = data['constructionTime'].replace({'Î´Öª': np.nan})
    data["constructionTime"] = data["constructionTime"].astype("float")
    return data


def strange_building(data):
    correct_type = [1, 2, 3, 4, np.nan]
    raw_type = data['buildingType'].unique()
    strange_type = []
    for t in raw_type:
        if t not in correct_type:
            strange_type.append(t)
    print("strange type", strange_type)
    data['buildingType'] = data['buildingType'].replace(strange_type, np.nan)
    data["buildingType"] = data["buildingType"].astype("float")
    return data


def drop_columns(data, drop_list):
    for column in drop_list:
        data = data.drop([column], axis=1)
    return data


def data_splitting(data, y, size):
    x_1, x_2, y_1, y_2 = train_test_split(data, y, test_size=size, random_state=42)
    return x_1, y_1, x_2, y_2


def missing_method_1(x_data):
    x_data_1 = x_data.dropna()
    for column in x_data_1:
        missing = x_data_1[column].isnull().sum()
        print("%s: missing value %d" % (column, missing))
    y_data = x_data_1['totalPrice']
    x_data = x_data_1.drop(['totalPrice'], axis=1)
    return x_data, y_data


def missing_method_2(x_data, with_missing, continuous, discrete):
    for feature in with_missing:
        if feature in continuous:
            x_data[feature] = x_data[feature].fillna(x_data[feature].mean())
        elif feature in discrete:
            x_data[feature] = x_data[feature].fillna(x_data[feature].mode()[0])
    y_data = x_data['totalPrice']
    x_data = x_data.drop(['totalPrice'], axis=1)
    return x_data, y_data


def normalization(x_train):
    scaler_norm = MinMaxScaler()
    x_train_norm = scaler_norm.fit_transform(x_train.values)
    x_train_norm = pd.DataFrame(x_train_norm, index=x_train.index, columns=x_train.columns)
    return x_train_norm, scaler_norm


def standardization(x_train):
    scaler_stnd = StandardScaler()
    x_train_norm = scaler_stnd.fit_transform(x_train.values)
    x_train_norm = pd.DataFrame(x_train_norm, index=x_train.index, columns=x_train.columns)
    return x_train_norm, scaler_stnd


def correlation(data):
    corr_matrix = data.corr()
    print("Features correlation:")
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_matrix)
    plt.savefig("corr.png")
    plt.close()
    print(corr_matrix["totalPrice"].sort_values(ascending=False))
    sorted_corr = abs(corr_matrix["totalPrice"]).sort_values(ascending=False)
    sorted_corr = sorted_corr.drop("totalPrice")
    print("correlation after sorting\n", sorted_corr, type(sorted_corr))
    corr_list = []
    for feature in sorted_corr.index:
        corr_list.append(feature)
    print("corr list", corr_list)
    return sorted_corr, corr_list


def same_elem(list1, list2):
    set_1 = set(list1)
    set_2 = set(list2)
    intersection = set_1.intersection(set_2)
    return list(intersection)


def pretraining_score(x_pt, y_pt):
    linear_reg = LinearRegression()
    linear_reg = linear_reg.fit(x_pt, y_pt)
    train_predict = linear_reg.predict(x_pt)
    train_rmse = np.sqrt(mean_squared_error(y_pt, train_predict))
    return train_rmse, train_predict


def pre_training(x_pt):
    print("Pre-training data set information:")
    data_info(x_pt)
    with_missing = missing_info(x_pt, "pretraining_missing")
    continuous = ["Lng", "Lat", "square", "ladderRatio", "communityAverage"]
    discrete = ["Cid", "tradeTime", "followers", "livingRoom", "drawingRoom", "kitchen", "bathRoom", "floor",
                "buildingType", "constructionTime", "renovationCondition", "buildingStructure", "elevator",
                "fiveYearsProperty", "subway", "district"]
    y_mean = [np.mean(x_pt['totalPrice'])] * len(x_pt["totalPrice"])
    print("y_pt mean = ", y_mean[0])
    baseline_rmse = np.sqrt(mean_squared_error(x_pt["totalPrice"], y_mean))
    print("Baseline_rmse of pre-training data set = %.5f" % baseline_rmse)

    # look at pre-training data
    x_pt.hist(bins=50, figsize=(20, 15))
    plt.savefig('x_pt.png')
    plt.close()

    sns.distplot(x_pt["totalPrice"])
    plt.savefig('y_pt.png')
    plt.close()

    # missing value method 1: drop all rows with missing value + standardization
    x_pt_1, y_pt_1 = missing_method_1(x_pt)
    print("pt1 shape", np.shape(x_pt_1), np.shape(y_pt_1))
    rmse_pt_1, predict_1 = pretraining_score(x_pt_1, y_pt_1)
    print("RMSE of pre-training set with missing value imputation method 1 = %.5f" % rmse_pt_1)

    # missing value method 2: fill in mean for continuous features, mode for discrete features
    x_pt_2, y_pt_2 = missing_method_2(x_pt, with_missing, continuous, discrete)
    print("pt2 shape", np.shape(x_pt_2), np.shape(y_pt_2))
    rmse_pt_2, predict_2 = pretraining_score(x_pt_2, y_pt_2)
    print("RMSE of pre-training set with missing value imputation method 2 = %.5f" % rmse_pt_2)

    # normalization
    print("x_pt 1 predict top 5\n", predict_1[0:5])
    x_pt_1_norm, scaler_norm_1 = normalization(x_pt_1)
    rmse_pt_norm_1, predict_norm_1 = pretraining_score(x_pt_1_norm, y_pt_1)
    print("RMSE of pre-training set with missing value imputation method 1 and normalization = %.5f" % rmse_pt_norm_1)
    x_pt_2_norm, scaler_norm_2 = normalization(x_pt_2)
    rmse_pt_norm_2, predict_norm_2 = pretraining_score(x_pt_2_norm, y_pt_2)
    print("RMSE of pre-training set with missing value imputation method 2 and normalization = %.5f" % rmse_pt_norm_2)

    # standardization
    x_pt_1_stnd, scaler_stnd_1 = standardization(x_pt_1)
    rmse_pt_stnd_1, predict_stnd_1 = pretraining_score(x_pt_1_stnd, y_pt_1)
    x_pt_1_stnd.hist(bins=50, figsize=(20, 15))
    plt.savefig('x_pt_1_stnd.png')
    plt.close()

    print("RMSE of pre-training set with missing value imputation method 1 and standardization = %.5f" % rmse_pt_stnd_1)
    x_pt_2_stnd, scaler_stnd_2 = standardization(x_pt_2)
    rmse_pt_stnd_2, predict_stnd_2 = pretraining_score(x_pt_2_stnd, y_pt_2)
    print("RMSE of pre-training set with missing value imputation method 2 and standardization = %.5f" % rmse_pt_stnd_2)

    # Features correlation
    sorted_corr, corr_list = correlation(x_pt)
    corr_rmse_1 = []
    corr_rmse_norm_1 = []
    corr_rmse_stnd_1 = []
    corr_rmse_2 = []
    corr_rmse_norm_2 = []
    corr_rmse_stnd_2 = []
    for i in range(1, len(corr_list) + 1):
        columns = corr_list[0:i]
        x_1 = pd.DataFrame(x_pt_1, index=x_pt_1.index, columns=columns)
        x_n_1 = pd.DataFrame(x_pt_1_norm, index=x_pt_1_norm.index, columns=columns)
        x_s_1 = pd.DataFrame(x_pt_1_stnd, index=x_pt_1_stnd.index, columns=columns)
        x_2 = pd.DataFrame(x_pt_2, index=x_pt_2.index, columns=columns)
        x_n_2 = pd.DataFrame(x_pt_2_norm, index=x_pt_2_norm.index, columns=columns)
        x_s_2 = pd.DataFrame(x_pt_2_stnd, index=x_pt_2_stnd.index, columns=columns)
        r1, p1 = pretraining_score(x_1, y_pt_1)
        rn1, pn1 = pretraining_score(x_n_1, y_pt_1)
        rs1, ps1 = pretraining_score(x_s_1, y_pt_1)
        r2, p2 = pretraining_score(x_2, y_pt_2)
        rn2, pn2 = pretraining_score(x_n_2, y_pt_2)
        rs2, ps2 = pretraining_score(x_s_2, y_pt_2)
        corr_rmse_1.append(round(r1, 5))
        corr_rmse_norm_1.append(round(rn1, 5))
        corr_rmse_stnd_1.append(round(rs1, 5))
        corr_rmse_2.append(round(r2, 5))
        corr_rmse_norm_2.append(round(rn2, 5))
        corr_rmse_stnd_2.append(round(rs2, 5))
    print("RMSE 1", corr_rmse_1)
    print("RMSE norm 1", corr_rmse_norm_1)
    print("RMSE stnd 1", corr_rmse_stnd_1)
    plt.plot(range(1, len(corr_list) + 1), corr_rmse_1, marker='o', c='blue', label="Pretrain RMSE 1")
    plt.plot(range(1, len(corr_list) + 1), corr_rmse_norm_1, marker='x', c='green', alpha=0.75, label="Pretrain norm RMSE 1")
    plt.plot(range(1, len(corr_list) + 1), corr_rmse_stnd_1, marker='o', c='red', alpha=0.25, label="Pretrain stnd RMSE 1")
    plt.legend(loc='upper right')
    plt.xticks(range(len(corr_list) + 1))
    plt.xlabel("Number of features")
    plt.ylabel("RMSE")
    plt.savefig('rmse_pt_1.png')
    plt.close()
    print("RMSE 2", corr_rmse_2)
    print("RMSE norm 2", corr_rmse_norm_2)
    print("RMSE stnd 2", corr_rmse_stnd_2)
    print("corr list length", len(corr_list))
    plt.plot(range(1, len(corr_list) + 1), corr_rmse_2, marker='o', c='blue', label="Pretrain RMSE 2")
    plt.plot(range(1, len(corr_list) + 1), corr_rmse_norm_2, marker='x', c='green', alpha=0.6, label="Pretrain norm RMSE 2")
    plt.plot(range(1, len(corr_list) + 1), corr_rmse_stnd_2, marker='^', c='red', alpha=0.2, label="Pretrain stnd RMSE 2")
    plt.legend(loc='upper right')
    plt.xticks(range(len(corr_list) + 1))
    plt.xlabel("Number of features")
    plt.ylabel("RMSE")
    plt.savefig('rmse_pt_2.png')
    plt.close()


def method_1_prime(x_train, x_test):
    x_train = x_train.dropna()
    x_test = x_test.dropna()
    y_train_1 = x_train['totalPrice']
    x_train_1 = x_train.drop(['totalPrice'], axis=1)
    y_test_1 = x_test['totalPrice']
    x_test_1 = x_test.drop(['totalPrice'], axis=1)
    return x_train_1, y_train_1, x_test_1, y_test_1


def method_2_prime(x_train, with_missing_tr, x_test, with_missing_test, continuous, discrete):
    for feature in with_missing_tr:
        if feature in continuous:
            x_train[feature] = x_train[feature].fillna(x_train[feature].mean())
        elif feature in discrete:
            x_train[feature] = x_train[feature].fillna(x_train[feature].mode()[0])
    y_train_2 = x_train['totalPrice']
    x_train_2 = x_train.drop(['totalPrice'], axis=1)
    for feature in with_missing_test:
        if feature in continuous:
            x_test[feature] = x_test[feature].fillna(x_train[feature].mean())
        elif feature in discrete:
            x_test[feature] = x_test[feature].fillna(x_train[feature].mode()[0])
    y_test_2 = x_test['totalPrice']
    x_test_2 = x_test.drop(['totalPrice'], axis=1)
    return x_train_2, y_train_2, x_test_2, y_test_2


def linear_regression(x_train, y_train, x_test, y_test):
    linear_reg = LinearRegression()
    linear_reg.fit(x_train, y_train)
    train_predict = linear_reg.predict(x_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
    test_predict = linear_reg.predict(x_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
    return train_rmse, test_rmse


def preprocessing(x_prime):
    x_train, y_train, x_val, y_val = data_splitting(x_prime, x_prime['totalPrice'], 0.2)
    print("D_train shape", np.shape(x_train), np.shape(y_train))
    print("D_val shape", np.shape(x_val), np.shape(y_val))
    print("Pre-training data set information:")
    data_info(x_train)
    with_missing_tr = missing_info(x_train, "Dtrain_missing")
    with_missing_val = missing_info(x_val, "Dval_missing")
    print("Features in training data with missing values: ", with_missing_tr)
    continuous = ["Lng", "Lat", "square", "ladderRatio", "communityAverage"]
    discrete = ["Cid", "tradeTime", "followers", "livingRoom", "drawingRoom", "kitchen", "bathRoom", "floor",
                "buildingType", "constructionTime", "renovationCondition", "buildingStructure", "elevator",
                "fiveYearsProperty", "subway", "district"]
    y_mean = [np.mean(x_train['totalPrice'])] * len(x_train["totalPrice"])
    print("y_train mean = ", y_mean[0])
    baseline_rmse = np.sqrt(mean_squared_error(x_train["totalPrice"], y_mean))
    print("Baseline_rmse of training data set = %.5f" % baseline_rmse)

    # look at D_train
    print("x_train top 5\n", x_train.head())
    x_train.hist(bins=50, figsize=(20, 15))
    plt.savefig('x_train.png')
    plt.close()

    sns.distplot(x_train["totalPrice"])
    plt.savefig('y_train.png')
    plt.close()

    # Missing values imputation
    x_tr_1, y_tr_1, x_val_1, y_val_1 = method_1_prime(x_train, x_val)
    x_tr_2, y_tr_2, x_val_2, y_val_2 = method_2_prime(x_train, with_missing_tr, x_val, with_missing_val, continuous, discrete)
    rmse_tr_1, rmse_val_1 = linear_regression(x_tr_1, y_tr_1, x_val_1, y_val_1)
    print("RMSE of training set with missing value imputation method 1 = %.5f" % rmse_tr_1)
    print("RMSE of validation set with missing value imputation method 1 = %.5f" % rmse_val_1)
    rmse_tr_2, rmse_val_2 = linear_regression(x_tr_2, y_tr_2, x_val_2, y_val_2)
    print("RMSE of training set with missing value imputation method 2 = %.5f" % rmse_tr_2)
    print("RMSE of validation set with missing value imputation method 2 = %.5f" % rmse_val_2)
    if rmse_val_1 < rmse_val_2:
        x_tr_pro = x_tr_1
        y_tr_pro = y_tr_1
        x_val_pro = x_val_1
        y_val_pro = y_val_1
        imputation = method_1_prime
        print("Missing value iputation method 1 performs better.")
    else:
        x_tr_pro = x_tr_2
        y_tr_pro = y_tr_2
        x_val_pro = x_val_2
        y_val_pro = y_val_2
        imputation = method_2_prime
        print("Missing value iputation method 2 performs better.")

    # Features scaling: normalization
    x_tr_1_norm, scaler_norm_1 = normalization(x_tr_1)
    x_val_1_norm = scaler_norm_1.transform(x_val_1)
    x_val_1_norm = pd.DataFrame(x_val_1_norm, index=x_val_1.index, columns=x_val_1.columns)
    print("check x_val_1_norm\n", x_val_1_norm.head())
    rmse_tr_norm_1, rmse_val_norm_1 = linear_regression(x_tr_1_norm, y_tr_1, x_val_1_norm, y_val_1)
    print("RMSE of training set with missing value imputation method 1 and normalization = %.5f" % rmse_tr_norm_1)
    print("RMSE of validation set with missing value imputation method 1 and normalization = %.5f" % rmse_val_norm_1)
    x_tr_2_norm, scaler_norm_2 = normalization(x_tr_2)
    x_val_2_norm = scaler_norm_2.transform(x_val_2)
    x_val_2_norm = pd.DataFrame(x_val_2_norm, index=x_val_2.index, columns=x_val_2.columns)
    rmse_tr_norm_2, rmse_val_norm_2 = linear_regression(x_tr_2_norm, y_tr_2, x_val_2_norm, y_val_2)
    print("RMSE of training set with missing value imputation method 2 and normalization = %.5f" % rmse_tr_norm_2)
    print("RMSE of validation set with missing value imputation method 2 and normalization = %.5f" % rmse_val_norm_2)

    x_tr_1_norm.hist(bins=50, figsize=(20, 15))
    plt.title('Normalization')
    plt.savefig('x_tr_1_norm.png')
    plt.close()

    # Features scaling: standardization
    x_tr_1_stnd, scaler_stnd_1 = standardization(x_tr_1)
    x_val_1_stnd = scaler_stnd_1.transform(x_val_1)
    x_val_1_stnd = pd.DataFrame(x_val_1_stnd, index=x_val_1.index, columns=x_val_1.columns)
    rmse_tr_stnd_1, rmse_val_stnd_1 = linear_regression(x_tr_1_stnd, y_tr_1, x_val_1_stnd, y_val_1)
    print("RMSE of training set with missing value imputation method 1 and standardization = %.5f" % rmse_tr_stnd_1)
    print("RMSE of validation set with missing value imputation method 1 and standardization = %.5f" % rmse_val_stnd_1)
    x_tr_2_stnd, scaler_stnd_2 = standardization(x_tr_2)
    x_val_2_stnd = scaler_stnd_2.transform(x_val_2)
    x_val_2_stnd = pd.DataFrame(x_val_2_stnd, index=x_val_2.index, columns=x_val_2.columns)
    rmse_tr_stnd_2, rmse_val_stnd_2 = linear_regression(x_tr_2_stnd, y_tr_2, x_val_2_stnd, y_val_2)
    print("RMSE of training set with missing value imputation method 2 and standardization = %.5f" % rmse_tr_stnd_2)
    print("RMSE of validation set with missing value imputation method 2 and standardization = %.5f" % rmse_val_stnd_2)

    x_tr_1_stnd.hist(bins=50, figsize=(20, 15))
    plt.title('Standardization')
    plt.savefig('x_tr_1_stnd.png')
    plt.close()

    # Feature correlation
    sorted_corr, corr_list = correlation(x_train)
    tr_rmse_1 = []
    tr_rmse_1n = []
    tr_rmse_1s = []
    val_rmse_1 = []
    val_rmse_1n = []
    val_rmse_1s = []
    tr_rmse_2 = []
    tr_rmse_2n = []
    tr_rmse_2s = []
    val_rmse_2 = []
    val_rmse_2n = []
    val_rmse_2s = []
    for i in range(1, len(corr_list) + 1):
        columns = corr_list[0:i]
        x_t_1 = pd.DataFrame(x_tr_1, index=x_tr_1.index, columns=columns)
        x_t_1n = pd.DataFrame(x_tr_1_norm, index=x_tr_1_norm.index, columns=columns)
        x_t_1s = pd.DataFrame(x_tr_1_stnd, index=x_tr_1_stnd.index, columns=columns)
        x_v_1 = pd.DataFrame(x_val_1, index=x_val_1.index, columns=columns)
        x_v_1n = pd.DataFrame(x_val_1_norm, index=x_val_1_norm.index, columns=columns)
        x_v_1s = pd.DataFrame(x_val_1_stnd, index=x_val_1_stnd.index, columns=columns)
        rt1, rv1 = linear_regression(x_t_1, y_tr_1, x_v_1, y_val_1)
        rt1n, rv1n = linear_regression(x_t_1n, y_tr_1, x_v_1n, y_val_1)
        rt1s, rv1s = linear_regression(x_t_1s, y_tr_1, x_v_1s, y_val_1)
        tr_rmse_1.append(round(rt1, 5))
        val_rmse_1.append(round(rv1, 5))
        tr_rmse_1n.append(round(rt1n, 5))
        val_rmse_1n.append(round(rv1n, 5))
        tr_rmse_1s.append(round(rt1s, 5))
        val_rmse_1s.append(round(rv1s, 5))

        x_t_2 = pd.DataFrame(x_tr_2, index=x_tr_2.index, columns=columns)
        x_t_2n = pd.DataFrame(x_tr_2_norm, index=x_tr_2_norm.index, columns=columns)
        x_t_2s = pd.DataFrame(x_tr_2_stnd, index=x_tr_2_stnd.index, columns=columns)
        x_v_2 = pd.DataFrame(x_val_2, index=x_val_2.index, columns=columns)
        x_v_2n = pd.DataFrame(x_val_2_norm, index=x_val_2_norm.index, columns=columns)
        x_v_2s = pd.DataFrame(x_val_2_stnd, index=x_val_2_stnd.index, columns=columns)
        rt2, rv2 = linear_regression(x_t_2, y_tr_2, x_v_2, y_val_2)
        rt2n, rv2n = linear_regression(x_t_2n, y_tr_2, x_v_2n, y_val_2)
        rt2s, rv2s = linear_regression(x_t_2s, y_tr_2, x_v_2s, y_val_2)
        tr_rmse_2.append(round(rt2, 5))
        val_rmse_2.append(round(rv2, 5))
        tr_rmse_2n.append(round(rt2n, 5))
        val_rmse_2n.append(round(rv2n, 5))
        tr_rmse_2s.append(round(rt2s, 5))
        val_rmse_2s.append(round(rv2s, 5))

    print("RMSE train 1", tr_rmse_1)
    print("RMSE train 1 normalization", tr_rmse_1n)
    print("RMSE train 1 standardization", tr_rmse_1s)
    print("RMSE validation 1", val_rmse_1)
    print("RMSE validation 1 normalization", val_rmse_1n)
    print("RMSE validation 1 standardization", val_rmse_1s)
    print("Lowest training RMSE 1", min(tr_rmse_1), "index", tr_rmse_1.index(min(tr_rmse_1)))
    print("Lowest validation RMSE 1", min(val_rmse_1), "index", val_rmse_1.index(min(val_rmse_1)))
    plt.plot(range(1, len(corr_list) + 1), tr_rmse_1, marker='o', c='blue', label="Trainining RMSE 1")
    plt.plot(range(1, len(corr_list) + 1), val_rmse_1, marker='o', c='red', label="Validation RMSE 1")
    plt.legend(loc='upper right')
    plt.xticks(range(len(corr_list) + 1))
    plt.xlabel("Number of features")
    plt.ylabel("RMSE")
    plt.savefig('rmse_tr_val_1.png')
    plt.close()

    print("RMSE train 2", tr_rmse_2)
    print("RMSE train 2 normalization", tr_rmse_2n)
    print("RMSE train 1 standardization", tr_rmse_2s)
    print("RMSE validation 2", val_rmse_2)
    print("RMSE validation 2 normalization", val_rmse_2n)
    print("RMSE validation 2 standardization", val_rmse_2s)
    print("corr list length", len(corr_list))
    print("Lowest training RMSE 2", min(tr_rmse_2), "index", tr_rmse_2.index(min(tr_rmse_2)))
    print("Lowest validation RMSE 2", min(val_rmse_2), "index", val_rmse_2.index(min(val_rmse_2)))
    plt.plot(range(1, len(corr_list) + 1), tr_rmse_2, marker='o', c='blue', label="Trainining RMSE 2")
    plt.plot(range(1, len(corr_list) + 1), val_rmse_2, marker='o', c='red', label="Validation RMSE 2")
    plt.legend(loc='upper right')
    plt.xticks(range(len(corr_list) + 1))
    plt.xlabel("Number of features")
    plt.ylabel("RMSE")
    plt.savefig('rmse_tr_val_2.png')
    plt.close()

    if val_rmse_2.index(min(val_rmse_2)) != 20:
        cols_keep = corr_list[0:val_rmse_2.index(min(val_rmse_2)) + 1]
        # x_tr_pro = pd.DataFrame(x_tr_pro, index=x_tr_pro.index, columns=cols_keep)
        # x_val_pro = pd.DataFrame(x_val_pro, index=x_val_pro.index, columns=cols_keep)
        print("After feature selection", cols_keep)

    return x_tr_pro, y_tr_pro, x_val_pro, y_val_pro, cols_keep, imputation
