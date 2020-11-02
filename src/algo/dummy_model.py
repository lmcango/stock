import logging
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, Ridge, Lasso, LassoCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures



def create_features(df_stock, nlags=14, addTomorrow=False):
    df_resampled = df_stock.resample('1D').mean()
    df_resampled = df_resampled[df_resampled.index.to_series().apply(lambda x: x.weekday() not in [5, 6])]
    lags_col_names = []
    for i in range(nlags + 1):
        df_resampled['lags_' + str(i)] = df_resampled['close'].shift(i)
        lags_col_names.append('lags_' + str(i))
    df = df_resampled[lags_col_names]
    df = df.dropna(axis=0)

    if addTomorrow:
        df_last_row = df.tail(1)
        df_last_row = df_last_row.shift(periods=1, axis="columns")
        df = df.append(df_last_row)

    return df


def create_X_Y(df_lags):
    X = df_lags.drop('lags_0', axis=1)
    Y = df_lags[['lags_0']]
    return X, Y


class Stock_model(BaseEstimator, TransformerMixin):

    def __init__(self, data_fetcher):
        self.log = logging.getLogger()
        self.lr = LinearRegression()
        #tscv = TimeSeriesSplit(n_splits=100)
        #self.lr = LogisticRegressionCV(cv=tscv)
        #self.lr = Lasso(alpha=-1.0, max_iter=10000, tol=0.0001)
        #self.lr = LassoCV(cv=tscv)
        self._data_fetcher = data_fetcher
        self.log.warning('here')

    def fit(self, X, Y=None):

        data = self._data_fetcher(X)
        train_data = data.head(int(0.95 * data.shape[0]))
        test_data = data.tail(int(0.05 * data.shape[0])+1)

        train_features = create_features(train_data)
        train_features, train_Y = create_X_Y(train_features)

        # prepare a range of alpha values to test
        #alphas = np.array([10, 5, 1, 0.1, 0.01, 0.001, 0.0001, 0, -1, -5, -10])
        # fit a ridge regression model, testing each alpha
        #grid = GridSearchCV(estimator=self.lr, param_grid=dict(alpha=alphas))
        #grid.fit(train_features, train_Y)
        #print(grid)
        # summarize the results of the grid search
        #print(grid.best_score_)
        #print(grid.best_estimator_.alpha)

        #polynom = PolynomialFeatures(2)
        #train_features = train_features.to_numpy()
        #X_train = polynom.fit_transform(train_features)

        self.lr.fit(train_features, train_Y)
        #self.lr.fit(X_train, train_Y)

        test_features = create_features(test_data)
        test_features, test_Y = create_X_Y(test_features)

        #test_features = test_features.to_numpy()
        #X_test = polynom.fit_transform(test_features)

        predictions = self.lr.predict(test_features)
        #predictions = self.lr.predict(X_test)

        np_Y = test_Y.values
        TP = 0  # BUY
        P = 0
        TN = 0  # SELL
        N = 0
        nb_pred = 0
        BA = 'NaN'
        for i in range(predictions.size - 1):
            if (np_Y[i + 1] > np_Y[i] and predictions[i + 1] > np_Y[i]):
                TP = TP + 1
            elif (np_Y[i + 1] < np_Y[i] and predictions[i + 1] < np_Y[i]):
                TN = TN + 1

            if (np_Y[i + 1] > np_Y[i]):
                P = P + 1
            elif (np_Y[i + 1] < np_Y[i]):
                N = N + 1

        if (P + N == 0):
            print("NOT ENOUGH DATA\n")
        else:
            BA = (TP / P + TN / N) / 2
        print("***********")
        print("BA = ", BA)
        print("***********")

        return self

    def fitLogReg(self, X, Y=None):
        data = self._data_fetcher(X)
        train_data = data.head(int(0.80 * data.shape[0]))
        test_data = data.tail(int(0.20 * data.shape[0])+1)

        train_features = create_features(train_data)
        train_features["tmp"] = 0
        train_features.loc[train_features['lags_0'] > train_features['lags_1'], 'tmp'] = 1
        train_features.loc[train_features['lags_0'] <= train_features['lags_1'], 'tmp'] = 0
        train_features["lags_0"] = train_features["tmp"]
        del train_features["tmp"]
        train_features, train_Y = create_X_Y(train_features)

        self.lr.fit(train_features, train_Y)

        test_features = create_features(test_data)
        test_features["tmp"] = 0
        test_features.loc[test_features['lags_0'] > test_features['lags_1'], 'tmp'] = 1
        test_features.loc[test_features['lags_0'] <= test_features['lags_1'], 'tmp'] = 0
        test_features["lags_0"] = test_features["tmp"]
        del test_features["tmp"]
        test_features, test_Y = create_X_Y(test_features)

        predictions = self.lr.predict(test_features)

        np_Y = test_Y.values

        cm = confusion_matrix(np_Y, predictions)
        print(cm)
        BA = (cm[0][0] / (cm[0][0] + cm[1][0]) + cm[1][1] / (cm[0][1] + cm[1][1])) / 2

        print("BA = ", BA)

        return self

    def predict(self, X, Y=None):
        data = self._data_fetcher(X, last=True)
        df_features = create_features(data, addTomorrow=True)
        df_features, Y = create_X_Y(df_features)
        predictions = self.lr.predict(df_features)

        if (predictions.flatten()[-1] > Y.values[-2]):
            return "BUY (today: %f, tomorrow: %f)" % (Y.values[-2], predictions.flatten()[-1])
        elif (predictions.flatten()[-1] < Y.values[-2]):
            return "SELL (today: %f, tomorrow: %f)" % (Y.values[-2], predictions.flatten()[-1])
        else:
            return "NO ACTION (today: %f, tomorrow: %f)" % (Y.values[-2], predictions.flatten()[-1])


    def analyse_perf(self, ticker):
        data = self._data_fetcher(ticker, last=True)
        df_features = create_features(data)
        df_features, Y = create_X_Y(df_features)
        predictions = self.lr.predict(df_features)
        TP = 0  # BUY
        P = 0
        TN = 0  # SELL
        N = 0
        nb_pred = 0
        BA = 'NaN'
        np_Y = Y.values
        predictions = predictions.flatten()
        #print(np_Y)
        #print("\n")
        #print(predictions)
        for i in range(np_Y.size-1):
            if (np_Y[i + 1] > np_Y[i] and predictions[i + 1] > np_Y[i]):
                TP = TP + 1
            elif (np_Y[i + 1] < np_Y[i] and predictions[i + 1] < np_Y[i]):
                TN = TN + 1

            if (np_Y[i + 1] > np_Y[i]):
                P = P + 1
            elif (np_Y[i + 1] < np_Y[i]):
                N = N + 1

        if (P + N == 0):
            print("NOT ENOUGH DATA\n")
        else:
            BA = (TP / P + TN / N) / 2
        return "BA = %f" % (BA)
