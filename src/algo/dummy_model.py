import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression


def create_features(df_stock, nlags=10):
    df_resampled = df_stock.resample('1D').mean()
    df_resampled = df_resampled[df_resampled.index.to_series().apply(lambda x: x.weekday() not in [5, 6])]
    lags_col_names = []
    for i in range(nlags + 1):
        df_resampled['lags_' + str(i)] = df_resampled['close'].shift(i)
        lags_col_names.append('lags_' + str(i))
    df = df_resampled[lags_col_names]
    df = df.dropna(axis=0)
    return df


def create_X_Y(df_lags):
    X = df_lags.drop('lags_0', axis=1)
    Y = df_lags[['lags_0']]
    return X, Y


class Stock_model(BaseEstimator, TransformerMixin):

    def __init__(self, data_fetcher):
        self.log = logging.getLogger()
        self.lr = LinearRegression()
        self._data_fetcher = data_fetcher
        self.log.warning('here')

    def fit(self, X, Y=None):
        data = self._data_fetcher(X)
        df_features = create_features(data)
        df_features, Y = create_X_Y(df_features)
        self.lr.fit(df_features, Y)
        return self

    def predict(self, X, Y=None):
        data = self._data_fetcher(X, last=True)
        df_features = create_features(data)
        df_features, Y = create_X_Y(df_features)
        predictions = self.lr.predict(df_features)
        self.analyse(X)
        if (predictions.flatten()[-1] >= predictions.flatten()[-2]):
            return "BUY (today: %f, tomorrow : %f)" % (predictions.flatten()[-2], predictions.flatten()[-1])
        else:
            return "SELL (today: %f, tomorrow : %f)" % (predictions.flatten()[-2], predictions.flatten()[-1])

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
