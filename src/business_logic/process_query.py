import configparser
import logging

import joblib

from src.IO.get_data_from_yahoo import get_last_stock_price, get_all_tickers
from src.IO.storage_tools import create_bucket, get_model_from_bucket, upload_file_to_bucket, delete_model
from src.algo.dummy_model import Stock_model


def create_business_logic():
    data_fetcher = get_last_stock_price
    return BusinessLogic(Stock_model(data_fetcher))


class BusinessLogic:

    def __init__(self, model_creator):
        self._root_bucket = 'ango-hw-bucket'
        self._config = configparser.ConfigParser()
        self._config.read('application.conf')
        self._model_creator = model_creator
        self._create_bucket()


    def get_version(self):
        return self._config['DEFAULT']['version']

    def get_all_tickers(self):
        return get_all_tickers()

    def train_alltickers(self):
        tickers = get_all_tickers()
        for ticker in tickers:
            print("training ", ticker)
            if (ticker != "BF.B" and ticker != "BRK.B"):
                self.do_retrain(ticker)

    def get_bucket_name(self):
        return f'{self._root_bucket}_{self.get_version().replace(".", "")}'

    def _get_or_create_model(self, ticker):
        log = logging.getLogger()
        model_filename = self.get_model_filename_from_ticker(ticker)
        model = get_model_from_bucket(model_filename, self.get_bucket_name())
        if model is None:
            log.warning(f'training model for {ticker}')
            model = self._model_creator.fit(ticker)
            with open(model_filename, 'wb') as f:
                joblib.dump(model, f)
            upload_file_to_bucket(model_filename, self.get_bucket_name())
        return model

    def _delete_model(self, ticker):
        model_filename = self.get_model_filename_from_ticker(ticker)
        if get_model_from_bucket(model_filename, self.get_bucket_name()) is not None:
            delete_model(ticker, self.get_bucket_name())
        return

    def get_model_filename_from_ticker(self, ticker):
        return f'{ticker}.pkl'

    def _create_bucket(self):
        create_bucket(self.get_bucket_name())

    def do_predictions_for(self, ticker):
        if (ticker == "BF.B" or ticker == "BRK.B"):
            return "No Data"
        model = self._get_or_create_model(ticker)
        predictions = model.predict(ticker)
        return predictions

    def do_analyse_perf(self, ticker):
        model = self._get_or_create_model(ticker)
        perf = model.analyse_perf(ticker)
        return perf

    def do_retrain(self, ticker):
        #delete_model(ticker)
        model = self._get_or_create_model(ticker)
        #perf = model.analyse_perf(ticker)
        #return perf
        return
