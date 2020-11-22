from datetime import datetime, timedelta

from yahoo_fin import stock_info as si

import pandas as pd


def get_last_stock_price(ticker, last=False):
    #tickers = si.tickers_sp500()
    try:
        si.get_data(ticker)

    except Exception as e:
        return pd.DataFrame()

    if last:
        now = datetime.now()
        start_date = now - timedelta(days=60)
        return si.get_data(ticker, start_date)
    return si.get_data(ticker)

def get_all_tickers():
    tickers = si.tickers_sp500()
    return tickers