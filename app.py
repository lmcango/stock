from flask import Flask

from src.business_logic.process_query import create_business_logic

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    return f'Hello I think you should use an other route:!\nEX: get_stock_val/<ticker>\n'


@app.route('/get_pred/<ticker>', methods=['GET'])
def get_stock_value(ticker):
    bl = create_business_logic()
    prediction = bl.do_predictions_for(ticker)

    return f'{prediction}\n'

@app.route('/train_all', methods=['GET'])
def train_all():
    bl = create_business_logic()
    tickers = bl.get_all_tickers()
    bl.train_alltickers()
    #prediction = bl.do_predictions_for(ticker)

    return f'training completed\n'

@app.route('/perf/<ticker>', methods=['GET'])
def perf(ticker):
    bl = create_business_logic()
    perf = bl.do_analyse_perf(ticker)

    return f'{perf}\n'

@app.route('/retrain/<ticker>', methods=['GET'])
def retrain(ticker):
    bl = create_business_logic()
    perf = bl.do_retrain(ticker)

    return f'{ticker} has been retrained: perf = {perf}\n'

if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=8080, debug=True)
