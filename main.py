from generate_prediction_df import *
from prediction_lab import *
import configMarketData as config
from binance.client import Client
from SendGetDataFrameS3 import *

## a function to create data, train and predict
def run_prediction(TimesFrames = ["1d", "3d"], listPairs = ["BTCUSDT"]):
    """
    function to fetch historical price data, process it, train a model and run a prediction 
    TimesFrames: list of time frames to blend. The lowest time frame will be the target time frame.
    listPairs: tick to predict for
    """
    PricePredAnalysis = generate_prediction_df(Transpose = 10, client = Client, TimeFrames = TimesFrames, listPairs = listPairs, trainTestSplit = 0.7)
    PricePredAnalysis.find_LowestTF()
    PricePredAnalysis.collect_price_and_volume()
    PricePredAnalysis.collect_SR_Trend_Volume_Tweet()
    PricePredAnalysis.price_feat_eng()
    PricePredAnalysis.find_tops_and_bottom()
    PricePredAnalysis.add_moving_average()
    PricePredAnalysis.add_RSI()
    PricePredAnalysis.add_chopiness()
    PricePredAnalysis.add_normalize_price_data()
    PricePredAnalysis.add_price_feat_eng()
    PricePredAnalysis.transpose_per_timeframe()
    PricePredAnalysis.merge_timeframes()
    PricePredAnalysis.addTargets()
    PricePredAnalysis.add_day_month_year()
    PricePredAnalysis.apply_normalize()
    PricePredAnalysis.export_merged_df_tocsv()
    PricePredAnalysis.train_and_predict_signal_regressor(HyperParamOpt = False)
    PricePredAnalysis.render_regressor()
    return

def run_all_predictions(listAllPairs = ["BTCUSDT", "ETHUSDT", "ETHBTC"]):
    for currency in listAllPairs:
        run_prediction(TimesFrames = ["1d", "3d"], listPairs = [currency])  
        run_prediction(TimesFrames = ["3d", "1w"], listPairs = [currency])  
    return  

# set random seed
np.random.RandomState(1)
random.seed(1)
# generate a client
try:
    Client = Client(config.keys["APIKey"], config.keys["SecretKey"])
except:
    Client = None
    pass



   

run_all_predictions()

