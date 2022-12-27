from datetime import datetime, timezone
from binance.client import Client
from update_price_df import update_price_df
import configMarketData as config
from utils import TimeStampDiff
import copy
from support import generate_analysis
import pandas as pd
from calculate_sum_volumes import calculate_sum_volumes
from found_trend import found_trend
from generate_message import generate_message
import tweepy
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random 
from SendGetDataFrameS3 import *
from moving_average import moving_average
from RSI import RSI
from chopiness import chopiness
import plotly.graph_objects as go

def sort_df(df, pair, TF, max_column):
    """
    Sort and organize dataframe with multiple timeframes and pairs
    """
    df_temp = pd.DataFrame(columns = df.columns.values, index=range(0))
    for i in range(df.shape[0]):
        df_temp2 = df[(df[pair] == df[pair].values[i]) & (df[TF] == df[TF].values[i])]
        df_temp2 = df_temp2.sort_values(max_column, ascending = False)
        df_temp3 = df_temp[(df_temp[pair] == df[pair].values[i]) & (df_temp[TF] == df[TF].values[i])]
        if df_temp3.shape[0] == 0:
            df_temp = pd.concat([df_temp, df_temp2.iloc[0:0]])
    return df_temp
class generate_prediction_df():
    def __init__(self, n_data = 200,
                         Transpose = 10, 
                         yearStart = None, 
                         monthStart = None, 
                         dayStart = None, 
                         yearEnd = None, 
                         monthEnd = None, 
                         dayEnd = None, 
                         trainTestSplit = 0.7,
                         client = None,
                         listPairs = ["BTCUSDT"],
                         TimeFrames = ["1d"]):
        self.INCREASING_COLOR = '#17BECF'
        self.DECREASING_COLOR = '#7F7F7F'
        self.n_data = n_data
        ### Build data set
        # define starting date and end date for training and validation
        # start date
        if dayStart is None:
            dt = datetime(year=2020, month=1, day=1, tzinfo=timezone.utc)
            self.StartDate = int(dt.timestamp())*1000
        else:
            dt = datetime(year=yearStart, month=monthStart, day=dayStart, tzinfo=timezone.utc)
            self.StartDate = int(dt.timestamp())*1000
        print("Start date:", datetime.utcfromtimestamp(self.StartDate/1000).strftime('%Y-%m-%d'))
        # end date
        if yearEnd is None:
            dt = datetime.utcnow()
        else:
            dt = datetime(year=yearEnd, month=monthEnd, day=dayEnd, tzinfo=timezone.utc)
        self.EndDate = int(dt.timestamp())*1000
        print("End date:", datetime.utcfromtimestamp(self.EndDate/1000).strftime('%Y-%m-%d'))
        # list target
        self.Targets = []
        self.TargetsTimeFrames = [ "1w", "3d", "1d", "12h", "4h"]
        self.TargetType = "absolute diff" #["absolute exact", "absolute diff", "relative diff", "relative exact"]
        # define time frames
        self.TimeFrames = TimeFrames #[ "3d", "1d", "12h", "4h"]
        # define currencies
        self.ListPairs = listPairs
        # binance client
        self.Client = client
        # timestamp feature
        self.Time = "dateTime"
        # list features
        self.Features = ['open', 'high', 'low', 'close', 'volume', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol']
        self.FeaturesToTranspose = ['open', 'high', 'low', 'close', 'volume', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol']
        # self feature prices
        self.FeatPrices = ['open', 'high', 'low', 'close']
        self.FeatDiffPrices = []
        # client
        try:
            self.Client = Client(config.keys["APIKey"], config.keys["SecretKey"])
        except:
            pass
        # number fo transpose
        self.Transpose = Transpose
        # df with data
        self.DictDfAnalysis = {}
        # root target
        self.RootTarget = "close"
        # lowestTF
        dt = 1e10
        self.LowestTF = None
        for TF in self.TimeFrames:
            if dt > TimeStampDiff(1, TF):
                dt = TimeStampDiff(1, TF)
                self.LowestTF = TF
        # data to normalize
        self.MeanTranspose = "MeanTranspose"
        self.StdDevTranspose = "StdDevTranspose"
        # model saving
        self.DictModel = {}
        # to be ignore in training
        self.ignore = [self.MeanTranspose+"_"+self.LowestTF, self.StdDevTranspose+"_"+self.LowestTF, self.Time]
        self.ignore += ["top_"+x for x in self.TimeFrames]
        self.ignore += ["bot_"+x for x in self.TimeFrames]
        # fraction of data going to test set
        self.trainTestSplit = trainTestSplit
        
    @property
    def TimeFrames(self):
        return self._TimeFrames

    @property
    def TargetsTimeFrames(self):
        return self._TargetsTimeFrames

    @TargetsTimeFrames.setter
    def TargetsTimeFrames(self, input):
        self._TargetsTimeFrames = input        

    @TimeFrames.setter
    def TimeFrames(self, input):
        # update timeframes
        self._TimeFrames = input
        # update lowest timeframe
        dt = 1e10
        self._LowestTF = None
        for TF in self._TimeFrames:
            if dt > TimeStampDiff(1, TF):
                dt = TimeStampDiff(1, TF)
                self._LowestTF = TF
        # update target time frame
        self._TargetsTimeFrames = list(set(self._TimeFrames) & set(self._TargetsTimeFrames))

    def find_LowestTF(self):
        dt = 1e10
        self.LowestTF = None
        for TF in self.TimeFrames:
            if dt > TimeStampDiff(1, TF):
                dt = TimeStampDiff(1, TF)
                self.LowestTF = TF
        
    def collect_price_and_volume(self):
        ### collecte  data
        # collect prices and volume
        for Currency in self.ListPairs:
            for TimeFrame in self.TimeFrames:
                df,_,_ = update_price_df(TimeFrame = TimeFrame,
                                        currency = Currency,
                                        saved_currency = 0, 
                                        saved_TimeFrame = "",
                                        client = self.Client,
                                        n_data = self.n_data,
                                        StartTime = self.StartDate,
                                        EndTime = self.EndDate)
                """
                plt.figure()
                plt.plot(df[self.Time], df[self.RootTarget])
                plt.show()
                input()
                """
    def find_tops_and_bottom(self):
        for Currency in self.ListPairs:
            for TimeFrame in self.TimeFrames:
                df_temp = copy.deepcopy(self.DictDfAnalysis[Currency][TimeFrame])
                df_temp["dateTextFormat"] = pd.to_datetime(df_temp.dateTime, unit='ms')
                list_ressup, list_res, list_sup, list_hz, list_top, list_bot = generate_analysis(df_temp, nbr_days = df_temp.shape[0]-1)
                color_top = [3]*len(list_top)
                color_bot = [10]*len(list_bot)
                # plot support
                price_chart_figure = make_subplots(specs=[[{"secondary_y": True}]])
                price_chart_figure.add_trace(go.Candlestick(x=pd.to_datetime(df_temp.dateTime, unit='ms'),
                open=df_temp.open, high=df_temp.high,
                low=df_temp.low, close=df_temp.close,
                name=Currency,
                increasing=dict(line=dict(color=self.INCREASING_COLOR)),
                decreasing=dict(line=dict(color=self.DECREASING_COLOR))),
                secondary_y=True)
                price_chart_figure.update(layout_xaxis_rangeslider_visible=False)
                price_chart_figure.add_trace(go.Scatter(x=pd.to_datetime(df_temp.dateTime, unit='ms').iloc[list_top], 
                                        y=df_temp["high"].iloc[list_top],
                                        name='top',mode='markers',
                                        marker=dict(color=color_top,size=10)), secondary_y = True)
                price_chart_figure.add_trace(go.Scatter(x=pd.to_datetime(df_temp.dateTime, unit='ms').iloc[list_bot], 
                                        y=df_temp["low"].iloc[list_bot], 
                                        name='low',mode='markers', 
                                        marker=dict(color=color_bot,size=10)), secondary_y = True)
                price_chart_figure.show()

                # add top and bottom to dataset
                df_temp["top"] = 0
                df_temp["top"].iloc[list_top] = 1
                df_temp["bot"] = 0
                df_temp["bot"].iloc[list_bot] = 1
                self.Targets.append("top")
                self.Targets.append("bot")

                # save back dataset
                df_temp = df_temp.sort_index(axis=1)
                self.DictDfAnalysis[Currency][TimeFrame] = df_temp
        return 

    def update_df_with_analysis(self, df, DictAnalysis, i_index):
        for key in DictAnalysis.keys():
            if key not in df.columns.values:
                df[key] = 0
                self.Features.append(key)
            if DictAnalysis[key] is not None:
                df[key].values[i_index] = DictAnalysis[key]
            else:
                df[key].values[i_index] = 1
        return df

    def merge_timeframes(self):
        # concatenate dataframe
        # create list of DataFrame and reindex
        for Currency in self.DictDfAnalysis.keys():
            # find smallest time frame
            dt = 1e10
            lowest_TF = self.LowestTF
            df_ref = copy.deepcopy(self.DictDfAnalysis[Currency][lowest_TF]) #.set_index("dateTime")

            # add closeTime
            list_feat = list(set([self.Time]+self.Features+self.Targets+[self.MeanTranspose, self.StdDevTranspose]) & set(df_ref.columns.values.tolist()))
            list_feat.sort()
            df_ref = df_ref[list_feat]

            def update_columns_name(df_ref, TF, ref_time, lowest_TF, CloseTime):
                list_temp = []
                list_price = []
                for x in df_ref.columns.values:
                    if x == CloseTime:
                        list_temp.append(x)
                    else:
                        name = x +"_"+ TF
                        list_temp.append(name)
                        if x in self.FeatPrices:
                            list_price.append(name)
                        if x in self.FeatDiffPrices:
                            self.FeatDiffPrices.append(name)
                        if x in self.FeatPrices:
                            self.FeatPrices.append(name)

                df_ref.columns = np.array(list_temp)
                return df_ref, list_temp, list_price
                
            df_ref["CloseTime"] = df_ref[self.Time].values + TimeStampDiff(1,lowest_TF)
            df_ref, list_temp, list_price = update_columns_name(df_ref, lowest_TF, self.Time, lowest_TF, "CloseTime")
            self.DictDfAnalysis[Currency]["Features_merged"] = copy.deepcopy(list_temp)
            self.DictDfAnalysis[Currency]["FeatPrices"] = copy.deepcopy(list_price)

            for TimeFrame in self.DictDfAnalysis[Currency].keys():
                if TimeFrame != lowest_TF and TimeFrame in self.TimeFrames:
                    df_temp = copy.deepcopy(self.DictDfAnalysis[Currency][TimeFrame])
                    list_feat = list(set([self.Time]+self.Features+self.Targets+[self.MeanTranspose, self.StdDevTranspose]) & set(df_temp.columns.values.tolist()))
                    list_feat.sort()
                    df_temp = df_temp[list_feat]
                    df_temp["CloseTime"] = df_temp[self.Time].values + TimeStampDiff(1, TimeFrame)
                    df_temp = df_temp.drop(columns=[self.Time])
                    df_temp, list_temp, list_price  = update_columns_name(df_temp, TimeFrame, self.Time, lowest_TF, "CloseTime")
                    self.DictDfAnalysis[Currency]["Features_merged"].extend(copy.deepcopy(list_temp))
                    self.DictDfAnalysis[Currency]["FeatPrices"].extend(copy.deepcopy(list_price))

                    df_ref = pd.merge(df_ref, df_temp, on = "CloseTime", how = "left")
                    df_ref = df_ref.drop_duplicates(subset="CloseTime")
                    df_ref = df_ref.loc[:,~df_ref.columns.duplicated()]

            # remove tail data that were not in the training time frame
            df_ref = df_ref.rename(columns={self.Time+"_"+lowest_TF:self.Time})
            self.DictDfAnalysis[Currency]["Features_merged"] = [self.Time if x == self.Time+"_"+lowest_TF else x for x in self.DictDfAnalysis[Currency]["Features_merged"]]
            self.DictDfAnalysis[Currency]["Features_merged"].remove("CloseTime") 
            i_start = np.abs(df_ref[self.Time].values - self.StartDate).argmin()
            i_end = np.abs(df_ref[self.Time].values - self.EndDate).argmin()
            df_ref = df_ref.iloc[i_start:i_end+1]
            # remove rows with empty values
            df_ref = df_ref.fillna(method="ffill")
            df_ref = df_ref.dropna(how='any')  
            self.DictDfAnalysis[Currency]["merged"] = copy.deepcopy(df_ref)
            sendDataframeToS3(df_ref, "merged_df.csv")
        return self.DictDfAnalysis

    def transpose_per_timeframe(self):
        list_feat_temp = copy.deepcopy(self.FeaturesToTranspose)
        for Currency in self.ListPairs:
            for TimeFrame in self.TimeFrames:
                df_temp = copy.deepcopy(self.DictDfAnalysis[Currency][TimeFrame])
                for i in range(1,self.Transpose+1):
                    for feat in list_feat_temp:
                        if feat in df_temp.columns.values:
                            name = feat+"_"+str(i)
                            if feat in self.FeatPrices and name not in self.FeatPrices:
                                self.FeatPrices.append(name)
                            df_temp[name] = np.nan
                            df_temp[name].values[i:] = df_temp[feat].values[:-i]
                            if name not in self.Features:
                                self.Features.append(name)
                self.DictDfAnalysis[Currency][TimeFrame] = copy.deepcopy(df_temp)
                
    def add_normalize_price_data(self):
        for Currency in self.ListPairs:
            for TimeFrame in self.TimeFrames:
                df_temp = copy.deepcopy(self.DictDfAnalysis[Currency][TimeFrame])
                if self.MeanTranspose not in df_temp.columns.values:
                    df_temp[self.MeanTranspose] = np.nan
                if self.StdDevTranspose not in df_temp.columns.values:
                    df_temp[self.StdDevTranspose] = np.nan
                for i in range(self.Transpose, df_temp.shape[0]):
                    df_temp[self.MeanTranspose].values[i] = np.mean(df_temp["close"].values[i-self.Transpose:i+1])
                    if self.Transpose < 1:
                        a = 1.0
                    else:
                        a = max(df_temp["high"].values[i-self.Transpose:i+1]) - min(df_temp["low"].values[i-self.Transpose:i+1])
                    df_temp[self.StdDevTranspose].values[i] = copy.deepcopy(a)
                self.DictDfAnalysis[Currency][TimeFrame] = copy.deepcopy(df_temp)
        if self.MeanTranspose not in self.Features:
            self.Features.append(self.MeanTranspose)
        if self.StdDevTranspose not in self.Features:
            self.Features.append(self.StdDevTranspose)        

    def add_price_feat_eng(self):
        for Currency in self.ListPairs:
            for TimeFrame in self.TimeFrames:
                df_temp = copy.deepcopy(self.DictDfAnalysis[Currency][TimeFrame])
                for feat in self.FeaturesToTranspose:
                    if feat in df_temp.columns.values:
                        new_feat = feat+"_diff"
                        self.FeatDiffPrices.append(new_feat)
                        self.Features.append(new_feat)
                    df_temp[feat] = df_temp[feat].astype(float)
                    df_temp[new_feat] = df_temp[feat].diff()
                self.DictDfAnalysis[Currency][TimeFrame] = copy.deepcopy(df_temp)

    def collect_SR_Trend_Volume_Tweet(self):
        # update database of tweets
        StartDate = self.StartDate
        EndDate = self.EndDate
        period = 1*24*60*60
        if StartDate is None:
            StartDate = datetime.datetime.utcnow().timestamp() - period
        elif StartDate > 1e11:
            StartDate = int(StartDate / 1000)
        if EndDate is None:
            EndDate =   datetime.datetime.utcnow().timestamp()
        elif EndDate > 1e11:
            EndDate = int(EndDate / 1000)
        # collect prices and volume
        for Currency in self.ListPairs:
            for TimeFrame in self.TimeFrames:
                df,_,_ = update_price_df(TimeFrame = TimeFrame,
                                        currency = Currency,
                                        saved_currency = 0, #saved_currency,
                                        saved_TimeFrame = "", #saved_TimeFrame, 
                                        client = self.Client,
                                        n_data = self.n_data,
                                        StartTime = self.StartDate,
                                        EndTime = self.EndDate)
                # define range of time step to analyse
                IndexStart = self.n_data
                IndexEnd = int(min(df.shape[0], np.abs(df[self.Time].values - self.EndDate).argmin()))
                df_price_analysis = copy.deepcopy(df)
                # loop on time step 
                for i in range(IndexStart, IndexEnd+1):
                    df_stats = pd.DataFrame(columns=["token","value","timeFrame"],index=range(0))
                    df_temp = copy.deepcopy(df.iloc[0:i+1])
                    list_ressup, list_res, list_sup, list_hz, list_top, list_bot = generate_analysis(df_temp)
                    x_dw, y_dw, x_up, y_up, y_div_g, y_div_r = calculate_sum_volumes(df_temp, list_top, list_bot)   
                    df_trend = found_trend(df_temp, list_top, list_bot, TimeFrame)             
                    saved_message1, DictAnalysis = generate_message(df_temp, list_ressup, list_res, list_sup, list_hz, TimeFrame, Currency,
                                                    y_div_g, y_div_r, list_top, list_bot, df_trend, df_stats, self.n_data)
                    df_price_analysis = self.update_df_with_analysis(df_price_analysis, DictAnalysis, i)
                sendDataframeToS3(df_price_analysis, "df_price_analysis.csv")
                if Currency not in self.DictDfAnalysis.keys():
                    self.DictDfAnalysis.update({Currency: {TimeFrame: df_price_analysis, "FeatPrices": []}})
                else: 
                    self.DictDfAnalysis[Currency].update({TimeFrame: df_price_analysis, "FeatPrices": []})

    def price_feat_eng(self):

        for Currency in self.ListPairs:
            for TimeFrame in self.TimeFrames:
                df_temp = copy.deepcopy(self.DictDfAnalysis[Currency][TimeFrame])
                
                # add body size
                df_temp["body"] = df_temp["close"] - df_temp["open"]
                self.FeatDiffPrices.append("body")
                self.Features.append("body")
                self.FeaturesToTranspose.append("body")

                # add body size to total size
                df_temp["bodyRel"] = np.divide(df_temp["close"] - df_temp["open"],((df_temp["high"] - df_temp["low"])))
                #self.FeatDiffPrices.append("bodyRel")
                self.Features.append("bodyRel")
                self.FeaturesToTranspose.append("bodyRel")

                # add low wick size
                df_temp["lowWick"] = np.minimum(df_temp["close"], df_temp["open"]) - df_temp["low"]
                self.FeatDiffPrices.append("lowWick")
                self.Features.append("lowWick")
                self.FeaturesToTranspose.append("lowWick")

                # add relative low wick sizeto total size
                df_temp["lowWickRel"] = np.divide(np.minimum(df_temp["close"], df_temp["open"]) - df_temp["low"], df_temp["high"] - df_temp["low"])
                #self.FeatDiffPrices.append("highWickRel")
                self.Features.append("lowWickRel")
                self.FeaturesToTranspose.append("lowWickRel")

                # add high wick size
                df_temp["highWick"] = df_temp["high"] - np.maximum(df_temp["close"], df_temp["open"])
                self.FeatDiffPrices.append("highWick")
                self.Features.append("highWick")
                self.FeaturesToTranspose.append("highWick")

                # add relative high wick sizeto total size
                df_temp["highWickRel"] = np.divide((df_temp["high"] - np.maximum(df_temp["close"], df_temp["open"])), (df_temp["high"] - df_temp["low"]))
                #self.FeatDiffPrices.append("highWickRel")
                self.Features.append("highWickRel")
                self.FeaturesToTranspose.append("highWickRel")

                # save it back
                self.DictDfAnalysis[Currency][TimeFrame] = copy.deepcopy(df_temp)

    def add_moving_average(self):
        self = moving_average(self)

    def add_RSI(self):
        self = RSI(self)

    def add_chopiness(self):
        self = chopiness(self)     

    def addTargets(self):
        ### to add targets to the merged dataset
        # loop on currency
        for Currency in self.ListPairs:
            df_temp = copy.deepcopy(self.DictDfAnalysis[Currency]["merged"])
            # loop on target time frames
            for TF in self.TargetsTimeFrames:
                name = copy.deepcopy("target"+self.RootTarget+"_"+TF)
                # create new columns
                if name not in self.Targets:
                    self.Targets.append(name)
                    
                    df_temp[name] = np.nan
                    b = np.zeros(df_temp.shape[0])
                    b[:] = np.nan
                    dt = TimeStampDiff(1, TF)
                    # loop on all time steps
                    for i in range(df_temp.shape[0]):
                        idx = np.abs(df_temp[self.Time].values - (df_temp[self.Time].values[i]+dt)).argmin()
                        if np.abs(df_temp[self.Time].values[i] + dt - df_temp[self.Time].values[idx]) < 1:
                            a = float(copy.deepcopy(df_temp[self.RootTarget+"_"+TF].values[idx] - df_temp[self.RootTarget+"_"+self.LowestTF].values[i]))
                            b[i] = a
                    df_temp[name] = b
                    self.FeatDiffPrices.append(name)
            
            # remove lines with nans
            df_temp = df_temp.fillna(method="ffill")
            # forward 
            # save df in dataframe
            self.DictDfAnalysis[Currency]["merged"] = copy.deepcopy(df_temp)

        ## add target classification ##
        for Currency in self.ListPairs:
            df_temp = copy.deepcopy(self.DictDfAnalysis[Currency]["merged"])
            # loop on target time frames
            for TF in self.TargetsTimeFrames:
                name_regr = copy.deepcopy("target"+self.RootTarget+"_"+TF)
                name = copy.deepcopy("target"+self.RootTarget+"_UpDown_"+TF)
                # create new columns
                if name not in self.Targets:
                    df_temp[name] = np.sign(df_temp[name_regr]) 
                    self.Targets.append(name)

            # remove lines with nans
            df_temp = df_temp.fillna(method="ffill")
            # save df in dataframe
            self.DictDfAnalysis[Currency]["merged"] = copy.deepcopy(df_temp)

        # add target for tops 
        for Currency in self.ListPairs:
            df_temp = copy.deepcopy(self.DictDfAnalysis[Currency]["merged"])
            # loop on target time frames
            for TF in self.TargetsTimeFrames:   
                name_regr = copy.deepcopy("top"+"_"+TF)
                name = copy.deepcopy("target"+"top"+"_"+TF)
                # create new columns
                print("df_temp.columns", df_temp.columns)
                for i in range(len(df_temp.columns.values)):
                    print(df_temp.columns.values[i])
                if name not in self.Targets:
                    df_temp[name] = df_temp[name_regr]
                    self.Targets.append(name)  
            self.DictDfAnalysis[Currency]["merged"] = copy.deepcopy(df_temp)

        # add target for bottoms 
        for Currency in self.ListPairs:
            df_temp = copy.deepcopy(self.DictDfAnalysis[Currency]["merged"])
            # loop on target time frames
            for TF in self.TargetsTimeFrames:   
                name_regr = copy.deepcopy("bot"+"_"+TF)
                name = copy.deepcopy("target"+"bot"+"_"+TF)
                # create new columns
                print("df_temp.columns", df_temp.columns)
                for i in range(len(df_temp.columns.values)):
                    print(df_temp.columns.values[i])
                if name not in self.Targets:
                    df_temp[name] = df_temp[name_regr]
                    self.Targets.append(name)  
            self.DictDfAnalysis[Currency]["merged"] = copy.deepcopy(df_temp)
        return

    def export_merged_df_tocsv(self):
        # loop on currency
        for Currency in self.ListPairs:
            df_temp = copy.deepcopy(self.DictDfAnalysis[Currency]["merged"])
            sendDataframeToS3(df_temp, "df_merged_"+Currency+".csv")
        return

    def add_day_month_year(self):
        for Currency in self.ListPairs:
            df_temp = copy.deepcopy(self.DictDfAnalysis[Currency]["merged"])
            time = df_temp[self.Time].values
            time = [datetime.utcfromtimestamp(x/1000) for x in time]
            df_temp["day"] = [x.day for x in time]
            df_temp["month"] = [x.month for x in time]
            df_temp["year"] = [x.year for x in time]
            df_temp["weekday"] = [x.weekday() for x in time]
            self.Features.extend(["day", "month", "year", "weekday"])
            self.DictDfAnalysis[Currency]["merged"] = copy.deepcopy(df_temp)
 
    def split_train_test(self, TargetName):

        iTarget = self.Targets.index(TargetName)

        for Currency in self.ListPairs:
            target = TargetName 
            df_temp = copy.deepcopy(self.DictDfAnalysis[Currency]["merged"])
            idx = int(df_temp.shape[0]*self.trainTestSplit)
            self.DictDfAnalysis[Currency]["train"] = copy.deepcopy(df_temp.iloc[:idx])
            self.DictDfAnalysis[Currency]["test"] = copy.deepcopy(df_temp.iloc[idx:])
            features = self.DictDfAnalysis[Currency]["Features_merged"]
            xtrain = self.DictDfAnalysis[Currency]["train"][features]
            ytrain = self.DictDfAnalysis[Currency]["train"][target]
            a  = self.DictDfAnalysis[Currency]["train"][features+[target]]
            a.to_csv(target+".csv")
            xtest = self.DictDfAnalysis[Currency]["test"][features]
            ytest = self.DictDfAnalysis[Currency]["test"][target]
        return xtrain, ytrain, xtest, ytest

    def apply_normalize(self):
        lowest_TF = self.LowestTF
        for Currency in self.ListPairs:
            df_temp = copy.deepcopy(self.DictDfAnalysis[Currency]["merged"])
            for feat in df_temp.columns.values:
                if feat in self.FeatPrices:
                    df_temp[feat] = (df_temp[feat] - df_temp[self.MeanTranspose + "_" + lowest_TF])/df_temp[self.StdDevTranspose + "_" + lowest_TF]
                if feat in self.FeatDiffPrices and feat not in self.FeatPrices:
                    df_temp[feat] = (df_temp[feat])/df_temp[self.StdDevTranspose + "_" + lowest_TF]
            self.DictDfAnalysis[Currency]["merged"] = copy.deepcopy(df_temp)

    def train(self, TargetName, InferenceEngine, TF, seed = [1], HyperParamOpt = True):
        name = TargetName+"_"+TF+"_"+InferenceEngine

        # get training data
        xtrain, ytrain, xtest, ytest = self.split_train_test(TargetName)

        # rebalance training data for classification 
        print("TargetName:", TargetName)
        print("xtrain:", xtrain.columns)
        print(ytrain.value_counts())
        print(TargetName in xtrain.columns.values)
        
        if "top" in TargetName:
            print("ytrain.value_counts() -1-", ytrain.value_counts())
            from sklearn.utils import resample
            df_train_temp = copy.deepcopy(xtrain)
            df_train_temp[TargetName] = ytrain
            df_0 = df_train_temp[df_train_temp[TargetName]==0]
            df_1 = df_train_temp[df_train_temp[TargetName]==1]

            # Upsample minority class
            df_1_upsampled = resample(df_1, 
                                            replace=True,     # sample with replacement
                                            n_samples=df_0.shape[0],    # to match majority class
                                            random_state=42) # reproducible results

            # Combine majority class with upsampled minority class
            df_upsampled = pd.concat([df_0, df_1_upsampled]) #, df_minus1_upsampled])
            xtrain = df_upsampled[xtrain.columns.values]
            ytrain = df_upsampled[TargetName]

            # Display new class counts
            print("ytrain.value_counts() -2-", ytrain.value_counts())
            print("df_upsampled.label.value_counts()", df_upsampled[TargetName].value_counts())
            
        if "bot" in TargetName:
            from sklearn.utils import resample
            df_train_temp = copy.deepcopy(xtrain)
            df_train_temp[TargetName] = ytrain
            df_0 = df_train_temp[df_train_temp[TargetName]==0]
            df_1 = df_train_temp[df_train_temp[TargetName]==1]

            # Upsample minority class
            df_1_upsampled = resample(df_1, 
                                            replace=True,     # sample with replacement
                                            n_samples=df_0.shape[0],    # to match majority class
                                            random_state=42) # reproducible results

            # Combine majority class with upsampled minority class
            df_upsampled = pd.concat([df_0, df_1_upsampled]) #, df_minus1_upsampled])
            xtrain = df_upsampled[xtrain.columns.values]
            ytrain = df_upsampled[TargetName]

            # Display new class counts
            print("ytrain.value_counts() -2-", ytrain.value_counts())
            print("df_upsampled.label.value_counts()", df_upsampled[TargetName].value_counts())

        #for Currency in self.ListPairs:
        """xtrain = self.DictDfAnalysis[Currency]["train"][TargetName]
        ytrain = self.DictDfAnalysis[Currency]["train"][TargetName]
        xtest = self.DictDfAnalysis[Currency]["test"][TargetName]
        ytest = self.DictDfAnalysis[Currency]["test"][TargetName]"""

        # remove features to be ignore for training
        FeatList = [x for x in xtrain.columns.values if x not in self.ignore]
        #model.fit(xtrain.drop(self.ignore, axis = 1),ytrain)
        print("Duplicated columns", xtrain[FeatList].columns.duplicated())
        print(xtrain[FeatList].columns)

        if InferenceEngine == "GBR":
            model = GradientBoostingRegressor(random_state=1)
            #model = XGBRegressor(random_state=1)
        elif InferenceEngine == "XGBR":
            """
            gscv = GridSearchCV(
            estimator=XGBRegressor(random_state=1),
            param_grid={"learning_rate": (0.05, 0.10, 0.15),
                        "max_depth": [ 3, 4, 5, 6, 8],
                        "min_child_weight": [ 1, 3, 5, 7],
                        "gamma":[ 0.0, 0.1, 0.2],
                        'subsample': [0.6, 0.8, 1.0],  
                        "colsample_bytree":[ 0.3, 0.4],},
            cv=3, scoring='r2', verbose=0, n_jobs=-1)
            """
            if HyperParamOpt:
                tscv = TimeSeriesSplit(n_splits=3)
                gscv = RandomizedSearchCV(
                estimator=XGBRegressor(random_state=1),
                param_distributions={
                            #"booster" : ["gbtree", "gblinear", "dart"],
                            "n_estimators": range(50, 400, 50),
                            "learning_rate": [0.01, 0.05, 0.1, 0.2],
                            "max_depth": [ 3, 4, 5, 6, 8],
                            "min_child_weight": [ 1, 3, 5, 7],
                            "gamma":[ 0.0, 0.25, 1],
                            "reg_lambda": [0, 1, 10],
                            'subsample': [0.6, 0.8, 1.0],  
                            "colsample_bytree":[ 0.3, 0.4, 0.5],},
                cv=tscv, scoring='r2', verbose=0, n_jobs=1, n_iter=100, random_state=1)
                gscv.fit(xtrain[FeatList],ytrain)
                model = gscv.best_estimator_
                print(model)
                print("gscv.best_params_", gscv.best_params_)
                
            else:
                """options = {"n_estimators": 200,
                            "learning_rate": 0.1,
                            "max_depth": 5,
                            "min_child_weight": 3,
                            "gamma":0.25,
                            "reg_lambda": 1,
                            'subsample': 0.8,  
                            "colsample_bytree":0.3}"""
                options = {'subsample': 1.0, 'reg_lambda': 10, 'n_estimators': 100, 'min_child_weight': 3, 'max_depth': 5, 'learning_rate': 0.05, 'gamma': 0.0, 'colsample_bytree': 0.3}
                model=XGBRegressor(**options , random_state=1)

        elif InferenceEngine == "OLS":
            model = LinearRegression()
        elif InferenceEngine == "GBC":
            model = GradientBoostingRegressor(random_state=1)
            #model = XGBRegressor(random_state=1)
        elif InferenceEngine == "XGBC":
            """gscv = GridSearchCV(
            estimator=XGBClassifier(random_state=1),
            param_grid={"learning_rate": (0.05, 0.10, 0.15),
                        "max_depth": [ 3, 4, 5, 6, 8],
                        "min_child_weight": [ 1, 3, 5, 7],
                        "gamma":[ 0.0, 0.1, 0.2],
                        'subsample': [0.6, 0.8, 1.0],  
                        "colsample_bytree":[ 0.3, 0.4],},
            cv=3, scoring='r2', verbose=0, n_jobs=-1)"""
            tscv = TimeSeriesSplit(n_splits=3)
            gscv = RandomizedSearchCV(
            estimator=XGBClassifier(random_state=1),
            param_distributions={
                        #"booster" : ["gbtree", "gblinear", "dart"],
                        "learning_rate": [0.01, 0.05, 0.1, 0.2],
                        "max_depth": [ 3, 4, 5, 6, 8],
                        "min_child_weight": [ 1, 3, 5, 7],
                        "gamma":[ 0.0, 0.25, 1],
                        "reg_lambda": [0, 1, 10],
                        'subsample': [0.6, 0.8, 1.0],  
                        "colsample_bytree":[ 0.3, 0.4, 0.5],},
            cv=tscv, scoring='f1', verbose=0, n_jobs=-1, n_iter = 100, random_state=1)
            
            gscv.fit(xtrain[FeatList],ytrain)
            model = gscv.best_estimator_
            print("gscv.best_params_", gscv.best_params_)

        print(xtrain.isna().sum(), ytrain.isna().sum())
        model.fit(xtrain[FeatList],ytrain)
        self.DictModel.update({name: {InferenceEngine: model, "xtrain": xtrain, "ytrain": ytrain, "xtest": xtest, "ytest": ytest, "FeatList": FeatList}})

    def predict(self, TargetName, InferenceEngine, TF):
        name = TargetName+"_"+TF+"_"+InferenceEngine
        model = self.DictModel[name][InferenceEngine]
        xtest = self.DictModel[name]["xtest"]
        ytest = self.DictModel[name]["ytest"]
        xtrain = self.DictModel[name]["xtrain"]
        ytrain = self.DictModel[name]["ytrain"]
        FeatList = self.DictModel[name]["FeatList"]

        ytest_predict = model.predict(xtest[FeatList])
        ytrain_predict = model.predict(xtrain[FeatList])
        self.DictModel[name]["ytest_predict"] = ytest_predict
        self.DictModel[name]["ytrain_predict"] = ytrain_predict

        df = pd.DataFrame(columns = ["name", "value"], index = range(len(model.feature_importances_)))
        df["name"] = xtrain[FeatList].columns.values
        df["value"] = model.feature_importances_
        df = df.sort_values("value", ascending = False)
        fig = go.Figure(go.Bar(y=df["name"].values[0:30],
                x=df["value"].values[0:30],
                orientation='h'))
        fig.update_layout(title = TF+" "+TargetName+" "+InferenceEngine)
        fig.show()        

    def plot_prediction(self, TargetName, InferenceEngine, TF):
        name = TargetName+"_"+TF+"_"+InferenceEngine
        model = self.DictModel[name][InferenceEngine]
        xtest = self.DictModel[name]["xtest"]
        ytest = self.DictModel[name]["ytest"]
        xtrain = self.DictModel[name]["xtrain"]
        ytrain = self.DictModel[name]["ytrain"]
        FeatList = self.DictModel[name]["FeatList"]
        ytest_predict = self.DictModel[name]["ytest_predict"]
        ytrain_predict = self.DictModel[name]["ytrain_predict"]
        
        print("InferenceEngine")
        print("R2 ", r2_score(ytest[:-1], ytest_predict[:-1]))
        print("R2 lag = ", r2_score(ytest[:-1], ytest[1:]))

        xtest[self.Time+"_days"] = pd.to_datetime(xtest[self.Time], unit='ms')
        xtest[name+"_measured"] = xtest[self.RootTarget+"_"+TF]*xtest[self.StdDevTranspose+ "_" + self.LowestTF] + xtest[self.MeanTranspose+ "_" + self.LowestTF]
        xtest["ytest_predict"] = ytest_predict
        a = xtest["ytest_predict"]*xtest[self.StdDevTranspose+ "_" + self.LowestTF]
        xtest["predict2"] = xtest["close_1_"+TF].values[0]*xtest[self.StdDevTranspose+ "_" + self.LowestTF] + xtest[self.MeanTranspose+ "_" + self.LowestTF]
        xtest["predict2"] = xtest["predict2"] + a.cumsum()
        xtest["predict1"] = xtest["close_1_"+TF].values*xtest[self.StdDevTranspose+ "_" + self.LowestTF] + xtest[self.MeanTranspose+ "_" + self.LowestTF]
        xtest["predict1"] = xtest["predict1"] + a

        ### for regressor
        if InferenceEngine in ["XGBR", "GBR", "OLS"]:
            
            fig = px.line(xtest, x=self.Time+"_days", y="ytest_predict")
            fig.add_trace(go.Scatter(mode="lines", x=xtest[self.Time+"_days"], y=xtest[self.RootTarget+"_"+TF], name=self.RootTarget+"_"+TF)) 
            fig.update_layout(title = TF+" "+TargetName+" "+InferenceEngine) 
            fig.show()
            
            fig = px.line(xtest, x=self.Time+"_days", y=name+"_measured")
            fig.add_trace(go.Scatter(mode="lines", x=xtest[self.Time+"_days"], y=xtest["predict2"], name="predict2")) 
            fig.add_trace(go.Scatter(mode="lines", x=xtest[self.Time+"_days"], y=xtest["predict1"], name="predict1"))
            fig.update_layout(title = TF+" "+TargetName+" "+InferenceEngine) 
            fig.show() 

            ### test 
            xtest["ytest_predict"] = ytest_predict
            print(self.StdDevTranspose +"_"+self.LowestTF)
            a = xtest["ytest_predict"]*xtest[self.StdDevTranspose+ "_" + self.LowestTF]

            print("a", a)
            print("a.cumsum()",a.cumsum())
            print("xtest[close_1_+TF].values[0]", xtest["close_1_"+TF].values[0])

            xtest["predict2"] = xtest["close_1_"+TF].values[0]*xtest[self.StdDevTranspose+ "_" + self.LowestTF] + xtest[self.MeanTranspose+ "_" + self.LowestTF]
            xtest["predict2"] = xtest["predict2"] + a.cumsum()
            xtest["predict1"] = xtest["close_1_"+TF].values*xtest[self.StdDevTranspose+ "_" + self.LowestTF] + xtest[self.MeanTranspose+ "_" + self.LowestTF]
            xtest["predict1"] = xtest["predict1"] + a

            xmeasured_test = xtest["close_"+TF]*xtest[self.StdDevTranspose+ "_" + self.LowestTF] + xtest[self.MeanTranspose+ "_" + self.LowestTF]

            ### train

            xtrain["ytrain_predict"] = ytrain_predict
            print(self.StdDevTranspose +"_"+self.LowestTF)
            a = xtrain["ytrain_predict"]*xtrain[self.StdDevTranspose+ "_" + self.LowestTF]

            print("a", a)
            print("a.cumsum()",a.cumsum())
            print("xtrain[close_1_+TF].values[0]", xtrain["close_1_"+TF].values[0])

            xtrain["predict2"] = xtrain["close_1_"+TF].values[0]*xtrain[self.StdDevTranspose+ "_" + self.LowestTF] + xtrain[self.MeanTranspose+ "_" + self.LowestTF]
            xtrain["predict2"] = xtrain["predict2"] + a.cumsum()
            xtrain["predict1"] = xtrain["close_1_"+TF].values*xtrain[self.StdDevTranspose+ "_" + self.LowestTF] + xtrain[self.MeanTranspose+ "_" + self.LowestTF]
            xtrain["predict1"] = xtrain["predict1"] + a

            xmeasured_train = xtrain["close_"+TF]*xtrain[self.StdDevTranspose+ "_" + self.LowestTF] + xtrain[self.MeanTranspose+ "_" + self.LowestTF]

            df_temp1 = pd.DataFrame(columns = ["measured", "predict"], index = range(len(ytest_predict)))
            df_temp1["measured"] = ytest.values #xtest[self.RootTarget+"_"+TF]
            df_temp1["predict"] = ytest_predict
            print(df_temp1)
            fig = px.box(df_temp1, x="measured", y="predict", points="all")
            fig.show()

        elif InferenceEngine in ["XGBC", "GBC"]:

            cm = confusion_matrix(ytest.values, ytest_predict)
            cm_dict = {'tn': cm[0, 0], 'fp': cm[0, 1], 'fn': cm[1, 0], 'tp': cm[1, 1]}
            print("cm_dict",cm_dict)
            print("accuracy_score:", accuracy_score(ytest.values, ytest_predict))

            # plot tops
            if "top" in TargetName:
                list_top = np.where(ytest_predict == 1)[0]
                list_bot = []
            elif "bot" in TargetName:
                list_bot = np.where(ytest_predict == 1)[0]
                list_top = []                
            color_top = ["#EC5130"]*len(list_top)
            color_bot = ["#64EC30"]*len(list_bot)
            # plot support
            price_chart_figure = make_subplots(specs=[[{"secondary_y": True}]])
            price_chart_figure.add_trace(go.Candlestick(x=pd.to_datetime(xtest.dateTime, unit='ms'),
                        open=xtest["open"+"_"+TF]*xtest[self.StdDevTranspose+ "_" + self.LowestTF] + xtest[self.MeanTranspose+ "_" + self.LowestTF],
                        high=xtest["high"+"_"+TF]*xtest[self.StdDevTranspose+ "_" + self.LowestTF] + xtest[self.MeanTranspose+ "_" + self.LowestTF],
                        low=xtest["low"+"_"+TF]*xtest[self.StdDevTranspose+ "_" + self.LowestTF] + xtest[self.MeanTranspose+ "_" + self.LowestTF],
                        close=xtest["close"+"_"+TF]*xtest[self.StdDevTranspose+ "_" + self.LowestTF] + xtest[self.MeanTranspose+ "_" + self.LowestTF],
            increasing=dict(line=dict(color=self.INCREASING_COLOR)),
            decreasing=dict(line=dict(color=self.DECREASING_COLOR))),
            secondary_y=True)
            price_chart_figure.update(layout_xaxis_rangeslider_visible=False)
            price_chart_figure.add_trace(go.Scatter(x=pd.to_datetime(xtest.dateTime, unit='ms').iloc[list_top], 
                                    y=xtest["high"+"_"+TF].iloc[list_top]*xtest[self.StdDevTranspose+ "_" + self.LowestTF].iloc[list_top] + xtest[self.MeanTranspose+ "_" + self.LowestTF].iloc[list_top],
                                    name='top',mode='markers',
                                    marker=dict(color=color_top,size=10)), secondary_y = True)
            price_chart_figure.add_trace(go.Scatter(x=pd.to_datetime(xtest.dateTime, unit='ms').iloc[list_bot], 
                                    y=xtest["low"+"_"+TF].iloc[list_bot]*xtest[self.StdDevTranspose+ "_" + self.LowestTF].iloc[list_bot] + xtest[self.MeanTranspose+ "_" + self.LowestTF].iloc[list_bot],
                                    name='low',mode='markers', 
                                    marker=dict(color=color_bot,size=10)), secondary_y = True)
            price_chart_figure.update_layout(title = TF+" "+TargetName+" "+InferenceEngine)
            price_chart_figure.show()

    def train_and_predict_signal_regressor(self, HyperParamOpt = True):

        # initialize
        InferenceEngine = "XGBR"
        TargetNameTop = "target"+"top"
        TargetNameBot = "target"+"bot"
        TF = self.LowestTF
        """quantile = 0.95"""

        # train for tops 
        self.train(TargetNameTop+"_"+TF, InferenceEngine, TF, np.arange(10), HyperParamOpt = HyperParamOpt)
        self.predict(TargetNameTop+"_"+TF, InferenceEngine, TF)
        # train for bots
        self.train(TargetNameBot+"_"+TF, InferenceEngine, TF ,np.arange(10), HyperParamOpt = HyperParamOpt)
        self.predict(TargetNameBot+"_"+TF, InferenceEngine, TF)

        return

    def render_regressor(self, quantile = 0.95):
        # initialize
        InferenceEngine = "XGBR"
        TargetNameTop = "target"+"top"
        TargetNameBot = "target"+"bot"
        TF = self.LowestTF

        # plot candles with regression
        # data top
        name_top = TargetNameTop+"_"+TF+"_"+TF+"_"+InferenceEngine
        xtest_top = self.DictModel[name_top]["xtest"]
        ytest_top = self.DictModel[name_top]["ytest"]
        xtrain_top = self.DictModel[name_top]["xtrain"]
        ytrain_top = self.DictModel[name_top]["ytrain"]
        ytest_predict_top = self.DictModel[name_top]["ytest_predict"]
        ytrain_predict_top = self.DictModel[name_top]["ytrain_predict"]
        list_top = np.where(ytest_predict_top > np.quantile(ytest_predict_top, quantile))[0]
        # data bot
        name_bot = TargetNameBot+"_"+TF+"_"+TF+"_"+InferenceEngine
        xtest_bot = self.DictModel[name_bot]["xtest"]
        ytest_bot = self.DictModel[name_bot]["ytest"]
        xtrain_bot = self.DictModel[name_bot]["xtrain"]
        ytrain_bot = self.DictModel[name_bot]["ytrain"]
        ytest_predict_bot = self.DictModel[name_bot]["ytest_predict"]
        ytrain_predict_bot = self.DictModel[name_bot]["ytrain_predict"]
        list_bot = np.where(ytest_predict_bot > np.quantile(ytest_predict_bot, quantile))[0]
                        
        # organise plot                  
        # plot condle sticks
        price_chart_figure = make_subplots()
        time = pd.to_datetime(xtest_bot.dateTime, unit='ms')
        open = xtest_bot["open"+"_"+TF]*xtest_bot[self.StdDevTranspose+ "_" + self.LowestTF] + xtest_bot[self.MeanTranspose+ "_" + self.LowestTF]
        high = xtest_bot["high"+"_"+TF]*xtest_bot[self.StdDevTranspose+ "_" + self.LowestTF] + xtest_bot[self.MeanTranspose+ "_" + self.LowestTF]
        low = xtest_bot["low"+"_"+TF]*xtest_bot[self.StdDevTranspose+ "_" + self.LowestTF] + xtest_bot[self.MeanTranspose+ "_" + self.LowestTF]
        close = xtest_bot["close"+"_"+TF]*xtest_bot[self.StdDevTranspose+ "_" + self.LowestTF] + xtest_bot[self.MeanTranspose+ "_" + self.LowestTF]
        price_chart_figure.add_trace(go.Candlestick(x=time,
                    open=open,
                    high=high,
                    low=low,
                    close=close,
                    increasing=dict(line=dict(color=self.INCREASING_COLOR)),
                    decreasing=dict(line=dict(color=self.DECREASING_COLOR))))
        # plot tops and bottom
        # define color map
        cmax = max(0.5,max(ytest_predict_top))
        cmin = min(-0.5,min(-ytest_predict_bot))
        c0 = -cmin/(cmax-cmin)
        c1 = max(0,(-0.5-cmin)/(cmax-cmin))
        c2 = min(1,(0.5-cmin)/(cmax-cmin))
        
        colorscale1 = [(0,"green"), (c1,"green"), (c0,"white"), (c2, "red"), (1, "red")] 

        price_chart_figure.update(layout_xaxis_rangeslider_visible=False)
        price_chart_figure.update_layout(title = TF+" "+InferenceEngine) 

        # tops 
        timeTops = pd.to_datetime(xtest_top.dateTime.iloc[list_top], unit='ms')
        priceTops = xtest_top["high"+"_"+TF].iloc[list_top]*xtest_top[self.StdDevTranspose+ "_" + self.LowestTF].iloc[list_top] + \
                                        xtest_top[self.MeanTranspose+ "_" + self.LowestTF].iloc[list_top]
        colorTops = np.take(ytest_predict_top, list_top)

        price_chart_figure.add_trace(go.Scatter(x=timeTops, 
                                y=priceTops,
                                name='top test',
                                mode='markers',
                                text=colorTops,
                                marker=dict(color=colorTops, size=10, showscale=True, coloraxis ="coloraxis",
                                colorscale = colorscale1)))

        timeBots = pd.to_datetime(xtest_top.dateTime.iloc[list_bot], unit='ms')
        priceBots = xtest_top["low"+"_"+TF].iloc[list_bot]*xtest_top[self.StdDevTranspose+ "_" + self.LowestTF].iloc[list_bot] + \
                                        xtest_top[self.MeanTranspose+ "_" + self.LowestTF].iloc[list_bot]
        colorBots = -np.take(ytest_predict_bot, list_bot)

        price_chart_figure.add_trace(go.Scatter(x=timeBots, 
                                y=priceBots,
                                name='bot test',
                                mode='markers',
                                text=colorBots,
                                marker=dict(color=colorBots, size=10, showscale=True, coloraxis ="coloraxis",
                                colorscale = colorscale1))) 

        price_chart_figure.layout.paper_bgcolor='rgba(0,0,0,0)'
        price_chart_figure.layout.plot_bgcolor='rgba(0,0,0,0)'
        price_chart_figure.layout.colorscale.diverging = colorscale1
        price_chart_figure.update_coloraxes(cmin=cmin)
        price_chart_figure.update_coloraxes(cmax=cmax)
        price_chart_figure.show()

        # boxplot
        df_temp1 = pd.DataFrame(columns = ["measured", "predict"], index = range(len(ytest_predict_top)))
        df_temp1["measured"] = ytest_top.values 
        df_temp1["predict"] = ytest_predict_top
        fig = px.box(df_temp1, x="measured", y="predict", points="all", title = "TOP test")
        fig.show()

        df_temp1 = pd.DataFrame(columns = ["measured", "predict"], index = range(len(ytest_predict_bot)))
        df_temp1["measured"] = ytest_bot.values 
        df_temp1["predict"] = ytest_predict_bot
        fig = px.box(df_temp1, x="measured", y="predict", points="all", title = "BOT test")
        fig.show()

        # cross plot
        df_temp0 = pd.DataFrame(columns = ["measured", "predict", "name"], index = range(len(ytrain_predict_top)))
        df_temp0["measured"] = ytrain_top.values 
        df_temp0["predict"] = ytrain_predict_top
        df_temp0["name"] = "R2 test top:"+str(r2_score(ytrain_top.values, ytrain_predict_top))
        fig = px.scatter(df_temp0, x="measured", y="predict", color = "name")
        df_temp1 = pd.DataFrame(columns = ["measured", "predict", "name"], index = range(len(ytest_predict_bot)))
        df_temp1["measured"] = -ytest_bot.values 
        df_temp1["predict"] = -ytest_predict_bot
        df_temp1["name"] = "R2 test bot:"+str(r2_score(-ytest_bot.values, -ytest_predict_bot))
        fig2 = px.scatter(df_temp1, x="measured", y="predict", color = "name")
        fig.add_trace(fig2.data[0])
        fig.show()

        #export data
        xtop  = xtest_top.dateTime.iloc[list_top]
        ytop = np.take(ytest_predict_top, list_top)
        xbot  = xtest_bot.dateTime.iloc[list_bot]
        ybot = np.take(ytest_predict_bot, list_bot)   

        # generate dataframe 
        df_price = pd.DataFrame(data={"date": time, "low": low, "high": high, "open": open, "close": close})
        df_tops = pd.DataFrame(data={"date":timeTops ,"priceTops":priceTops, "colorTops":colorTops})
        df_bots = pd.DataFrame(data={"date":timeBots ,"priceBots":priceBots, "colorBots":colorBots})
        df_price = df_price.merge(df_tops, how='left', on='date')
        df_price = df_price.merge(df_bots, how='left', on='date')

        #
        df_price["utcNow"] = datetime.now(timezone.utc).strftime("%Y%m%d")
        filename = "df_BlindTestAndSignal_" + self.ListPairs[0] + "_" + TF + ".csv"
        sendDataframeToS3(df_price, filename, location="s3")
        df_price.to_csv(filename)

        return {"xtop" : xtop, "ytop": ytop, "xbot" : xbot, "ybot": ybot, "xtest": xtest_top}

# initialize color for plots
INCREASING_COLOR = '#17BECF'
DECREASING_COLOR = '#7F7F7F'
# unitlalize random state
np.random.RandomState(1)
random.seed(1)
# fetch data from S3 storage
try:
    saved_message = fetchDataframeFromS3("df_message.csv")
except:
    saved_message = pd.DataFrame(columns={"TimeFrame","currency","message","timestamp","score"}, index=range(0))
    pass
# sort the data from storage
saved_message = sort_df(saved_message, "currency", "TimeFrame", "timestamp")
#sendDataframeToS3(saved_message, "df_message.csv")