import pandas as pd
import numpy as np
import time
from utils import TimeStampDiff
from datetime import timezone
import datetime
from SendGetDataFrameS3 import *

"""def update_price_df(TimeFrame,currency, saved_currency, saved_TimeFrame, client, n_data = 200):
        print("update_price_df start ...")
        if (TimeFrame != saved_TimeFrame) or (currency != saved_currency):

            saved_TimeFrame = TimeFrame
            saved_currency = currency   
            n_candle = n_data + 1
            #file_name = "/app/df_"+currency+"_"+TimeFrame+".csv"
            file_name = "./df_"+currency+"_"+TimeFrame+".csv"
            try:
                df_old = pd.read_csv(file_name)             
                if df_old.shape[0] > n_data:
                    timestamp = int(df_old.dateTime.values[-1])
                    a = client._get_earliest_valid_timestamp(currency, TimeFrame)
                else:
                    currenttime = int(round(time.time() * 1000))
                    timestamp = currenttime - TimeStampDiff(n_candle, TimeFrame) #TimestampMin[TimeFrame]
            except Exception:
                currenttime = int(round(time.time() * 1000))
                timestamp = currenttime - TimeStampDiff(n_candle, TimeFrame) #TimestampMin[TimeFrame]TimestampMin[TimeFrame]
                pass
            
            print("request ", currency, TimeFrame)
            print("get candle2")
            try: 
                candle = client.get_historical_klines(
                    currency, TimeFrame, timestamp, limit=10)
                print(len(candle))
                print("arrive candle2")    
                print("data received")
                df = pd.DataFrame(candle, columns=['dateTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime',
                                'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
                #df.dateTime = pd.to_datetime(df.dateTime, unit='ms')
                #df.closeTime = pd.to_datetime(df.closeTime, unit='ms')
                print(df.shape)
                #print(df.head())
                #df = df.reset_index()
                #file_name = "./df_"+currency+"_"+TimeFrame+"_"+a+".csv"
                #df.to_csv(file_name+"temp", index=False)
                #df = pd.read_csv(file_name+"temp")
                #print(df.head())
            except: #in case there is no connection
                df = pd.read_csv(file_name)
                pass

            # concatenate files
            try:
                print("concatenate old and new data") 
                print("New data")
                if df_old["dateTime"].iloc[-1] < df["dateTime"].iloc[-1] or df.shape[0] == 1:
                    df = pd.concat([df_old.iloc[:-1], df], ignore_index=True)
                elif df.shape[0] < n_candle:
                    df = df_old
                #df = df.iloc[-1000:]
                if df.shape[0] >= n_candle:
                    df.to_csv(file_name, index=False)
                print(file_name+" saved with "+str(df.shape[0])+" records.")
                print()
            except Exception:
                print("failed to concatenate")
                df.to_csv(file_name, index=False)
                print(file_name+" saved with "+str(df.shape[0])+" records.")
                pass

            # check dtype 
            df[['open', 'high', 'low', 'close', 'volume']] =  df[['open', 'high', 'low', 'close', 'volume']].astype("float")
        print("update_price_df done ...")
        return df,saved_currency, saved_TimeFrame"""

def update_price_df(TimeFrame,currency, saved_currency, saved_TimeFrame, client, n_data = 200, StartTime=None, EndTime=None):

        # fetch data currently saved in df
        file_name = "./df_"+currency+"_"+TimeFrame+".csv"
        if client is not None:
            print("update_price_df start ...")
            #time_now = datetime.timestamp(datetime.now())*1000
            StartTimeStamp = None
            EndTimeStamp = None
            currenttime = None
            if (TimeFrame != saved_TimeFrame) or (currency != saved_currency):
                saved_TimeFrame = TimeFrame
                saved_currency = currency   

                # define number of candle to download
                n_candle = n_data + 1
                try:
                    #df_old = pd.read_csv(file_name)
                    df_old = fetchDataframeFromS3(file_name)
                    # check there is at least enough data in teh saved file
                    if df_old.shape[0] > n_data:
                        # if enough do not update
                        StartTimeStamp = None #int(df_old.dateTime.values[-1])
                        # find the average time step
                        currenttime = int(round(time.time() * 1000))
                        dt = np.nanmedian(df_old["dateTime"].diff())
                        if int(currenttime - df_old["dateTime"].values[-1]) > dt:
                            StartTimeStamp = currenttime
                        #a = client._get_earliest_valid_StartTimeStamp(currency, TimeFrame)
                    else:
                        # otherwise define a new stating time stamp
                        currenttime = int(round(time.time() * 1000))
                        StartTimeStamp = currenttime - TimeStampDiff(n_candle, TimeFrame) #TimestampMin[TimeFrame]
                except Exception:
                    if currenttime is None:
                        print("a n_candle", n_candle)
                        print("a TimeFrame", TimeFrame)
                        print("a currenttime", currenttime)
                        currenttime = int(round(time.time() * 1000))
                        print("b currenttime", currenttime)
                        print("TimeStampDiff(n_candle, TimeFrame)", TimeStampDiff(n_candle, TimeFrame))
                        print("currenttime - TimeStampDiff(n_candle, TimeFrame)", currenttime - TimeStampDiff(n_candle, TimeFrame))
                    StartTimeStamp = currenttime - TimeStampDiff(n_candle, TimeFrame) #TimestampMin[TimeFrame]TimestampMin[TimeFrame]
                    print("a StartTimeStamp", StartTimeStamp)
                    pass

                # in case we have a specific start date
                if StartTime is not None:
                    try:
                        # load existing data
                        #df_old = pd.read_csv(file_name)
                        df_old = fetchDataframeFromS3(file_name)
                        # find the average time step
                        dt = np.nanmedian(df_old["dateTime"].diff())
                        # find the minimal StartTimeStamp require for this the starting time
                        minTime = int(StartTime - n_candle * dt)
                        if (int(df_old["dateTime"].values[0]) - minTime) < dt and df_old.shape[0] > n_data:
                            # if there is enough data and the starting point is included in it, no need to update dataset 
                            StartTimeStamp = None
                        else:
                            # other update the time stamp
                            StartTimeStamp = minTime
                    except:
                        pass

                # in case we have a specific end date
                if EndTime is not None:
                    try:
                        # load existing data
                        # df_old = pd.read_csv(file_name)
                        df_old = fetchDataframeFromS3(file_name)
                        # find the average time step
                        dt = np.nanmedian(df_old["dateTime"].diff())
                        # find the minimal timestamp require for this the end time
                        #maxTime = int(EndTime - n_candle * dt)
                        maxTime = EndTime
                        print(int(maxTime - df_old["dateTime"].values[-1]))
                        print("EndTime", pd.to_datetime(EndTime, unit='ms'))
                        print("df_old[dateTime].values[-1]", pd.to_datetime(df_old["dateTime"].values[-1], unit='ms'))
                        #if (int(maxTime - df_old["dateTime"].values[-1])) < dt and df_old.shape[0] > n_data:
                        if (int(maxTime - df_old["dateTime"].values[-1])) < dt and df_old.shape[0] > n_data:
                            # if there is enough data and the end point is included in it, no need to update dataset 
                            EndTimeStamp = None
                        else:
                            # other update the time stamp
                            EndTimeStamp = maxTime
                            if StartTime is None:
                                StartTime = df_old["dateTime"].values[-1] + dt
                            StartTimeStamp = StartTime
                    except:
                        pass
                    
                print("request ", currency, TimeFrame)
                print("get candle2")
                # if start and and time are defined
                if StartTimeStamp is not None and EndTimeStamp is None:
                    candle = client.get_historical_klines(currency, TimeFrame, start_str = int(StartTimeStamp))
                    print(len(candle))
                    print("data received")
                    df = pd.DataFrame(candle, columns=['dateTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime',
                                    'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
                # if only start time is defined
                elif StartTimeStamp is not None and EndTimeStamp is not None:
                    candle = client.get_historical_klines(currency, TimeFrame, start_str = int(StartTimeStamp), 
                                end_str = int(EndTimeStamp))
                    print(len(candle))   
                    print("data received")
                    df = pd.DataFrame(candle, columns=['dateTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime',
                                    'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
                # if no start time is define, use existing data-
# if only start time is defined
                elif StartTimeStamp is None and EndTimeStamp is not None:
                    candle = client.get_historical_klines(currency, TimeFrame, start_str = int(StartTimeStamp), end_str = int(EndTimeStamp))
                    print(len(candle))   
                    print("data received")
                    df = pd.DataFrame(candle, columns=['dateTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime',
                                    'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
                # if no start time is define, use existing data-
                else:
                    #df = pd.read_csv(file_name)
                    df = fetchDataframeFromS3(file_name)
                    #pass

                # concatenate files
                try:
                    print("concatenate old and new data") 
                    print("New data")
                    if df_old["dateTime"].iloc[-1] < df["dateTime"].iloc[-1] or df.shape[0] == 1:
                        df = pd.concat([df_old.iloc[:-1], df], ignore_index=True)
                    if df_old["dateTime"].iloc[0] > df["dateTime"].iloc[0]: # or df.shape[0] == 1:
                        df = pd.concat([df, df_old.iloc[0:]], ignore_index=True)
                    if df.shape[0] < n_candle and df.shape[0] >1:
                        df = df_old
                    #df = df.iloc[-1000:]
                    df = df.drop_duplicates(subset=['dateTime'])
                    if df.shape[0] >= n_data: #candle:
                        sendDataframeToS3(df, file_name)
                        #df.to_csv(file_name, index=False)
                    print(file_name+" saved with "+str(df.shape[0])+" records.")
                    print()
                except Exception:
                    print("failed to concatenate")
                    df = df.drop_duplicates(subset=['dateTime'])
                    sendDataframeToS3(df, file_name)
                    #df.to_csv(file_name, index=False)
                    print(file_name+" saved with "+str(df.shape[0])+" records.")
                    pass
                #df.drop_duplicates(subset=['dateTime'])
                # check dtype 
                df[['open', 'high', 'low', 'close', 'volume']] =  df[['open', 'high', 'low', 'close', 'volume']].astype("float")
        else:
            #df = pd.read_csv(file_name)
            df = fetchDataframeFromS3(file_name)

        df = df.drop_duplicates(subset=['dateTime'])
        print("update_price_df done ...")

        ##########################################
        # cut dataset to the given dates
        """if StartTime is not None:
            print(df['dateTime'].values[0])
            idx = np.abs(df['dateTime'].values-StartTime).argmin()
            df = df.iloc[max(idx - n_data, 0):]
        if EndTime is not None:
            idx = np.abs(df['dateTime'].values-EndTime).argmin()
            df = df.iloc[:idx+1]
        """
        
        return df,saved_currency, saved_TimeFrame



"""
from datetime import date, datetime
from binance.client import Client
#from update_price_df import update_price_df2
import configMarketData as config
from utils import TimeStampDiff

### Build data set
# define starting date and end date for training and validation
# start date
dt = datetime(year=2020, month=1, day=1)
StartDate = int(dt.timestamp())*1000
# end date
dt = datetime(year=2021, month=1, day=1)
EndDate = int(dt.timestamp())*1000
# define time frames
TimeFrames = [ "3d", "1d", "12h", "4h"]
# define currencies
ListPairs = ["BTCUSDT", "ETHUSDT", "AAVEUSDT", "ADAUSDT", "BNBUSDT", "COMPUSDT",
                 "COTIUSDT", "DEGOUSDT", "DOGEUSDT", "DOTUSDT", "AXSUSDT", "ENJUSDT", "FILUSDT", "MATICUSDT",
                 "SOLUSDT", "THETAUSDT", "VETUSDT", "XLMUSDT", "XRPUSDT", "FTMUSDT", "LINKUSDT", "KSMUSDT",
                 "LUNAUSDT", "FTMUSDT", "FTTUSDT", "AVAXUSDT", "RUNEUSDT", "FETUSDT", "SUPERUSDT", "ICPUSDT", 
                 "ROSEUSDT", "TFUELUSDT", "ARUSDT", "ALICEUSDT", "UNIUSDT", "SUSHIUSDT", "CRVUSDT", "GRTUSDT",
                 "ZILUSDT","ALGOUSDT", "ATOMUSDT", "BANDUSDT", "RENUSDT", "EGLDUSDT", "TLMUSDT", "SRMUSDT",
                 "SANDUSDT", "SHIBUSDT", "RAYUSDT", "MANAUSDT"]
# settings
n_data = 200
# binance client
client = Client(config.keys["APIKey"], config.keys["SecretKey"])
TimeFrame  = "1d"
Currency = "BTCUSDT"
df,saved_currency,saved_TimeFrame = update_price_df(TimeFrame,Currency, 0, "", client, n_data, time)
"""
"""
### collecte  data
# collect prices and volume
for Currency in ListPairs:
    for TimeFrame in TimeFrames:
        time = StartDate
        while time < EndDate:
            df,saved_currency,saved_TimeFrame = update_price_df3(TimeFrame,Currency, 0, "", client, n_data, time)
            print("time1", time)
            time += TimeStampDiff(1, TimeFrame)
            print("time2", time)
"""
