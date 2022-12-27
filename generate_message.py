import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from datetime import datetime, timezone
from utils import TimeStampDiff
import copy

def generate_message(df, list_ressup, list_res, list_sup, list_hz, TimeFrame, currency, y_div_g, y_div_r, list_top, list_bot, df_trend, df_stats, n_data =200):

    # initialize dict
    DictAnalysis = {}

    # iniialize
    #n_data = 200
    tol = 3e-2
    deltay = max(df["high"].values[-n_data:]) - min(df["low"].values[-n_data:])
    current_price = df["close"].values[-1]
    df_message = pd.DataFrame(columns=["TimeFrame", "currency", "message","score", "score_orig"], index=range(1))
    #saved_message = pd.DataFrame(columns={"TimeFrame","currency","message"}, index=range(1))#.to_json(date_format='iso', orient='split')
    df_message["TimeFrame"] = TimeFrame
    df_message["currency"] = currency
    df_message["message"] = "No warning"
    df_message["timestamp"] = datetime.timestamp(datetime.now())*1000
    df_message["score"] = 0
    df_message["score_orig"] = 0
    list_all = []
    for x in [list_ressup, list_res, list_sup, list_hz]:
        list_all.extend(x)

    closest_dist = 1e20
    message_closest = {}

    message_closest = {"closest" : {"TimeFrame": TimeFrame, "currency": currency, "message": ""}}

    print("messages for support and resistance.")
    for x in list_all:
                
        if abs(x["rel_dist"]) < closest_dist:
            closest_dist = abs(x["rel_dist"])
            score = 0
            # find values of lines at all dates
                
            #line_values = x["line_y"]
            # above a resistance
            if x["rel_dist"] < 0:
                # breaking out of a support (open below, close above)
                if df["open"].values[-1] > x["current_line_value"] and df["close"].values[-1] < x["current_line_value"]:
                #if df["open"].values[-1] > line_values[-1]:
                    #message_closest["closest"].update({"message":"breaking out support at "+str(x["current_line_value"])+". "})
                    message_closest["closest"]["message"] += "breaking out support at "+str(x["current_line_value"])+". "
                    score += -1
                    # update dict_analysis
                    DictAnalysis.update({"BreakingSupport":x["current_line_value"]})
                    
                # broke out of support (open below and close higher, previous closed above and opend below)
                if df["open"].values[-1] < x["current_line_value"] and \
                   df["open"].values[-2] > x["current_line_value"] and \
                   df["close"].values[-2] < x["current_line_value"]:
                    #message_closest["closest"].update({"message":"broke out support at "+str(x["current_line_value"])+". "})
                    message_closest["closest"]["message"] += "broke out support at "+str(x["current_line_value"])+". "
                    score += -1.5
                    # update dict_analysis
                    DictAnalysis.update({"BrokeSupport":x["current_line_value"]})

                # bounced on resistance (open and close above, previous was opening and closing above)
                if df["open"].values[-1] < x["current_line_value"] and \
                   df["close"].values[-1] < df["open"].values[-1] and \
                   df["open"].values[-2] < x["current_line_value"] and \
                   df["close"].values[-2] < x["current_line_value"] and \
                   df["close"].values[-2] > df["open"].values[-2] and \
                   abs((df["high"].values[-1] -x["current_line_value"])/deltay) < tol :
                    #message_closest["closest"].update({"message":"bounced on resistance at "+str(x["current_line_value"])+". "})
                    message_closest["closest"]["message"] += "bounced on resistance at "+str(x["current_line_value"])+". "
                    score += -1
                    # update dict_analysis
                    DictAnalysis.update({"BouncedResistance":x["current_line_value"]})
            # below a support
            elif x["rel_dist"] > 0:
                # breaking out of a resistance (open below, close above)
                if df["open"].values[-1] < x["current_line_value"] and df["close"].values[-1] > x["current_line_value"]:
                    message_closest["closest"]["message"] += "breaking out resistance at "+str(x["current_line_value"])+". "
                    #message_closest["closest"].update({"message":"breaking out resistance at "+str(x["current_line_value"])+". "})
                    score += 1  
                    # update dict_analysis
                    DictAnalysis.update({"BreakingResitance":x["current_line_value"]})
                # broke out of resistance (open below and close higher, previous closed above and opend below)
                if df["open"].values[-1] > x["current_line_value"] and \
                   df["open"].values[-2] < x["current_line_value"] and \
                   df["close"].values[-2] > x["current_line_value"]:
                    #message_closest["closest"].update({"message":"broke out resistance "+str(x["current_line_value"])+". "})
                    message_closest["closest"]["message"] += "broke out resistance "+str(x["current_line_value"])+". "
                    score += 1
                    # update dict_analysis
                    DictAnalysis.update({"BrokeResistance":x["current_line_value"]})
                # bounced on support (open and close above, previous was opening and closing above)
                if df["open"].values[-1] > x["current_line_value"] and \
                   df["close"].values[-1] > df["open"].values[-1] and \
                   df["open"].values[-2] > x["current_line_value"] and \
                   df["close"].values[-2] > x["current_line_value"] and \
                   df["close"].values[-2] < df["open"].values[-2] and \
                   abs((df["low"].values[-1] -x["current_line_value"])/deltay) < tol :
                    #message_closest["closest"].update({"message":"bounced on support at "+str(x["current_line_value"]+". ")})
                    message_closest["closest"]["message"]+= "bounced on support at "+str(x["current_line_value"])+". "
                    score += 1.5
                    # update dict_analysis
                    DictAnalysis.update({"BouncedSupport":x["current_line_value"]})

    # add message from the volume indicator
    print("Support for volumes.")
    text = None
    for i in range(2):
        if text is None:
            if np.isnan(y_div_g[-1-i]) == False:
                text = "Buying volume overtakes selling volume. "
                if "message" in message_closest["closest"].keys():
                    message_closest["closest"]["message"] = message_closest["closest"]["message"] + text
                else:
                    message_closest["closest"].update({"message":text})
                score += 0.5
                # update dict_analysis
                DictAnalysis.update({"BuyVolumeTops":None})
            elif np.isnan(y_div_r[-1-i]) == False:
                text = "Selling volume overtakes buying volume. "
                if "message" in message_closest["closest"].keys():
                    message_closest["closest"]["message"] = message_closest["closest"]["message"] + text
                else:
                    message_closest["closest"].update({"message":text})
                score -= 0.5
                # update dict_analysis
                DictAnalysis.update({"SellVolumeTops":None})

    # add message from the trend indicator 
    # check if the current timestamp has a trend
    timestamp_trend = df_trend["timeFrame"].values[-1]
    #print("timestamp_trend:", datetime.utcfromtimestamp(timestamp_trend/1000).strftime('%Y-%m-%d'))
    end_timestamp = pd.Timestamp(df_trend["end"].values[-1])
    end_timestamp = pd.to_datetime(end_timestamp).timestamp() * 1000
    print("end_timestamp:", datetime.utcfromtimestamp(end_timestamp/1000).strftime('%Y-%m-%d'))
    #print(datetime.utcnow().timestamp()*1000 - TimeStampDiff(1,timestamp_trend))
    print("end_timestamp",end_timestamp)
    #print(datetime.utcnow().timestamp()*1000 - TimeStampDiff(1,timestamp_trend) - end_timestamp)
    """if datetime.utcnow().timestamp()*1000 - TimeStampDiff(1,timestamp_trend) < end_timestamp:
        text = "Trend indicates "+df_trend["trend"].values[-1]+". "
        if "message" in message_closest["closest"].keys():
            message_closest["closest"]["message"] = message_closest["closest"]["message"] + text
        else:
             message_closest["closest"].update({"message":text})
        # update the score with trend
        trend_score = {"bull": +1, "bear": -1,"bullish reversal": +1.5, "bearish revearsal":-1.5, "consolidation":+0.5, "expansion":-0.5 }
        score += trend_score[df_trend["trend"].values[-1]]
        # update dict_analysis
        DictAnalysis.update({df_trend["trend"].values[-1] : None})"""
    text = "Trend indicates "+df_trend["trend"].values[-1]+". "
    if "message" in message_closest["closest"].keys():
        message_closest["closest"]["message"] = message_closest["closest"]["message"] + text
    else:
            message_closest["closest"].update({"message":text})
    # update the score with trend
    trend_score = {"bull": +1, "bear": -1,"bullish reversal": +1.5, "bearish revearsal":-1.5, "consolidation":+0.5, "expansion":-0.5 }
    score += trend_score[df_trend["trend"].values[-1]]
    # update dict_analysis
    DictAnalysis.update({df_trend["trend"].values[-1] : None})

    # add message from the tweets
    df_stats_temp = df_stats.loc[((df_stats['token']+"USDT" == currency) & (df_stats['timeFrame'] == TimeFrame))]
    if df_stats_temp.shape[0] > 0:
        score = score*1.5 #df_stats_temp["value"][0]/8 
        text = "Trending on twitter."
        if "message" in message_closest["closest"].keys():
            message_closest["closest"]["message"] = message_closest["closest"]["message"] + text
        else:
            message_closest["closest"].update({"message":text})
        # update dict_analysis
        DictAnalysis.update({"tweet" : df_stats_temp["value"].values[0]})

    # modify score based on Time Frames
    print("account for time frame in scoring.")
    score_orig = copy.deepcopy(score)
    coef = TimeStampDiff(1, TimeFrame) / (24*3600*1000)
    score = copy.deepcopy(score*coef)

    print("add messages to the dataframe.")
    if  message_closest["closest"]["message"] != "":
        #list_message.append(message_closest)
        df_temp = pd.DataFrame(columns=["TimeFrame", "currency", "message","timestamp"],index=range(1))
        df_temp["TimeFrame"] = message_closest["closest"]["TimeFrame"]
        df_temp["currency"] = message_closest["closest"]["currency"]
        df_temp["message"] = message_closest["closest"]["message"]
        df_temp["timestamp"] = datetime.timestamp(datetime.now())*1000
        df_temp["score"] = score
        df_temp["score_orig"] = score_orig
        #df_message = df_message.append(df_temp)
        df_message = copy.deepcopy(df_temp)
        #print(df_message)
    #     
    #saved_message = df_message.to_json(date_format='iso', orient='split')
    #print("saved_message",saved_message)
    print("End generate_message")
    return df_message, DictAnalysis #saved_message