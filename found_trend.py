
import pandas as pd
import numpy as np
import copy

def found_trend(df,list_top, list_bot,timeFrame):

    tol = 1e-2
    list_tot = list(set(list_top+list_bot))
    list_tot.sort()
    df_status = pd.DataFrame(columns={"index","top_or_bot","low_or_high", "value", "timestamp"}, index=range(len(list_tot)))
    df_status["index"] = list_tot
    df_status["timestamp"] = df["dateTime"].values[list_tot]
    # find values
    for i in range(df_status.shape[0]):
        if df_status["index"].values[i] in list_top:
            df_status["value"].values[i] = df["high"].values[df_status["index"].values[i]]
        if df_status["index"].values[i] in list_bot:
            df_status["value"].values[i] = df["low"].values[df_status["index"].values[i]]
    # look if last data points already indicate a LL or HH
    if df_status["index"].values[-1] in list_top:
        if df["low"].values[-1] < df_status["value"].values[-1]:
            df_temp = pd.DataFrame(columns=df_status.columns.values, index=range(1))
            df_temp["index"] = df.index.values[-1]
            df_temp["value"] = df["low"].values[-1]
            df_temp["timestamp"] = df["dateTime"].values[-1]
            df_status = pd.concat([df_status, copy.deepcopy(df_temp)])
            list_bot.append(df_temp["index"].values[0])
    if df_status["index"].values[-1] in list_bot:
        if df["high"].values[-1] > df_status["value"].values[-1]:
            df_temp = pd.DataFrame(columns=df_status.columns.values, index=range(1))
            df_temp["index"] = df.index.values[-1]
            df_temp["value"] = df["high"].values[-1]
            df_temp["timestamp"] = df["dateTime"].values[-1]
            df_status = pd.concat([df_status, copy.deepcopy(df_temp)])
            list_top.append(df_temp["index"].values[0])
    # find low or high
    psi = (np.nanmax(df_status["value"])-np.nanmin(df_status["value"]))*tol
    for i in range(2,df_status.shape[0]):
        if df_status["index"].values[i] in list_top:
            if df_status["value"].values[i] > df_status["value"].values[i-2] + psi:
                df_status["top_or_bot"].values[i] = "top"
                df_status["low_or_high"].values[i] = "HH"
            elif df_status["value"].values[i] < df_status["value"].values[i-2] - psi:
                df_status["top_or_bot"].values[i] = "top"
                df_status["low_or_high"].values[i] = "LH"
            else:
                df_status["top_or_bot"].values[i] = "top"
                df_status["low_or_high"].values[i] = "H"
        if df_status["index"].values[i] in list_bot:
            if df_status["value"].values[i] > df_status["value"].values[i-2] + psi:
                df_status["top_or_bot"].values[i] = "bot"
                df_status["low_or_high"].values[i] = "HL"
            elif df_status["value"].values[i] < df_status["value"].values[i-2] - psi:
                df_status["top_or_bot"].values[i] = "bot"
                df_status["low_or_high"].values[i] = "LL"
            else:
                df_status["top_or_bot"].values[i] = "bot"
                df_status["low_or_high"].values[i] = "H"
        
    # initialize the start and finish dates
    df_status["start"] = min(df["dateTime"].values) 
    df_status["end"] = max(df["dateTime"].values) 
    df_status["trend"] = "?" 
    
    # find trend
    for i in range(1,df_status.shape[0]):
        if df_status["low_or_high"].values[i-1] == "HL" and df_status["low_or_high"].values[i] == "HH":
                df_status["start"].values[i] =  df_status["timestamp"].values[i-1]
                df_status["end"].values[i] = df_status["timestamp"].values[i]
                df_status["trend"].values[i] = "bull"
        elif df_status["low_or_high"].values[i-1] == "HH" and df_status["low_or_high"].values[i] == "HL":
                df_status["start"].values[i] =  df_status["timestamp"].values[i-1]
                df_status["end"].values[i] = df_status["timestamp"].values[i]
                df_status["trend"].values[i] = "bull"
        elif df_status["low_or_high"].values[i-1] == "LL" and df_status["low_or_high"].values[i] == "HH":
                df_status["start"].values[i] =  df_status["timestamp"].values[i-1]
                df_status["end"].values[i] = df_status["timestamp"].values[i]
                df_status["trend"].values[i] = "bullish reversal"
        elif df_status["low_or_high"].values[i-1] == "HL" and df_status["low_or_high"].values[i] == "LH":
                df_status["start"].values[i] =  df_status["timestamp"].values[i-1]
                df_status["end"].values[i] = df_status["timestamp"].values[i]
                df_status["trend"].values[i] = "consolidation"
        elif df_status["low_or_high"].values[i-1] == "HH" and df_status["low_or_high"].values[i] == "LL":
                df_status["start"].values[i] =  df_status["timestamp"].values[i-1]
                df_status["end"].values[i] = df_status["timestamp"].values[i]
                df_status["trend"].values[i] = "bearish revearsal"
        elif df_status["low_or_high"].values[i-1] == "LH" and df_status["low_or_high"].values[i] == "LL":
                df_status["start"].values[i] =  df_status["timestamp"].values[i-1]
                df_status["end"].values[i] = df_status["timestamp"].values[i]
                df_status["trend"].values[i] = "bear"
        elif df_status["low_or_high"].values[i-1] == "LL" and df_status["low_or_high"].values[i] == "LH":
                df_status["start"].values[i] =  df_status["timestamp"].values[i-1]
                df_status["end"].values[i] = df_status["timestamp"].values[i]
                df_status["trend"].values[i] = "bear"
        elif df_status["low_or_high"].values[i-1] == "LH" and df_status["low_or_high"].values[i] == "HL":
                df_status["start"].values[i] =  df_status["timestamp"].values[i-1]
                df_status["end"].values[i] = df_status["timestamp"].values[i]
                df_status["trend"].values[i] = "consolidation"
        elif df_status["low_or_high"].values[i-1] == "LL" and df_status["low_or_high"].values[i] == "HH":
                df_status["start"].values[i] =  df_status["timestamp"].values[i-1]
                df_status["end"].values[i] = df_status["timestamp"].values[i]
                df_status["trend"].values[i] = "expansion"
        elif df_status["low_or_high"].values[i-1] == "HH" and df_status["low_or_high"].values[i] == "LL":
                df_status["start"].values[i] =  df_status["timestamp"].values[i-1]
                df_status["end"].values[i] = df_status["timestamp"].values[i]
                df_status["trend"].values[i] = "expansion"
        else:
                df_status["start"].values[i] =  df_status["timestamp"].values[i-1]
                df_status["end"].values[i] = df_status["timestamp"].values[i]
                df_status["trend"].values[i] = "?"      


    #For current price
    """df_status_temp = pd.DataFrame(columns=df_status.columns.values, index=range(1))
    df_status_temp["start"].values[i] =  df_status["timestamp"].values[i]
    df_status_temp["end"].values[i] = df["dateTime"].values[-1]
    df_status_temp["trend"].values[i] = "?"
    """

    
    df_status["start"] = pd.to_datetime(df_status["start"],unit='ms')
    df_status["end"] = pd.to_datetime(df_status["end"],unit='ms')
    df_status["timeFrame"] = timeFrame
    #df_status.to_csv("df_status.csv")
    df_status = df_status[df_status["trend"] != "?"]

    return df_status


"""import plotly.express as px
import pandas as pd

df = pd.DataFrame([
    dict(Task="Job A", Start='2009-01-01', Finish='2009-02-28', Resource="Alex"),
    dict(Task="Job B", Start='2009-03-05', Finish='2009-04-15', Resource="Alex"),
    dict(Task="Job C", Start='2009-02-20', Finish='2009-05-30', Resource="Max")
])
df.to_csv("df_test.csv")
fig = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", color="Resource")
fig.show()"""



