import numpy as np
import pandas as pd
import copy
from support import generate_analysis
from utils import cross_detection
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def RSI_formula(df, close, n_period = 14):
    #print(df)
    df["U"] = np.where(df[close].diff() < 0, 0,  df[close].diff())
    #print(df["U"])
    df["D"] = np.where(-df[close].diff() < 0, 0,  -df[close].diff())
    #print(df["D"])
    df["U"] = df["U"].rolling(window = n_period).mean()
    #print(df["U"])
    df["D"] = df["D"].rolling(window = n_period).mean()
    #print(df["D"])
    df["RS"] = df["U"]/df["D"]
    df["RSI"] = 100 - 100/(1+df["RS"])

    return df, ["RSI"]

def divergence(df, rsi, close, list_top, list_bot):
    list_divergence = []
    list_feat = []
    
    #### bullish divervence ####
    name = "BullishDiv"
    list_feat.append(copy.deepcopy(name))
    df[name] = 0
    for i in range(1,len(list_bot)):
        p1 = df["low"].values[list_bot[i-1]]
        p2 = df["low"].values[list_bot[i]]
        i1 = df[rsi].values[list_bot[i-1]]
        i2 = df[rsi].values[list_bot[i]]
        if p1 > p2 and i1 < i2:
            df[name].values[list_bot[i]] = 1
            list_divergence.append({copy.deepcopy(name):[list_bot[i-1], list_bot[i]]})

    #### hidden bullish divergence ####
    name = "HiddenBullishDiv"
    list_feat.append(copy.deepcopy(name))
    df[name] = 0
    for i in range(1,len(list_bot)):
        p1 = df["low"].values[list_bot[i-1]]
        p2 = df["low"].values[list_bot[i]]
        i1 = df[rsi].values[list_bot[i-1]]
        i2 = df[rsi].values[list_bot[i]]
        if p1 < p2 and i1 > i2:
            df[name].values[list_bot[i]] = 1
            list_divergence.append({copy.deepcopy(name):[list_bot[i-1], list_bot[i]]})

    #### Bearish divervence ####
    name = "BearishDiv"
    list_feat.append(copy.deepcopy(name))
    df[name] = 0
    for i in range(1,len(list_top)):
        p1 = df["high"].values[list_top[i-1]]
        p2 = df["high"].values[list_top[i]]
        i1 = df[rsi].values[list_top[i-1]]
        i2 = df[rsi].values[list_top[i]]
        if p1 > p2 and i1 < i2:
            df[name].values[list_top[i]] = 1
            list_divergence.append({copy.deepcopy(name):[list_top[i-1], list_top[i]]})

    #### hidden bearish divergence ####
    name = "HiddenBearishDiv"
    list_feat.append(copy.deepcopy(name))
    df[name] = 0
    for i in range(1,len(list_top)):
        p1 = df["high"].values[list_top[i-1]]
        p2 = df["high"].values[list_top[i]]
        i1 = df[rsi].values[list_top[i-1]]
        i2 = df[rsi].values[list_top[i]]
        if p1 < p2 and i1 > i2:
            df[name].values[list_top[i]] = 1
            list_divergence.append({copy.deepcopy(name):[list_top[i-1], list_top[i]]})

    return df, list_feat, list_divergence

"""def rsi_cross_detection(df, rsi):
    oversold = 70
    overbought = 30
    list_name = []

    #### detect into oversold ####
    name = "RSIintoOverSold"
    df[name] = 0    
    for i in range(1,df.shape[0]):
        if df["RSI"].values[i] - df["RSI"].values[i-1] > 0:
            if df["RSI"].values[i-1] <= oversold and df["RSI"].values[i] >  oversold:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect out of oversold ####
    name = "RSIoutOfOverSold"
    df[name] = 0    
    for i in range(1,df.shape[0]):
        if df["RSI"].values[i] - df["RSI"].values[i-1] < 0:
            if df["RSI"].values[i-1] > oversold and df["RSI"].values[i] <=  oversold:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect in oversold ####
    name = "RSIinOverSold"
    df[name] = 0    
    for i in range(0,df.shape[0]):
        if df["RSI"].values[i] >= oversold:
            df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect into overbought ####
    name = "RSIintoOverBought"
    df[name] = 0    
    for i in range(1,df.shape[0]):
        if df["RSI"].values[i] - df["RSI"].values[i-1] < 0:
            if df["RSI"].values[i-1] > overbought and df["RSI"].values[i] <=  overbought:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect out of overbought ####
    name = "RSIoutOfOverBought"
    df[name] = 0    
    for i in range(1,df.shape[0]):
        if df["RSI"].values[i] - df["RSI"].values[i-1] > 0:
            if df["RSI"].values[i-1] < overbought and df["RSI"].values[i] >=  overbought:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect in overbought ####
    name = "RSIinOverBought"
    df[name] = 0    
    for i in range(0,df.shape[0]):
        if df["RSI"].values[i] <= oversold:
            df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect cross below 50 ####
    name = "RSIabove50"
    df[name] = 0    
    for i in range(1,df.shape[0]):
        if df["RSI"].values[i] - df["RSI"].values[i-1] > 0:
            if df["RSI"].values[i-1] < 50 and df["RSI"].values[i] >=  50:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect cross above 50 ####
    name = "RSIbelow50"
    df[name] = 0    
    for i in range(1,df.shape[0]):
        if df["RSI"].values[i] - df["RSI"].values[i-1] < 0:
            if df["RSI"].values[i-1] > 50 and df["RSI"].values[i] <=  50:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    return df, list_name"""

def plot_RSI(df, close = "close", list_divergence = []):
    #df.to_csv("df.csv")
    fig = make_subplots(rows=2, cols=1)
    for item in list_divergence:
        print(item)
        name = list(item.keys())[0]
        if name in ["HiddenBullishDiv", "BullishDiv"]:
            trace_price = go.Scatter(
                        x=[item[name][0], item[name][1]],
                        y=[df["low"].values[item[name][0]],df["low"].values[item[name][1]]],
                        name=name, line = {"color" :"#20B403"}, showlegend = False, mode='lines')
            trace_rsi = go.Scatter(
                        x=[item[name][0], item[name][1]],
                        y=[df["RSI"].values[item[name][0]],df["RSI"].values[item[name][1]]],
                        name=name, line = {"color" :"#20B403"}, showlegend = False, mode='lines')
        elif name in ["HiddenBearishDiv", "BearishDiv"]:
            trace_price = go.Scatter(
                        x=[item[name][0], item[name][1]],
                        y=[df["high"].values[item[name][0]],df["high"].values[item[name][1]]],
                        name=name, line = {"color" :"#FC1805"}, showlegend = False, mode='lines')
            trace_rsi = go.Scatter(
                        x=[item[name][0], item[name][1]],
                        y=[df["RSI"].values[item[name][0]],df["RSI"].values[item[name][1]]],
                        name=name, line = {"color" :"#FC1805"}, showlegend = False, mode='lines')
        fig.add_trace(copy.deepcopy(trace_price), row=1, col=1)
        fig.add_trace(copy.deepcopy(trace_rsi), row=2, col=1)
    trace1 = go.Scatter(
        x=np.arange(df.shape[0]),
        y=df[close],
        name=close)
    trace2 = go.Scatter(
        x=np.arange(df.shape[0]),
        y=df["RSI"],
        name='RSI')
    
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=2, col=1)
    fig.update_layout(xaxis=dict(tickangle=90))
    fig.show()

def RSI(self):
    for Currency in self.ListPairs:
        for TimeFrame in self.TimeFrames:
            df_temp = copy.deepcopy(self.DictDfAnalysis[Currency][TimeFrame])

            # calculate RSI
            df_temp, list_feat = RSI_formula(df_temp, "close")
            for feat in list_feat:
                self.Features.append(copy.deepcopy(feat))
                self.FeaturesToTranspose.append(copy.deepcopy(feat))

            # detect crossing of RSI
            df_temp, list_feat = cross_detection(df_temp, "RSI")
            for feat in list_feat:
                self.Features.append(copy.deepcopy(feat))
                self.FeaturesToTranspose.append(copy.deepcopy(feat))

            # detect divergence 
            list_ressup, list_res, list_sup, list_hz, list_top, list_bot = generate_analysis(df_temp, nbr_days = df_temp.shape[0]-1, window = 20, step_factor = 5)
            df_temp, list_feat, list_divergence = divergence(df_temp, "RSI", "close", list_top, list_bot)
            for feat in list_feat:
                self.Features.append(copy.deepcopy(feat))
                self.FeaturesToTranspose.append(copy.deepcopy(feat))

            # plot RSI
            plot_RSI(df_temp, "close", list_divergence)

            # df_temp.to_csv("df_temp.csv")

            # save it back
            self.DictDfAnalysis[Currency][TimeFrame] = copy.deepcopy(df_temp)

    return self

#df = pd.DataFrame(data = {"close": np.random.shuffle(np.arange(300))})
#df = RSI_formula(df, "close")

#df =pd.read_csv("df.csv")
#list_divergence = [{'BullishDiv': [1607, 1621]}, {'HiddenBearishDiv': [1525, 1546]}]
#plot_RSI(df, "close", list_divergence)

