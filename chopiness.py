import numpy as np
import pandas as pd
import copy
from support import generate_analysis
from utils import cross_detection
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def chopiness_formula(df, close, n_period = 14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(n_period).sum()/n_period
    MaxHi = df['high'].rolling(window = n_period).max()
    MinLo = df['low'].rolling(window = n_period).min()
    sumATR = atr.rolling(window = n_period).sum()
    df["chopiness"] = 100*np.log10(sumATR/(MaxHi - MinLo))/np.log10(n_period)
    #CI = 100 * LOG10( SUM(ATR(1), n) / ( MaxHi(n) - MinLo(n) ) ) / LOG10(n)
    return df, ["chopiness"]

def plot_chopiness(df, close = "close", indicator = "chopiness"):
    #df.to_csv("df.csv")
    fig = make_subplots(rows=2, cols=1)
    trace1 = go.Scatter(
        x=np.arange(df.shape[0]),
        y=df[close],
        name=close)
    trace2 = go.Scatter(
        x=np.arange(df.shape[0]),
        y=df[indicator],
        name=indicator)
    
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=2, col=1)
    fig.update_layout(xaxis=dict(tickangle=90))
    fig.show()

def chopiness(self, indicator="chopiness"):
    for Currency in self.ListPairs:
        for TimeFrame in self.TimeFrames:
            df_temp = copy.deepcopy(self.DictDfAnalysis[Currency][TimeFrame])

            # calculate chopiness
            df_temp, list_feat = chopiness_formula(df_temp, "close")
            for feat in list_feat:
                self.Features.append(copy.deepcopy(feat))
                self.FeaturesToTranspose.append(copy.deepcopy(feat))

            # detect crossing of chopiness
            df_temp, list_feat = cross_detection(df_temp, indicator)
            for feat in list_feat:
                self.Features.append(copy.deepcopy(feat))
                self.FeaturesToTranspose.append(copy.deepcopy(feat))

            # plot RSI
            plot_chopiness(df_temp, "close", indicator)

            # df_temp.to_csv("df_temp.csv")

            # save it back
            self.DictDfAnalysis[Currency][TimeFrame] = copy.deepcopy(df_temp)

    return self

#df = pd.DataFrame(data = {"close": np.random.shuffle(np.arange(300))})
#df = RSI_formula(df, "close")

#df =pd.read_csv("df.csv")
#list_divergence = [{'BullishDiv': [1607, 1621]}, {'HiddenBearishDiv': [1525, 1546]}]
#plot_RSI(df, "close", list_divergence)

