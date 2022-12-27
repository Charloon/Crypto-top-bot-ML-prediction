import numpy as np
import pandas as pd
import copy
from support import generate_analysis
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def cross_detection(df, rsi):
    oversold = 70
    overbought = 30
    list_name = []

    #### detect into oversold ####
    name = rsi+"intoOverSold"
    df[name] = 0    
    for i in range(1,df.shape[0]):
        if df[rsi+""].values[i] - df[rsi+""].values[i-1] > 0:
            if df[rsi+""].values[i-1] <= oversold and df[rsi+""].values[i] >  oversold:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect out of oversold ####
    name = rsi+"outOfOverSold"
    df[name] = 0    
    for i in range(1,df.shape[0]):
        if df[rsi].values[i] - df[rsi].values[i-1] < 0:
            if df[rsi].values[i-1] > oversold and df[rsi].values[i] <=  oversold:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect in oversold ####
    name = rsi+"inOverSold"
    df[name] = 0    
    for i in range(0,df.shape[0]):
        if df[rsi].values[i] >= oversold:
            df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect into overbought ####
    name = rsi+"intoOverBought"
    df[name] = 0    
    for i in range(1,df.shape[0]):
        if df[rsi].values[i] - df[rsi].values[i-1] < 0:
            if df[rsi].values[i-1] > overbought and df[rsi].values[i] <=  overbought:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect out of overbought ####
    name = rsi+"outOfOverBought"
    df[name] = 0    
    for i in range(1,df.shape[0]):
        if df[rsi].values[i] - df[rsi].values[i-1] > 0:
            if df[rsi].values[i-1] < overbought and df[rsi].values[i] >=  overbought:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect in overbought ####
    name = rsi+"inOverBought"
    df[name] = 0    
    for i in range(0,df.shape[0]):
        if df[rsi].values[i] <= oversold:
            df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect cross below 50 ####
    name = rsi+"above50"
    df[name] = 0    
    for i in range(1,df.shape[0]):
        if df[rsi].values[i] - df[rsi].values[i-1] > 0:
            if df[rsi].values[i-1] < 50 and df[rsi].values[i] >=  50:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    #### detect cross above 50 ####
    name = rsi+"below50"
    df[name] = 0    
    for i in range(1,df.shape[0]):
        if df[rsi].values[i] - df[rsi].values[i-1] < 0:
            if df[rsi].values[i-1] > 50 and df[rsi].values[i] <=  50:
                df[name].values[i] = 1
    list_name.append(copy.deepcopy(name))

    return df, list_name