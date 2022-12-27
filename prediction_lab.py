from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timezone
from SendGetDataFrameS3 import *

INCREASING_COLOR = '#17BECF'
DECREASING_COLOR = '#7F7F7F'

def train_and_predict_signal_regressor(PricePredAnalysis, TF, quantile = 0.95, HyperParamOpt = True):

    # initialize
    InferenceEngine = "XGBR"
    TargetNameTop = "target"+"top"
    TargetNameBot = "target"+"bot"
    """quantile = 0.95"""

    # train for tops 
    PricePredAnalysis.train(TargetNameTop+"_"+TF, "XGBR", TF, np.arange(10), HyperParamOpt = HyperParamOpt)
    PricePredAnalysis.predict(TargetNameTop+"_"+TF, "XGBR", TF)
    # train for bots
    PricePredAnalysis.train(TargetNameBot+"_"+TF, "XGBR", TF ,np.arange(10), HyperParamOpt = HyperParamOpt)
    PricePredAnalysis.predict(TargetNameBot+"_"+TF, "XGBR", TF)

    # plot candles with regression
    # data top
    name_top = TargetNameTop+"_"+TF+"_"+TF+"_"+InferenceEngine
    xtest_top = PricePredAnalysis.DictModel[name_top]["xtest"]
    ytest_top = PricePredAnalysis.DictModel[name_top]["ytest"]
    xtrain_top = PricePredAnalysis.DictModel[name_top]["xtrain"]
    ytrain_top = PricePredAnalysis.DictModel[name_top]["ytrain"]
    ytest_predict_top = PricePredAnalysis.DictModel[name_top]["ytest_predict"]
    ytrain_predict_top = PricePredAnalysis.DictModel[name_top]["ytrain_predict"]
    #list_top = np.where(ytest_predict_top > max(ytest_predict_top)*0.1)[0]
    list_top = np.where(ytest_predict_top > np.quantile(ytest_predict_top, quantile))[0]
    #list_top = np.where(ytest_predict_top > 0.5)[0]
    #print("list_top", list_top)
    #print(np.take(ytest_predict_top, list_top))
    #input()
    # data bot
    name_bot = TargetNameBot+"_"+TF+"_"+TF+"_"+InferenceEngine
    xtest_bot = PricePredAnalysis.DictModel[name_bot]["xtest"]
    ytest_bot = PricePredAnalysis.DictModel[name_bot]["ytest"]
    xtrain_bot = PricePredAnalysis.DictModel[name_bot]["xtrain"]
    ytrain_bot = PricePredAnalysis.DictModel[name_bot]["ytrain"]
    ytest_predict_bot = PricePredAnalysis.DictModel[name_bot]["ytest_predict"]
    ytrain_predict_bot = PricePredAnalysis.DictModel[name_bot]["ytrain_predict"]
    #list_bot = np.where(ytest_predict_bot > np.max(ytest_predict_bot)*0.1)[0]
    list_bot = np.where(ytest_predict_bot > np.quantile(ytest_predict_bot, quantile))[0]
    #list_bot = np.where(ytest_predict_bot < -0.5)[0]
                      
    # organise plot                  
    # plot condle sticks
    price_chart_figure = make_subplots()
    """price_chart_figure.add_trace(go.Candlestick(x=pd.to_datetime(xtrain_bot.dateTime, unit='ms'),
                open=xtrain_bot["open"+"_"+TF]*xtrain_bot[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF] + xtrain_bot[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF],
                high=xtrain_bot["high"+"_"+TF]*xtrain_bot[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF] + xtrain_bot[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF],
                low=xtrain_bot["low"+"_"+TF]*xtrain_bot[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF] + xtrain_bot[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF],
                close=xtrain_bot["close"+"_"+TF]*xtrain_bot[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF] + xtrain_bot[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF],
                increasing=dict(line=dict(color=INCREASING_COLOR)),
                decreasing=dict(line=dict(color=DECREASING_COLOR))),
                secondary_y=True)"""
    time = pd.to_datetime(xtest_bot.dateTime, unit='ms')
    open = xtest_bot["open"+"_"+TF]*xtest_bot[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF] + xtest_bot[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF]
    high = xtest_bot["high"+"_"+TF]*xtest_bot[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF] + xtest_bot[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF]
    low = xtest_bot["low"+"_"+TF]*xtest_bot[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF] + xtest_bot[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF]
    close = xtest_bot["close"+"_"+TF]*xtest_bot[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF] + xtest_bot[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF]
    price_chart_figure.add_trace(go.Candlestick(x=time,
                open=open,
                high=high,
                low=low,
                close=close,
                increasing=dict(line=dict(color=INCREASING_COLOR)),
                decreasing=dict(line=dict(color=DECREASING_COLOR))))
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
    priceTops = xtest_top["high"+"_"+TF].iloc[list_top]*xtest_top[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF].iloc[list_top] + \
                                    xtest_top[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF].iloc[list_top]
    colorTops = np.take(ytest_predict_top, list_top)

    price_chart_figure.add_trace(go.Scatter(x=timeTops, 
                            y=priceTops,
                            name='top test',
                            mode='markers',
                            text=colorTops,
                            marker=dict(color=colorTops, size=10, showscale=True, coloraxis ="coloraxis",
                            colorscale = colorscale1)))

    timeBots = pd.to_datetime(xtest_top.dateTime.iloc[list_bot], unit='ms')
    priceBots = xtest_top["low"+"_"+TF].iloc[list_bot]*xtest_top[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF].iloc[list_bot] + \
                                    xtest_top[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF].iloc[list_bot]
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
    #plt.figure()
    df_temp1 = pd.DataFrame(columns = ["measured", "predict"], index = range(len(ytest_predict_top)))
    df_temp1["measured"] = ytest_top.values 
    df_temp1["predict"] = ytest_predict_top
    fig = px.box(df_temp1, x="measured", y="predict", points="all", title = "TOP test")
    fig.show()

    #plt.figure()
    df_temp1 = pd.DataFrame(columns = ["measured", "predict"], index = range(len(ytest_predict_bot)))
    df_temp1["measured"] = ytest_bot.values 
    df_temp1["predict"] = ytest_predict_bot
    fig = px.box(df_temp1, x="measured", y="predict", points="all", title = "BOT test")
    fig.show()

    # cross plot
    #fig = make_subplots()
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

    #000000000000000000000000000000000000000000000000
    df_price["utcNow"] = datetime.now(timezone.utc).strftime("%Y%m%d")
    filename = "df_BlindTestAndSignal_" + PricePredAnalysis.ListPairs[0] + "_" + TF + ".csv"
    sendDataframeToS3(df_price, filename, location="s3")
    df_price.to_csv(filename)

    return {"xtop" : xtop, "ytop": ytop, "xbot" : xbot, "ybot": ybot, "xtest": xtest_top}


def train_and_predict_signal_classifier(PricePredAnalysis, TF):
    # initialize
    InferenceEngine = "XGBC"
    TargetNameTop = "target"+"top"
    TargetNameBot = "target"+"bot"

    # train for tops
    PricePredAnalysis.train(TargetNameTop+"_"+TF, "XGBC", TF)
    PricePredAnalysis.predict(TargetNameTop+"_"+TF, "XGBC", TF)
    # train for bots
    PricePredAnalysis.train(TargetNameBot+"_"+TF, "XGBC", TF)
    PricePredAnalysis.predict(TargetNameBot+"_"+TF, "XGBC", TF)

    # plot candles with regression
    # data top
    
    name_top = TargetNameTop+"_"+TF+"_"+TF+"_"+InferenceEngine
    xtest_top = PricePredAnalysis.DictModel[name_top]["xtest"]
    ytest_top = PricePredAnalysis.DictModel[name_top]["ytest"]
    xtrain_top = PricePredAnalysis.DictModel[name_top]["xtrain"]
    ytrain_top = PricePredAnalysis.DictModel[name_top]["ytrain"]
    ytest_predict_top = PricePredAnalysis.DictModel[name_top]["ytest_predict"]
    ytrain_predict_top = PricePredAnalysis.DictModel[name_top]["ytrain_predict"]
    list_top = np.where(ytest_predict_top == 1)[0]
    print("list_top", list_top)

    # data bot
    name_bot = TargetNameBot+"_"+TF+"_"+TF+"_"+InferenceEngine
    xtest_bot = PricePredAnalysis.DictModel[name_bot]["xtest"]
    ytest_bot = PricePredAnalysis.DictModel[name_bot]["ytest"]
    xtrain_bot = PricePredAnalysis.DictModel[name_bot]["xtrain"]
    ytrain_bot = PricePredAnalysis.DictModel[name_bot]["ytrain"]
    ytest_predict_bot = PricePredAnalysis.DictModel[name_bot]["ytest_predict"]
    ytrain_predict_bot = PricePredAnalysis.DictModel[name_bot]["ytrain_predict"]
    #list_bot = np.where(ytest_predict_bot > np.max(ytest_predict_bot)*0.1)[0]
    list_bot = np.where(ytest_predict_bot == 1)[0]
                      
    # organise plot                  
    # plot condle sticks
    price_chart_figure = make_subplots()
    """price_chart_figure.add_trace(go.Candlestick(x=pd.to_datetime(xtrain_bot.dateTime, unit='ms'),
                open=xtrain_bot["open"+"_"+TF]*xtrain_bot[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF] + xtrain_bot[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF],
                high=xtrain_bot["high"+"_"+TF]*xtrain_bot[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF] + xtrain_bot[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF],
                low=xtrain_bot["low"+"_"+TF]*xtrain_bot[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF] + xtrain_bot[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF],
                close=xtrain_bot["close"+"_"+TF]*xtrain_bot[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF] + xtrain_bot[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF],
                increasing=dict(line=dict(color=INCREASING_COLOR)),
                decreasing=dict(line=dict(color=DECREASING_COLOR))),
                secondary_y=True)"""
    price_chart_figure.add_trace(go.Candlestick(x=pd.to_datetime(xtest_bot.dateTime, unit='ms'),
                open=xtest_bot["open"+"_"+TF]*xtest_bot[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF] + xtest_bot[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF],
                high=xtest_bot["high"+"_"+TF]*xtest_bot[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF] + xtest_bot[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF],
                low=xtest_bot["low"+"_"+TF]*xtest_bot[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF] + xtest_bot[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF],
                close=xtest_bot["close"+"_"+TF]*xtest_bot[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF] + xtest_bot[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF],
                increasing=dict(line=dict(color=INCREASING_COLOR)),
                decreasing=dict(line=dict(color=DECREASING_COLOR))))
    # plot tops and bottom
    # define color map
    cmax = max(ytest_predict_top)
    cmin = min(-ytest_predict_bot)
    c0 = -cmin/(cmax-cmin)

    print(cmin, c0, cmax)
    
    
    colorscale1 = [(0.0,"green"), (c0,"white"), (1.0, "red")]

    price_chart_figure.update(layout_xaxis_rangeslider_visible=False)

    price_chart_figure.add_trace(go.Scatter(x=pd.to_datetime(xtest_top.dateTime.iloc[list_top], unit='ms'), 
                            y=xtest_top["high"+"_"+TF].iloc[list_top]*xtest_top[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF].iloc[list_top] + \
                                    xtest_top[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF].iloc[list_top],
                            name='top test',
                            mode='markers',
                            marker=dict(color=np.take(ytest_predict_top, list_top),size=10, showscale=True, coloraxis ="coloraxis",
                            colorscale = colorscale1)))

    price_chart_figure.add_trace(go.Scatter(x=pd.to_datetime(xtest_top.dateTime.iloc[list_bot], unit='ms'), 
                            y=xtest_top["low"+"_"+TF].iloc[list_bot]*xtest_top[PricePredAnalysis.StdDevTranspose+ "_" + PricePredAnalysis.LowestTF].iloc[list_bot] + \
                                    xtest_top[PricePredAnalysis.MeanTranspose+ "_" + PricePredAnalysis.LowestTF].iloc[list_bot],
                            name='bot test',mode='markers',
                            marker=dict(color=-np.take(ytest_predict_bot, list_bot),size=10, showscale=True, coloraxis ="coloraxis",
                            colorscale = colorscale1))) 

    price_chart_figure.layout.paper_bgcolor='rgba(0,0,0,0)'
    price_chart_figure.layout.plot_bgcolor='rgba(0,0,0,0)'
    price_chart_figure.layout.colorscale.diverging = colorscale1
    price_chart_figure.show()

    # boxplot
    #plt.figure()
    df_temp1 = pd.DataFrame(columns = ["measured", "predict"], index = range(len(ytest_predict_top)))
    df_temp1["measured"] = ytest_top.values 
    df_temp1["predict"] = ytest_predict_top
    fig = px.box(df_temp1, x="measured", y="predict", points="all", title = "TOP test")
    fig.show()

    #plt.figure()
    df_temp1 = pd.DataFrame(columns = ["measured", "predict"], index = range(len(ytest_predict_bot)))
    df_temp1["measured"] = ytest_bot.values 
    df_temp1["predict"] = ytest_predict_bot
    fig = px.box(df_temp1, x="measured", y="predict", points="all", title = "BOT test")
    fig.show()

    # cross plot
    #fig = make_subplots()
    df_temp0 = pd.DataFrame(columns = ["measured", "predict", "name"], index = range(len(ytrain_predict_top)))
    df_temp0["measured"] = ytrain_top.values 
    df_temp0["predict"] = ytrain_predict_top
    df_temp0["name"] = "R2 train top:"+str(r2_score(ytrain_top.values, ytrain_predict_top))
    fig = px.scatter(df_temp0, x="measured", y="predict", color = "name")
    df_temp1 = pd.DataFrame(columns = ["measured", "predict", "name"], index = range(len(ytest_predict_bot)))
    df_temp1["measured"] = -ytest_bot.values 
    df_temp1["predict"] = -ytest_predict_bot
    df_temp1["name"] = "R2 test top:"+str(r2_score(-ytest_top.values, -ytest_predict_bot))
    fig2 = px.scatter(df_temp1, x="measured", y="predict", color = "name")
    fig.add_trace(fig2.data[0])
    fig.show()

    #export data
    xtop  = xtest_top.dateTime.iloc[list_top]
    ytop = np.take(ytest_predict_top, list_top)
    xbot  = xtest_bot.dateTime.iloc[list_bot]
    ybot = np.take(ytest_predict_bot, list_bot)   

    return {"xtop" : xtop, "ytop": ytop, "xbot" : xbot, "ybot": ybot}

    






