#from xgboost.sklearn import XGBRegressor
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
import copy
import plotly.graph_objects as go
#from fbprophet import Prophet
#from fbprophet.plot import plot_plotly, plot_components_plotly
#import matplotlib.pyplot as plt
#from itertools import combinations
#from scipy.optimize import minimize
from SendGetDataFrameS3 import *



def generate_analysis(df, nbr_days = 200, window = 20, step_factor = 5): 

    def find_top_bot(df, nbr_days = 200, window = 20, step_factor = 5):   

        window = min(window, int(float(nbr_days) / 2)) # int(float(nbr_days) / window_factor)
        step = 1 #int(float(window) / step_factor)
        border = max(1, int(float(window)/3))
        extension = 3
        nbr_days = min(nbr_days, df.shape[0])
        #tol = 5e-2

        #df_search = df.tail(-nbr_days)
        list_top = []
        list_bot = []
        #df["index"] = np.arange(df.shape[0])

        # extend data set
        df_extendsion = pd.DataFrame(data=df.tail(1).values, columns=df.columns)
        for i in range(extension):
            df_extendsion["dateTime"] = 2*df["dateTime"].values[-1] - df["dateTime"].values[-2] 
            df_extendsion.index.values[0] = df.index.values[-1]+1 
            df = pd.concat([df, df_extendsion])
        df[['open', 'high', 'low', 'close', 'volume']] =  df[['open', 'high', 'low', 'close', 'volume']].astype("float")

        for i in range(df.shape[0] - nbr_days, df.shape[0] - window, step):
            jmin = df["low"].iloc[i:i+window].idxmin() #+ i
            jmax = df["high"].iloc[i:i+window].idxmax() #+ i          
            #jmin = df["low"].iloc[i:i+window].argmin() + i
            #jmax = df["high"].iloc[i:i+window].argmax() + i
            if (jmax < i + window - border) and (jmax > i+border):
                #list_top.append([jmax,df["high"].values[jmax]])
                list_top.append(jmax)
            if (jmin < i + window-border) and (jmin > i+border):
                list_bot.append(jmin)
                #list_bot.append([jmin,df["low"].values[jmin]])
        c = list(set(list_top))
        list_bot = list(set(list_bot))
        list_top = list(set(list_top))
        list_tot = list(set(list_bot + list_top))
        list_bot.sort()
        list_top.sort()
        list_tot.sort()

        # look for missing tops and bottoms
        # missing tops
        for i in range(len(list_bot)-1):
            found = False
            for ele in list_top:
                if ele > list_bot[i] and ele < list_bot[i+1]:
                    found = True
            #found = True in (() for ele in list_top)
            if found == False:
                j = df["high"].iloc[list_bot[i]+1:list_bot[i+1]].idxmax() #+ i
                list_top.append(j)
                list_top.sort()
                
        # missing bottom
        for i in range(len(list_top)-1):
            found = False
            for ele in list_bot:
                if ele > list_top[i] and ele < list_top[i+1]:
                    found = True
            #found = True in (() for ele in list_top)
            if found == False:
                j = df["low"].iloc[list_top[i]+1:list_top[i+1]].idxmin() #+ i
                list_bot.append(j)
                list_bot.sort()
    

        return list_top, list_bot, list_tot


    def find_lines(df, list_tot, list_top, list_bot, tol=1e-2, constrain = None):
        list_line = list()
        max_point = 0
        nbr_max = 0
        list_tot.sort(reverse=False)

        high_top = max(df["high"].values[list_top]) if len(list_top) > 0 else max(df["high"].values)
        low_bot = min(df["low"].values[list_bot]) if len(list_bot) > 0 else min(df["low"])
        deltay = high_top - low_bot        #deltay = max(df["high"].values[list_top]) - min(df["low"].values[list_bot])
        current_price = df["close"].values[-1]
        current_date = df["dateTime"].values[-1]
        current_point = copy.deepcopy(df.index.values[-1])
        for i in range(len(list_tot)):

            # coordinate of i
            i_x = list_tot[i]
            #print("i_x",i_x)
            if i_x in list_top:
                i_y = df["high"].values[i_x]
            elif i_x in list_bot:
                i_y = df["low"].values[i_x]

            for j in range(i+1,len(list_tot)):

                """# coordiante of j
                j_x = list_tot[j]
                #print("   ","j_x",j_x)
                if j_x in list_top:
                    j_y = df["high"].values[j_x]
                elif j_x in list_bot:
                    j_y = df["low"].values[j_x]"""

                # coordiante of j
                j_x = list_tot[j]
                #print("   ","j_x",j_x)
                if j_x in list_top:
                    j_y = df["high"].values[j_x]
                elif j_x in list_bot:
                    j_y = df["low"].values[j_x]                    

                # find line coordinate
                if constrain != "horizontal":
                    a = (j_y - i_y)/(j_x - i_x)
                else:
                    a = 0.0
                b = j_y - a*j_x

                # find near by points
                list_point = [i_x]
                #list_point = [i_x]
                # for k in range(j+1, len(list_tot)):
                for k in range(j, len(list_tot)):    
                    # coordiante of k
                    k_x = list_tot[k]
                    #print("     ","k_x", k_x)
                    if k_x in list_top:
                        k_y = df["high"].values[k_x]
                    elif k_x in list_bot:
                        k_y = df["low"].values[k_x]  
                    # test if point is on the line
                    diff = abs(k_y - (a*k_x + b))/deltay #k_y
                    if diff < tol:
                        #print("add",diff)
                        list_point.append(k_x)

                # find oldest point
                oldest_x = min(list_point)

                # define coordinates for plot
                line_x = [df["dateTime"].values[oldest_x], df["dateTime"].values[df.index.values[-1]]]
                line_y = [a*oldest_x+b, a*df.index.values[-1]+b]

                # number of point 
                nbr_point = len(list_point)

                # nbr point above and below
                list_temp = list_top+list_bot
                nbr_above = 0
                nbr_below = 0
                """for ii in range(len(list_temp)):
                    if list_temp[ii] > oldest_x:

                        ii_x = list_temp[ii]

                        if ii_x not in list_point:
                            if ii_x in list_top:
                                ii_y = df["high"].values[ii_x]
                            elif ii_x in list_bot:
                                ii_y = df["low"].values[ii_x]

                            if ii_y > (a*ii_x+b)*(1.0+1e-2):
                                nbr_above += 1
                            elif ii_y < (a*ii_x+b)*(1.0-2e-2):
                                nbr_below += 1"""

                # apply constrain
                flag_add = True
                if constrain == "top" and nbr_above > 0:
                    flag_add = False
                elif constrain == "bottom" and nbr_below > 0:
                    flag_add = False
                elif constrain == "horizontal" and a != 0.0:
                    flag_add = False
                elif constrain == "horizontal":
                    for l in list_line:
                        if abs((l["b"]-b)/b) < 1e-3:
                            flag_add = False

                # add distance to the current price
                current_line_value = a*current_date + b #a*current_point + b #
                relative_distance = (current_price - current_line_value)/current_price

                # save
                if flag_add == True:
                    list_line.append({"nbr_point":nbr_point,
                                    "line_x": line_x,
                                    "line_y": line_y,
                                    "oldest_x": oldest_x, 
                                    "a": a,
                                    "b": b,
                                    "list_point": list_point,
                                    "nbr_above": nbr_above,
                                    "nbr_below": nbr_below,
                                    "rel_dist": relative_distance,
                                    "current_line_value":current_line_value})
                    max_point = max(max_point, nbr_point)

            # count max nbr of line with the max number of point
            nbr_max = 0
            for i in range(len(list_line)):
                if list_line[i]["nbr_point"] == max_point:
                    nbr_max += 1

        return list_line, max_point, nbr_max


    def opt_tol(df, list_tot, list_top, list_bot, constrain = None, min_connection = 3):

        list_tol = [2e-2] #[1e-2, 2e-2, 3e-2, 4e-2] #, 5e-2, 7e-2, 9e-2]
        list_line = []
        max_point = 0
        nbr_max = 0
        for tol in list_tol:
            
            list_line_temp, max_point_temp, nbr_max_temp = find_lines(df, list_tot, list_top, list_bot, tol, constrain)
            nbr_3point_plus = 0
            for line in list_line_temp:
                if line["nbr_point"] >= min_connection:
                    nbr_3point_plus += 1

            #if (nbr_max_temp <= 5 and max_point == 0):# or (nbr_max_temp <= 2 and max_point_temp > max_point):
            if nbr_3point_plus < 3 or max_point == 0:
                list_line = list_line_temp
                max_point = max_point_temp
                nbr_max = nbr_max_temp
            
        return list_line, max_point, nbr_max    



    ################################################################################################
    # find tops and bottom
    list_top, list_bot, list_tot = find_top_bot(df, nbr_days, window, step_factor)   

    def select_lines_hz(list_line, max_point, list_final, name, min_connection = 2, price_tol = 10000):
        # keep only lines with high number of points
        for i in range(len(list_line)):
            if list_line[i]["nbr_point"] >= min_connection:  #max(2, min(3, max_point)):
                a = list_line[i]
                a.update({"name":name})
                list_final.append(copy.deepcopy(a))

        # join lines that have the same points 
        for i in range(len(list_final)-1):
            for j in range(i+1,len(list_final)):
                if list_final[i]["list_point"] == list_final[j]["list_point"]:
                    list_final[j] = None

        """for j1 in range(len(list_final)-1):
            for j2 in range(j1+1,len(list_final)):
                if list_final[j1] != None:
                    if list_final[j2] != None:
                        list_comp = list(set(list_final[j1]["list_point"]).intersection(list_final[j2]["list_point"]))
                        if len(list_comp) >= 1:
                            if len(list_final[j1]["list_point"]) > len(list_final[j2]["list_point"]):
                                list_final[j2] = None
                            elif len(list_final[j1]["list_point"]) < len(list_final[j2]["list_point"]):
                                list_final[j1] = None
                            else:
                                list_final[j2] = None"""

        #   
        for j1 in range(len(list_final)-1):
            for j2 in range(j1+1,len(list_final)):
                if list_final[j1] != None:
                    if list_final[j2] != None:
                        delta = np.abs(list_final[j1]["b"] - list_final[j2]["b"])
                        if delta < price_tol:
                            if len(list_final[j1]["list_point"]) > len(list_final[j2]["list_point"]):
                                list_final[j2] = None
                            elif len(list_final[j1]["list_point"]) < len(list_final[j2]["list_point"]):
                                list_final[j1] = None
                            else:
                                list_final[j2] = None

        # remove duplicates
        a = []
        if None in list_final:
            a = [x for x in list_final if x != None]
            list_final = a

        return list_final

    """def find_HH_LL(list_point, df):
        list_values = [df["high"].values[i_x]]"""

    def select_lines_supres(list_line, max_point, list_final, list_bot, name, min_connection = 3):
        # keep only lines with high number of points
        #initialize
        
        if name == "resistance":
            a_top = -1e20
            list_values = df["high"].iloc[list_bot].tolist()
            itop = list_bot[list_values.index(max(list_values))]
            ligne_boundary = None
        elif name == "support":
            a_top = 1e20
            list_values = df["low"].iloc[list_bot].tolist()
            itop = list_bot[list_values.index(min(list_values))]
            ligne_boundary = None

        for i in range(len(list_line)):
            if list_line[i]["nbr_point"] >= min_connection:  #max(2, min(3, max_point)):
                a = list_line[i]
                a.update({"name":name})
                list_final.append(copy.deepcopy(a))
            # keep line that join the last two bottom or tops
                """elif list_line[i]["nbr_point"] == 2 and set(list_bot[-2:]).issubset(list_line[i]["list_point"]): # list_bot[-2:].all() in list_line[i]["list_point"]:
                a = list_line[i]
                a.update({"name":name})
                list_final.append(copy.deepcopy(a))"""
            elif list_line[i]["nbr_point"] == 2 and name == "resistance":
                if itop in list_line[i]["list_point"]:
                    if list_line[i]["a"] < 0 and list_line[i]["a"] > a_top:
                        a_top = list_line[i]["a"]
                        ligne_boundary = i
            elif list_line[i]["nbr_point"] == 2 and name == "support":
                if itop in list_line[i]["list_point"]:
                    if list_line[i]["a"] > 0 and list_line[i]["a"] < a_top:
                        a_top = list_line[i]["a"]
                        ligne_boundary = i 

        if ligne_boundary is not None:    
            a = list_line[ligne_boundary]
            a.update({"name":name})
            list_final.append(copy.deepcopy(a))
        
        # join lines that have the same points 
        for i in range(len(list_final)-1):
            for j in range(i+1,len(list_final)):
                if list_final[i]["list_point"] == list_final[j]["list_point"]:
                    list_final[j] = None

        # remove subset
        for j1 in range(len(list_final)-1):
            for j2 in range(j1+1,len(list_final)):
                if list_final[j1] != None:
                    if list_final[j2] != None:
                        list_comp = list(set(list_final[j1]["list_point"]).intersection(list_final[j2]["list_point"]))
                        if len(list_comp) >= 2:
                            if len(list_final[j1]["list_point"]) > len(list_final[j2]["list_point"]):
                                list_final[j2] = None
                            elif len(list_final[j1]["list_point"]) < len(list_final[j2]["list_point"]):
                                list_final[j1] = None
                            else:
                                list_final[j2] = None

        # remove duplicates
        a = []
        if None in list_final:
            a = [x for x in list_final if x != None]
            list_final = a

        return list_final


    def select_lines_sup_or_res(list_line, list_res, list_sup, list_final):
        for i in range(len(list_line)):
            if list_line[i]["nbr_point"] >= 3 and list_line[i]["nbr_below"] > 0 and list_line[i]["nbr_above"] > 0:
                flag_in_resistance = False
                for j in range(len(list_res)):
                    check =  all(item in list_line[i]["list_point"] for item in list_res[j]["list_point"])
                    if len(list(set(list_line[i]["list_point"]) & set(list_res[j]["list_point"]))) >= 2 :
                        flag_in_resistance = True
                        
                    #if list_res[j]["list_point"] == list_line[i]["list_point"]:
                    #    flag_in_resistance = True
                flag_in_support = False
                for j in range(len(list_sup)):
                    check = all(item in list_line[i]["list_point"] for item in list_sup[j]["list_point"])
                    if len(list(set(list_line[i]["list_point"]) & set(list_sup[j]["list_point"]))) >= 2 :
                        flag_in_support = True
                if flag_in_resistance == False and flag_in_support == False: 
                    a = list_line[i]
                    a.update({"name":"S/R"})
                    list_final.append(copy.deepcopy(a))

        # remove subset
        """for j1 in range(len(list_final)-1):
            for j2 in range(j1+1,len(list_final)):
                if list_final[j1] != None:
                    if list_final[j2] != None:
                        if set(list_final[j2]["list_point"]).issubset(list_final[j1]["list_point"]):
                            list_final[j2] = None
                        else:
                            list_final[j1] = None"""

        # remove duplicates
        a = []
        if None in list_final:
            a = [x for x in list_final if x != None]
            list_final = a

        return list_final

    # get horizontal lines
    list_line, max_point, _ = opt_tol(df, list_tot, list_top, list_bot, constrain = "horizontal", min_connection = 3)
    list_hz = []
    #mean_candle = np.mean(np.abs(df["high"].values[-nbr_days]-df["low"].values[-nbr_days]))
    price_tol = (max(np.abs(df["high"].values[-nbr_days:]) - min(np.abs(df["low"].values[-nbr_days:]))))/10
    list_hz = select_lines_hz(list_line, max_point, list_hz, "S/R horizontal", min_connection = 2, price_tol = price_tol )

    # get resistance
    list_line, max_point, _ = opt_tol(df, list_top, list_top, list_bot, constrain = "top", min_connection = 3)
    list_res = []
    list_res = select_lines_supres(list_line, max_point, list_res, list_top, "resistance", 3)

    # get support
    list_line, max_point, _ = opt_tol(df, list_bot, list_top, list_bot, constrain = "bottom", min_connection = 3)
    list_sup = []
    list_sup = select_lines_supres(list_line, max_point, list_sup, list_bot, "support", 3)

    # get resistance and support 
    list_line, max_point, _ = opt_tol(df, list_tot, list_top, list_bot,  constrain = None)
    list_ressup = []
    list_ressup = select_lines_sup_or_res(list_line, list_res, list_sup, list_ressup)
    # generate list to plot identified tops and bottom

    return list_ressup, list_res, list_sup, list_hz, list_top, list_bot

flag_test = False

if flag_test:
    #df = pd.read_csv("./df_BTCUSDT_4htest.csv")
    #df = pd.read_csv("./df_BTCUSDT_1d.csv")
    df = fetchDataframeFromS3("./df_BTCUSDT_1d.csv")

    nbr_days = 200

    list_ressup, list_res, list_sup, list_hz, list_top, list_bot = generate_analysis(df, nbr_days = 1000, window = 20, step_factor = 5)

    data = [dict(
        type='candlestick',
        open=df.open,
        high=df.high,
        low=df.low,
        close=df.close,
        x=df.dateTime,
        yaxis='y',
        name="currency"
    )]

    layout = dict({"xaxis": {"type": "date",  "rangeselector": {"visible": False}},
                    "template": "plotly_dark",
                    "height": 700,
                    'legend': {'orientation': 'h', 'y': 0.9, 'x': 0.3, 'yanchor': 'bottom'}})

    color_top = [3]*len(list_top)
    color_bot = [20]*len(list_bot)
    pred_chart_figure2 = go.Figure(data=data, layout=layout)
    pred_chart_figure2.add_trace(go.Scatter(x=df["dateTime"].iloc[list_top], y=df["high"].iloc[list_top],mode='markers',marker=dict(color=color_top,size=20)))
    pred_chart_figure2.add_trace(go.Scatter(x=df["dateTime"].iloc[list_bot], y=df["low"].iloc[list_bot],mode='markers',marker=dict(color=color_bot,size=20)))

    for x in list_ressup+list_res+list_sup+list_hz:
        pred_chart_figure2.add_trace(go.Scatter(x=x["line_x"], y=x["line_y"], mode='lines', name=x["name"]))

    pred_chart_figure2.update(layout_xaxis_rangeslider_visible=False)
    pred_chart_figure2.update_layout(xaxis_range=[df.dateTime.values[-nbr_days], df.dateTime.values[-1]])
    pred_chart_figure2.show()

    """fig = plt.figure()
    plt.plot(df.index, df["high"], label="high")
    plt.plot(df.index, df["low"], label="low")
    plt.scatter(list_top, df["high"].iloc[list_top], label="high")
    plt.scatter(list_bot, df["low"].iloc[list_bot], label="low")
    plt.xlim([1200, 1400])
    plt.show()"""
