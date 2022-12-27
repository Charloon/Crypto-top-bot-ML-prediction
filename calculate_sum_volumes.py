import pandas as pd
import numpy as np
import copy

def calculate_sum_volumes(df, list_top, list_bot):
    print("calculate_sum_volumes", calculate_sum_volumes)
    
    list_tot = list(set(list_top+list_bot))
    list_tot.sort()    
    
    list_vol = [0] #np.mean(df["volume"].iloc[list_tot])]
    """for i in range(len(list_tot)-1):
        list_vol.append(np.mean(df["volume"].values[list_tot[i]+1:list_tot[i+1]+1])*(list_tot[i+1]-list_tot[i]))
    list_vol.append(np.mean(df["volume"].values[list_tot[-1]+1:])*(df.index.values[-1]-list_tot[-1]))"""
    for i in range(1, len(list_tot)):
        list_vol.append(np.sum(df["volume"].values[list_tot[i-1]+1:list_tot[i]+1]))
    list_vol.append(np.sum(df["volume"].values[list_tot[-1]+1:]))
    """for i in range(1, len(list_tot)):
        list_vol.append(np.mean(df["volume"].values[list_tot[i-1]+1:list_tot[i]+1]))
    list_vol.append(np.mean(df["volume"].values[list_tot[-1]+1:]))"""
    """n = 1
    for i in range(1, len(list_tot)):
        list_vol.append(np.mean(df["volume"].values[list_tot[i]-n:list_tot[i]+1]))"""
    #list_vol.append(np.mean(df["volume"].values[-n:]))

    # plot 1
    x1 = []
    y1 = []
    for i in range(len(list_tot)-1):
        x1.append(list_tot[i])
        y1.append(list_vol[i])
        x1.append(list_tot[i+1])
        y1.append(list_vol[i])
    x1.append(list_tot[-1])
    y1.append(list_vol[-1])
    x1.append(df.index.values[-1])
    y1.append(list_vol[-1])   
    
    # plot 2
    x_dw = []
    y_dw = []
    x_up = []
    y_up = []
    for i in range(1,len(list_tot)):
        if list_tot[i] in list_bot:
            x_dw.append(list_tot[i])
            y_dw.append(list_vol[i])
            """try:
                x_up.append(list_tot[i+1])
                y_up.append(list_vol[i-1])
            except:
                pass"""
        if list_tot[i] in list_top:
            x_up.append(list_tot[i])
            y_up.append(list_vol[i])
            """try:
                x_dw.append(list_tot[i+1])
                y_dw.append(list_vol[i-1])
            except:
                pass"""
    if list_tot[-1] in list_top:
        x_dw.append(df.index.values[-1])
        y_dw.append(list_vol[-1])
        """x_up.append(df.index.values[-1])
        y_up.append(list_vol[-2])"""
    if list_tot[-1] in list_bot:
        x_up.append(df.index.values[-1])
        y_up.append(list_vol[-1])
        """x_dw.append(df.index.values[-1])
        y_dw.append(list_vol[-2])"""
    
    ## find divergence
    tol = 1e-2
    # when price goes down in down trend (LL)
    # but volume goes up
    """y_up_div_g=[np.nan]*len(y_up)
    price = copy.deepcopy(df["high"].values[x_up])
    for i in range(1,len(y_up_div_g)):
        Dprices = (price[i] - price[i-1])/(price[i] + price[i-1])*2
        Dy_up = (y_up[i] - y_up[i-1])/(y_up[i] + y_up[i-1])*2
        if Dprices < -tol and Dy_up > tol:
            y_up_div_g[i] = y_up[i]
            y_up_div_g[i-1] = y_up[i-1]

    # when price goes up in down trend (LL)
    # but volume of retracemnt diminish
    y_dw_div_g=[np.nan]*len(y_dw)
    price = copy.deepcopy(df["low"].values[x_dw])
    for i in range(1,len(y_dw_div_g)):
        Dprices = (price[i] - price[i-1])/(price[i] + price[i-1])*2
        Dy_dw = (y_dw[i] - y_dw[i-1])/(y_dw[i] + y_dw[i-1])*2
        if Dprices < -tol and Dy_dw < -tol:
            y_dw_div_g[i] = y_dw[i]
            y_dw_div_g[i-1] = y_dw[i-1]

    # when price goes down in down trend (HH)
    # but volume goes up on retracement
    y_up_div_r=[np.nan]*len(y_up)
    price = copy.deepcopy(df["high"].values[x_up])
    for i in range(1,len(y_up_div_r)):
        Dprices = (price[i] - price[i-1])/(price[i] + price[i-1])*2
        Dy_up = (y_up[i] - y_up[i-1])/(y_up[i] + y_up[i-1])*2
        if Dprices > tol and Dy_up < -tol:
            y_up_div_r[i] = y_up[i]
            y_up_div_r[i-1] = y_up[i-1]

    # when price goes up in retracement (HH)
    # but volume of retracement diminish
    y_dw_div_r=[np.nan]*len(y_dw)
    price = copy.deepcopy(df["low"].values[x_dw])
    for i in range(1,len(y_dw_div_r)):
        Dprices = (price[i] - price[i-1])/(price[i] + price[i-1])*2
        Dy_dw = (y_dw[i] - y_dw[i-1])/(y_dw[i] + y_dw[i-1])*2
        if Dprices > tol and Dy_dw > tol:
            y_dw_div_r[i] = y_dw[i]
            y_dw_div_r[i-1] = y_dw[i-1]"""


    """y = list_vol
    y_div_g=[np.nan]*len(list_vol)
    price = copy.deepcopy(df["high"].values[list_tot])
    for i in range(1,len(y_div_g)):
        Dprices = (price[i] - price[i-1])/(price[i] + price[i-1])*2
        Dy_up = (y[i] - y[i-1])/(y[i] + y[i-1])*2
        if i in list_bot:
            if Dprices < -tol and Dy_up > tol:  """ 
    

    # when price goes down in down trend (HH)
    # but volume goes up on retracement
    #y_up_price = copy.deepcopy(df["high"].values[x_up])

    """y_div_r=[np.nan]*len(list_vol)
    y_div_g=[np.nan]*len(list_vol)
    price_high = copy.deepcopy(df["high"].values[list_tot])
    price_low = copy.deepcopy(df["low"].values[list_tot])
    for i in range(3,len(list_vol)):
        Dvol = list_vol[i] - list_vol[i-1]
        if list_tot[i] in list_bot:
            Dprices_high = (price_high[i-1] - price_high[i-3])/(price_high[i-1] + price_high[i-3])*2
            Dprices_low = (price_low[i] - price_low[i-2])/(price_low[i] + price_low[i-2])*2
            # if in uptrend HH and HL but volume is lower in high than in lows, sell
            if Dprices_high > tol and Dprices_low > tol:
                if Dvol > 0:
                    y_div_r[i] = list_vol[i]
            # if in down trend but volume is lower in lows than in high, then buy
            elif Dprices_high < tol and Dprices_low < tol:
                if Dvol < 0:
                    y_div_g[i] = list_vol[i]
        elif list_tot[i] in list_top:
            Dprices_high = (price_high[i] - price_high[i-2])/(price_high[i] + price_high[i-2])*2
            Dprices_low = (price_low[i-1] - price_low[i-3])/(price_low[i-1] + price_low[i-3])*2
            # if in uptrend HH and HL
            # if in uptrend HH and HL but volume is lower in high than in lows, sell
            if Dprices_high > tol and Dprices_low > tol:
                if Dvol < 0:
                    y_div_r[i] = list_vol[i]
            # if in down trend but volume is lower in lows than in high, then buy
            elif Dprices_high < tol and Dprices_low < tol:
                if Dvol > 0:
                    y_div_g[i] = list_vol[i]"""

    y_div_r=[np.nan]*len(list_vol)
    y_div_g=[np.nan]*len(list_vol)
    for i in range(3,len(list_vol)-1):
        if list_vol[i-2] <= list_vol[i-1] \
            and list_vol[i] >= list_vol[i-1]:
            if list_tot[i-1] in list_top:
                y_div_r[i] = list_vol[i]
            elif list_tot[i-1] in list_bot:
                y_div_g[i] = list_vol[i]
    for i in range(3,len(list_vol)-1):
        if list_vol[i-2] >= list_vol[i-1] \
            and list_vol[i] <= list_vol[i-1]:
            if list_tot[i-1] in list_top:
                y_div_g[i] = list_vol[i]
            elif list_tot[i-1] in list_bot:
                y_div_r[i] = list_vol[i]
    


    # if in uptrend HH and HL
        # if volume greater in low than high
            # sell signal
        

    # when price goes up in retracement (HH)
    # but volume of retracement diminish
    # y_dw_price = copy.deepcopy(df["low"].values[x_dw])

    
    return x_dw, y_dw, x_up, y_up, y_div_g, y_div_r # y_dw_div_g, y_up_div_g, y_dw_div_r, y_up_div_r, y_up_price, y_dw_price