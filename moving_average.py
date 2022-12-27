import numpy as np
import copy

def detect_cross(self, df_temp, name1, name2): 
    # 20 MA cross 50 MA
    a = df_temp[name1]-df_temp['50MA']
    b = copy.deepcopy(a)
    b[0] = 0
    b[1:] = np.dot(b[1:],b[0:-1])
    df_temp[name1+"x"+name2] = 0
    df_temp[name1+"x"+name2] = np.where(b < 0, 1, 0)
    df_temp[name1+"x"+name2] = copy.deepcopy(np.where((b<0)&(a<0), -1, df_temp[name1+"x"+name2]))
    self.Features.append(name1+"x"+name2)
    return df_temp

def moving_average(self):

    for Currency in self.ListPairs:
        for TimeFrame in self.TimeFrames:
            df_temp = copy.deepcopy(self.DictDfAnalysis[Currency][TimeFrame])

            # 20 MA
            df_temp['20MA'] = df_temp['close'].rolling(window=20).mean()
            self.Features.append("20MA")
            self.FeatPrices.append("20MA")
            self.FeaturesToTranspose.append("20MA")
            df_temp['20MA_close'] = df_temp['close'].rolling(window=20).mean() - df_temp['close']
            #df_temp['20MA_close'] = df_temp['20MA_close'] / df_temp['close'].rolling(window=20).std()
            self.Features.append("20MA_close")
            self.FeatDiffPrices.append("20MA_close")
            self.FeaturesToTranspose.append("20MA_close")

            # 50 MA
            df_temp['50MA'] = df_temp['close'].rolling(window=50).mean()
            self.Features.append("50MA")
            self.FeatPrices.append("50MA")
            self.FeaturesToTranspose.append("50MA")
            df_temp['50MA_close'] = df_temp['close'].rolling(window=50).mean() - df_temp['close']
            #df_temp['50MA_close'] = df_temp['50MA_close'] / df_temp['close'].rolling(window=50).std()
            self.Features.append("50MA_close")
            self.FeatDiffPrices.append("50MA_close")
            self.FeaturesToTranspose.append("50MA_close")

            # 100 MA
            df_temp['100MA'] = df_temp['close'].rolling(window=100).mean()
            self.Features.append("100MA")
            self.FeatPrices.append("100MA")
            self.FeaturesToTranspose.append("100MA")
            df_temp['100MA_close'] = df_temp['close'].rolling(window=100).mean() - df_temp['close']
            #df_temp['100MA_close'] = df_temp['100MA_close'] / df_temp['close'].rolling(window=100).std()
            self.Features.append("100MA_close")
            self.FeatDiffPrices.append("100MA_close")
            self.FeaturesToTranspose.append("100MA_close")

            # 200 MA
            df_temp['200MA'] = df_temp['close'].rolling(window=200).mean()
            self.Features.append("200MA")
            self.FeatPrices.append("200MA")
            self.FeaturesToTranspose.append("200MA")
            df_temp['200MA_close'] = df_temp['close'].rolling(window=200).mean() - df_temp['close']
            #df_temp['200MA_close'] = df_temp['200MA_close'] / df_temp['close'].rolling(window=200).std()
            self.Features.append("200MA_close")
            self.FeatDiffPrices.append("200MA_close")
            self.FeaturesToTranspose.append("200MA_close")

            df_temp = detect_cross(self, df_temp, "20MA", "50MA")    
            df_temp = detect_cross(self, df_temp, "20MA", "100MA")
            df_temp = detect_cross(self, df_temp, "20MA", "200MA")
            df_temp = detect_cross(self, df_temp, "50MA", "100MA")
            df_temp = detect_cross(self, df_temp, "50MA", "200MA")
            df_temp = detect_cross(self ,df_temp, "100MA", "200MA")

            # save it back
            self.DictDfAnalysis[Currency][TimeFrame] = copy.deepcopy(df_temp)

    return self