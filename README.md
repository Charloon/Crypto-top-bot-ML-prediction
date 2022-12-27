# Crypto-top-bot-ML-prediction

This code is an experiment at predicting top and bottoms for cryptos. Data are fetched from Binance, preprocess to create different features, fed to a ML model for training on one part of the data and the predict on the rest.
Authentification keys are needed in files configS3.py and configMarketData to access WS S3 storage and Binance API.

Below is an illustration of the prediction on the test set. Reds dots are likely local tops, and green are likely local bottoms.
![image](https://user-images.githubusercontent.com/55462061/209724288-27bec832-5647-4d8a-a1c8-00717700cba6.png)

Many features are created for the ML model, such as prices, RSI, divergence, moving averages, chopiness, etc .. on two diferent time frames (1d and 3d). Below is an example of the top feature importance.
![image](https://user-images.githubusercontent.com/55462061/209724837-b655a8c1-4643-4836-8ba3-9413b4bd6c30.png)

