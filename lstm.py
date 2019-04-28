import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
api_key = 'JFPZP3S0PC8VRMZB'

# American Airlines stock market prices
ticker = "AAL"
# JSON file with all the stock market data for AAL from the last 20 years
url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

# Save data to this file
file_to_save = 'stock_market_data-%s.csv'%ticker

# If you haven't already saved data,
# Go ahead and grab the data from the url
# And store date, low, high, volume, close, open values to a Pandas DataFrame
if not os.path.exists(file_to_save):
    with urllib.request.urlopen(url_string) as url:
        data = json.loads(url.read().decode())
        # extract stock market data
        data = data['Time Series (Daily)']
        df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
        for k,v in data.items():
            date = dt.datetime.strptime(k, '%Y-%m-%d')
            data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                        float(v['4. close']),float(v['1. open'])]
            df.loc[-1,:] = data_row
            df.index = df.index + 1
    print('Data saved to : %s'%file_to_save)
    df.to_csv(file_to_save)

# If the data is already there, just load it from the CSV
else:
    print('File already exists. Loading data from CSV')
    df = pd.read_csv(file_to_save)
df = df.sort_values('Date')
del df["Unnamed: 0"]      # deleting the column "Unamed:0"
df=df[["Date","Open","High","Low","Close"]]    # Rearranging the columns
print(df.iloc[:,1:2])
print("--------------------------------")
print(df.head())
print("--------------------------------")
print(df.tail())
print("--------------------------------")
print(df.columns.values)
print("--------------------------------")
print(df.shape)
print("--------------------------------")
# We will be predicting the Opening the price for the test data for easy visualisation and comaprison of predicted and actual values
training_values=df.iloc[0:3268,1:2].values
testing_values=df.iloc[3268:,1:2].values
print(training_values.shape)
print(testing_values.shape)
sc = MinMaxScaler(feature_range = (0, 1))
training_values_scaled = sc.fit_transform(training_values)
testing_values_scaled=sc.transform(testing_values)
def create_dataset(data,k):
    datax,datay=[],[]
    print("data shape is",data.shape[0])
    for i in range(data.shape[0]-k):                # creating the dataset for RNN, x will be a series of "k" values and y will be k+1 value
        x=data[i:i+k,0]
        y=data[i+k,0]
        datax.append(x)
        datay.append(y)
    return np.array(datax),np.array(datay)
train_x,train_y=create_dataset(training_values_scaled,60)
test_x,test_y=create_dataset(testing_values_scaled,60)
print("--------------------------------------------")
print(train_x.shape)
print(test_x.shape)
print(test_y.shape)
train_x=np.reshape(train_x,(train_x.shape[0],train_x.shape[1],1))
test_x=np.reshape(test_x,(test_x.shape[0],test_x.shape[1],1))
print("--------------------------------------------")
print(train_x.shape)
# Initialising the RNN
lstm = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
lstm.add(LSTM(units = 50, return_sequences = True, input_shape = (train_x.shape[1], 1)))
lstm.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
lstm.add(LSTM(units = 50, return_sequences = True))
lstm.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
lstm.add(LSTM(units = 50, return_sequences = True))
lstm.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
lstm.add(LSTM(units = 50))
lstm.add(Dropout(0.2))

# Adding the output layer
lstm.add(Dense(units = 1))

# Compiling the RNN
lstm.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
lstm.fit(train_x, train_y, epochs = 100, batch_size = 32)
predicted_stock_price = lstm.predict(test_x)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
real_stock_price=sc.inverse_transform(test_y.reshape(-1,1))        #reshaping to a 2D data
# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

