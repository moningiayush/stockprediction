import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2023-07-15'

st.title("Stock Trend Prediction")
user_input = st.text_input('Enter stock ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)
df.head()

#Describing Data
st.subheader("Data from 2010 - 2023")
st.write(df.describe())

#visvalizations
st.subheader("Closing price vs Time chart")
fig = plt.figure(figsize = (12,6))
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader("Closing price vs Time chart with 100 days")
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader("Closing price vs Time chart with 100 and 200 days")
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma100,'g')
plt.plot(df['Close'])
st.pyplot(fig)

data_training = df['Close'][0:int(len(df)*0.70)].to_frame()
data_testing = df['Close'][int(len(df)*0.70):].to_frame()

print(data_training)
print(data_testing)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#load my model 
model = load_model(r'C:\Users\Monin\OneDrive\Desktop\future bull\keras_model.hs')

#testing
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test =[]
y_test = []
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])
x_test , y_test = np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor
#final draft
st.subheader("Predicition vs original")
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='orginal price')
plt.plot(y_predicted,'r',label='predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)