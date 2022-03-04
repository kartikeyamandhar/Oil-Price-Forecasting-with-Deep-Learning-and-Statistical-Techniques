#!/usr/bin/env python
# coding: utf-8

# In[178]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
import warnings
warnings.filterwarnings("ignore")


# In[179]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.seasonal import seasonal_decompose      
from pmdarima import auto_arima                              


# In[180]:


df=pd.read_csv('Book1.csv',thousands=',')


# In[181]:


df


# # Data Cleaning

# In[182]:


df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%d-%m-%Y')


# In[183]:


df['Date'] = pd.to_datetime(df['Date'])


# In[184]:


df=df.set_index('Date')


# In[185]:


df


# In[186]:


df.index


# In[187]:


df


# Checking if, in the above data all the dates between Feb 25, 2022 and Dec 12, 2011 are present or not.

# In[188]:


pd.date_range(start = '2011-12-15', end = '2011-12-13' ).difference(df.index)


# In[189]:


df.reindex(pd.date_range('2011-12-15', '2011-12-12')).isnull().all(1)


# As clearly seen, our data has all the dates between the above mentioned range.

# In[190]:


df.info()


# In[191]:


df=df.drop(columns=['Chg%'])


# In[192]:


df.columns


# In[193]:


df['Volume']=df['Volume'].astype(str)


# Removing 'K' and '.' from the Volume column

# In[194]:


df['Volume'] = df.apply(lambda x: x['Volume'][:-1], axis = 1) 


# In[195]:


df['Volume'] = df['Volume'].astype(str)
df['Volume']=pd.to_numeric(df['Volume'])


# In[196]:


df['Volume'] = df['Volume']*1000


# In[197]:


df=df.dropna()


# In[198]:


cols = df.columns
for col in cols:
    df[col]=df[col].astype(int)


# In[199]:


df


# In[200]:


df.info()


# In[201]:


df['Price'].plot()


# # Analysis and Pre-processing

# In[202]:


df.shape


# In[203]:


df['year'] = df.index.year
df['day_of_month'] = df.index.day
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month


# In[204]:


df


# In[205]:


sns.lineplot(x=df.index,y="Price",data=df)


# In[206]:


fig,(ax1, ax2)= plt.subplots(nrows=2)
fig.set_size_inches(10, 10)

sns.pointplot(data=df, x='year', y='Price', ax=ax1)
sns.pointplot(data=df, x='year', y='Volume', ax=ax2)


# # LSTM Modelling 

# In[207]:


from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# In[208]:


df.columns


# In[209]:


train_size = int(len(df) * 0.9)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))


# In[210]:


feature_cols = ['Open', 'High', 'Low', 'Volume']
feature_transformer = StandardScaler()
price_transformer = StandardScaler()

feature_transformer = feature_transformer.fit(train[feature_cols].to_numpy())
price_transformer = price_transformer.fit(train[['Price']])

train.loc[:, feature_cols] = feature_transformer.transform(train[feature_cols].to_numpy())
train['Price'] = price_transformer.transform(train[['Price']])

test.loc[:, feature_cols] = feature_transformer.transform(test[feature_cols].to_numpy())
test['Price'] = price_transformer.transform(test[['Price']])


# In[211]:


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


# In[212]:


time_steps = 10

X_train, y_train = create_dataset(train, train.Price, time_steps)
X_test, y_test = create_dataset(test, test.Price, time_steps)

print(X_train.shape, y_train.shape)


# In[213]:


model = keras.Sequential()
model.add(
    keras.layers.LSTM(128,activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True)
)
model.add(
    keras.layers.LSTM(50,activation='relu',return_sequences=False)
)

model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[214]:


history = model.fit(
    X_train, y_train, 
    epochs=25, 
    batch_size=10, 
    validation_split=0.1,
    verbose=1,
    shuffle=True
)


# In[215]:


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend();


# In[216]:


y_pred = model.predict(X_test)


# In[217]:


y_train_inv = price_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = price_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = price_transformer.inverse_transform(y_pred)


# In[218]:


plt.plot(y_test_inv.flatten(), marker='.', label="true")
plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Price')
plt.xlabel('Time Step')
plt.legend()
plt.show();


# In[219]:


from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s
from numpy import sqrt 


# In[220]:


mae1=mae(y_test_inv.reshape(-1,1),y_pred_inv)
mse1=mse(y_test_inv.reshape(-1,1),y_pred_inv)
rmse1=sqrt(mse1)


# In[221]:


mae1,rmse1,mse1


# In[222]:


np.mean(y_test_inv)


# In[223]:


df


# # Considering the whole dataframe for training and forecasting

# In[224]:


import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s
from math import sqrt
import os


# In[225]:


dfOil=df
dfOilPrice=dfOil['Price'].astype(float)


# In[226]:


scaler=StandardScaler()
scaler=scaler.fit(dfOilPrice.values.reshape(-1, 1))
dfOilPriceScaled=scaler.transform(dfOilPrice.values.reshape(-1, 1))


# In[227]:


oilPX=[]
oilPY=[]


# In[228]:


nextPrd=1
windSz=14

for i in range(windSz, len(dfOilPriceScaled)-nextPrd+1):    
    oilPX.append(dfOilPriceScaled[i-windSz:i])
    oilPY.append(dfOilPriceScaled[i+nextPrd-1:i+nextPrd,0])

oilPX,oilPY=np.array(oilPX),np.array(oilPY)


# In[229]:


print('dfOil shape= {}.'.format(dfOil.shape))
print('dfOilPriceScaled shape= {}.'.format(dfOilPriceScaled.shape))
print('oilPX shape== {}.'.format(oilPX.shape))
print('oilPY shape== {}.'.format(oilPY.shape))


# In[230]:


model=Sequential()
model.add(LSTM(100, activation='relu', input_shape=(oilPX.shape[1], oilPX.shape[2]), return_sequences=True))
model.add(LSTM(50, activation= 'relu',return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(oilPY.shape[1]))
model.compile(optimizer='adam',loss='mse')
history=model.fit(oilPX,oilPY,epochs=25,batch_size=16, validation_split=0.1,verbose=1)


# In[231]:


fig, ax = plt.subplots(figsize=(15, 5))
fig.subplots_adjust(bottom=0.15, left=0.2)
ax.plot(range(0,len(history.history['loss'])),history.history['loss'],color='blue',label='loss')
ax.plot(range(0,len(history.history['val_loss'])),history.history['val_loss'],color='red',label='val_loss')
ax.set_xlabel('loss')
ax.set_ylabel('val loss')
plt.legend()
plt.show()


# In[232]:


forecastD=60

predictedForDays=model.predict(oilPX[-forecastD:])
actualOilPForDays=dfOil['Price'][-forecastD:]
predictedForDaysInn=scaler.inverse_transform(predictedForDays)


# In[233]:


fig, ax = plt.subplots(figsize=(15, 5))
fig.subplots_adjust(bottom=0.15, left=0.2)
ax.plot(range(0,len(predictedForDaysInn)),dfOil['Price'][-forecastD:],color='blue',label='True Price of data')
ax.plot(range(0,len(predictedForDaysInn)),predictedForDaysInn,color='red',label='predicted')
ax.set_xlabel('true price')
ax.set_ylabel('predicted')
plt.legend()
plt.show()


# In[234]:


fig, ax = plt.subplots(figsize=(15, 5))
fig.subplots_adjust(bottom=0.15, left=0.2)
ax.plot(range(0,len(dfOil)-forecastD),dfOil['Price'][0:len(dfOil)-forecastD],color='blue',label='True Price of data')
ax.plot(range(len(dfOil)-forecastD,len(dfOil)),predictedForDaysInn,color='red',label='predicted')
ax.set_xlabel('true price')
ax.set_ylabel('predicted')
plt.legend()
plt.show()


# In[235]:


mae1=mae(actualOilPForDays,predictedForDaysInn)
mse1=mse(actualOilPForDays,predictedForDaysInn)
print(mse1)
rmse1=sqrt(mse1)
r2_score=r2s(actualOilPForDays,predictedForDaysInn)
print(r2_score)
n=forecastD
k=1
adjr=1-(((1-r2_score)*(n-1))/(n-k-1))
print(adjr)


# # Statistical Analysis

# In[236]:


df


# In[237]:


df.info()


# In[238]:


df['Price'].resample('Y').mean().plot.bar(title='Yearly average oil prices')


# In[239]:


df['Price'].resample('Y').plot()


# In[240]:


df['Price'].resample('Y').plot()
df.rolling(window=150).mean()['Price'].plot(color='black');


# Along the black line, we have found the rolling mean of 150 days, so as to generalise the trend of oil prices.

# In[241]:


df['Price'].expanding(min_periods=150).mean().plot(figsize=(12,5));


# Expanding upon the data and finding average trends of continuous 150 days.

# ETS Decompoition

# We apply an additive model when it seems that the trend is more linear and the seasonality and trend components seem to be constant over time. A multiplicative model is more appropriate when we are increasing (or decreasing) at a non-linear rate.

# In[242]:


df[df<5]=1


# In[243]:


from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df['Price'].values, model='multiplicative',period=12)  # model='mul' also works
result.plot();


# We can see a strong seasonal component 

# Simple Moving Average (SMA)

# In[244]:


df['6-month-SMA'] = df['Price'].rolling(window=6).mean()
df['12-month-SMA'] = df['Price'].rolling(window=12).mean()


# In[245]:



df['12-month-SMA'].plot(figsize=(15,5),legend=True)


# We just showed how to calculate the SMA based on some window. However, basic SMA has some weaknesses:
# 
# Smaller windows will lead to more noise, rather than signal
# It will always lag by the size of the window
# It will never reach to full peak or valley of the data due to the averaging.
# Does not really inform you about possible future behavior, all it really does is describe trends in your data.
# Extreme historical values can skew your SMA significantly

# EWMA will allow us to reduce the lag effect from SMA and it will put more weight on values that occured more recently (by applying more weight to the more recent values, thus the name). The amount of weight applied to the most recent values will depend on the actual parameters used in the EWMA and the number of periods given a window size. 

# In[246]:


df['EWMA12']=df['Price'].ewm(span=12,adjust=True).mean()


# In[247]:


df['EWMA12'].plot()


# In[248]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[249]:


df['TESadd12'] = ExponentialSmoothing(df['Price'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues


# In[250]:


df['TESmul12'] = ExponentialSmoothing(df['Price'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues


# In[251]:


df[['TESadd12','TESmul12']].iloc[:150].plot(figsize=(12,6)).autoscale(axis='x',tight=True);


# In[252]:


df


# In[253]:


df2=df


# In[254]:


df2


# In[255]:


df2=df2.drop(columns=['year', 'day_of_month',
       'day_of_week', 'month', '6-month-SMA', '12-month-SMA', 'EWMA12',
       'TESadd12', 'TESmul12'])


# In[256]:


df2.columns


# In[257]:


df2


# In[258]:


df3=df2.resample('W').mean().ffill()


# In[259]:


df3.index


# In[260]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


# In[261]:


df3=df3[:'2022-02-27']


# In[262]:


df3


# In[263]:


len(df3)


# # FORECASTING 

# Forecasting with the Holt-Winters Method

# In[264]:


len(df3)


# In[265]:


df3=df3.sort_index()


# In[266]:


for i in df3.columns:
    df3[i]=df3[i].fillna(int(df[i].mean()))


# In[267]:


train_data = df3.iloc[:500] 
test_data = df3.iloc[500:]


# In[268]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

fitted_model = ExponentialSmoothing(train_data['Price'],trend='mul',seasonal='mul',seasonal_periods=12).fit()


# In[269]:


test_predictions = fitted_model.forecast(33).rename('HW Forecast')


# In[270]:


train_data['Price'].plot(legend=True,label='TRAIN')
test_data['Price'].plot(legend=True,label='TEST',figsize=(12,8))


# In[271]:


test_data['Price'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True)


# In[272]:


test_predictions.plot()


# In[273]:


train_data['Price'].plot(legend=True,label='TRAIN')
test_data['Price'].plot(legend=True,label='TEST',figsize=(12,8))
test_predictions.plot(legend=True,label='PREDICTION')


# In[274]:


from statsmodels.tools.eval_measures import rmse

error = rmse(test_data['Price'], test_predictions)
print(f'RMSE Error: {error:11.10}')


# Stationarity or Not

# In[275]:


import warnings
warnings.filterwarnings("ignore")


# In[276]:



from statsmodels.tsa.stattools import acovf,acf,pacf,pacf_yw,pacf_ols


# In[277]:


from pandas.plotting import lag_plot

lag_plot(df3['Price']);


# In[106]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[107]:


acf(df3['Price']).shape


# In[108]:



title = 'Autocorrelation: Price'
lags = 40
plot_acf(df3['Price'],title=title,lags=lags);


# This plot indicates non-stationary data, as there are a large number of lags before ACF values drop off.

# In[109]:


title='Partial Autocorrelation: Price'
lags=40
plot_pacf(df3['Price'],title=title,lags=lags);


# # AR(p) AutoRegressive Model

# In[110]:


from statsmodels.tsa.ar_model import AR,ARResults,AutoReg


# In[111]:


train = df.iloc[:500]
test = df.iloc[500:]


# # Fitting AR(1) Model

# In[282]:


model = AutoReg(train['Price'],lags=1)
AR1fit = model.fit()


# In[283]:



start=len(train)
end=len(train)+len(test)-1
predictions1 = AR1fit.predict(start=start, end=end, dynamic=False).rename('AR(1) Predictions')


# In[284]:


predictions1


# Fit an AR(p) model where statsmodels chooses p

# In[285]:


ARfit = model.fit()

print(f'Coefficients:\n{ARfit.params}')


# In[286]:


start = len(train)
end = len(train)+len(test)-1
rename = f'AR(11) Predictions'

predictions11 = ARfit.predict(start=start,end=end,dynamic=False).rename(rename)


# In[287]:


from sklearn.metrics import mean_squared_error

labels = ['AR(1)','AR(11)']
preds = [predictions1,  predictions11]  # these are variables, not strings!

for i in range(2):
    error = mean_squared_error(test['Price'], preds[i])
    print(f'{labels[i]} Error: {error:11.10}')


# # Augmented Dickey-Fuller Test

# In[118]:


from statsmodels.tsa.stattools import ccovf,ccf
from statsmodels.tsa.stattools import adfuller,kpss,coint,bds,q_stat,grangercausalitytests,levinson_durbin
from statsmodels.tools.eval_measures import mse, rmse, meanabs


# In[119]:


from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


# In[120]:


adf_test(df3['Price'])


# # ARIMA

# In[121]:


from pmdarima import auto_arima

import warnings
warnings.filterwarnings("ignore")


# In[122]:


df3


# In[123]:


len(df3)


# In[124]:


auto_arima(df3['Price'])


# In[125]:


auto_arima(df3['Price'],error_action='ignore').summary()


# In[126]:


stepwise_fit = auto_arima(df3['Price'], start_p=1, start_q=1,
                          max_p=3, max_q=3, m=12,
                          start_P=0, seasonal=True,
                          d=None, D=1, trace=True,
                          error_action='ignore',   # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=True)           # set to stepwise

stepwise_fit.summary()


# In[127]:


from statsmodels.tsa.statespace.tools import diff
df3['d1'] = diff(df3['Price'],k_diff=1)
adf_test(df3['d1'],'Diff Price 1')


# This confirms that we reached stationarity after the first difference.

# In[128]:


len(df3)


# In[129]:


train = df3.iloc[:500]
test = df3.iloc[500:]


# In[130]:


import statsmodels.api as sm


# In[131]:


results = sm.tsa.statespace.SARIMAX(train['Price'],order=(5,1,12),
                                enforce_stationarity=False, enforce_invertibility=False).fit()
results.summary()


# In[132]:


# Obtain predicted values
start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('SARIMA Predictions')


# In[133]:


test['Price'].plot(legend=True,figsize=(12,6))
predictions.plot(legend=True)


# In[134]:


from sklearn.metrics import mean_squared_error

error = mean_squared_error(test['Price'], predictions)


# In[135]:


error = np.sqrt(error)
print(error)


# In[291]:


df=df.sort_index()


# In[292]:


df


# In[ ]:





# In[293]:


df=df.asfreq('d').ffill()


# In[301]:


df.index


# In[302]:


len(df)


# In[303]:


ax = df['Price'].plot(figsize=(12,6))
ax.autoscale(axis='x',tight=True)


# In[304]:


result = seasonal_decompose(df['Price'], model='mul')
result.plot();


# In[305]:


df


# In[306]:


len(df)


# In[307]:


df.index


# In[308]:


df4=df.asfreq('MS')


# In[309]:


df4


# In[310]:


len(df4)


# In[311]:


df4['Price'].plot()


# In[166]:


df4


# In[167]:


df4=df4.drop(columns=['year', 'day_of_month',
       'day_of_week', 'month', '6-month-SMA', '12-month-SMA', 'EWMA12',
       'TESadd12', 'TESmul12'])


# In[168]:


result = seasonal_decompose(df4['Price'], model='mul')
result.plot();


# In[169]:


train = df4.iloc[:125]
test = df4.iloc[125:]


# In[170]:


stepwise_fit = auto_arima(df4['Price'], start_p=0, start_q=0,
                          max_p=2, max_q=2, m=7,
                          start_P=0, seasonal=True,
                          d=None, D=1, trace=True,
                          error_action='ignore',   # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=False)           # set to stepwise

stepwise_fit.summary()


# In[171]:


model = sm.tsa.statespace.SARIMAX(train['Price'],order=(0,1,2),seasonal_order=(2,1,0,7))
results = model.fit()
results.summary()


# In[172]:



start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('SARIMA Predictions')


# In[173]:


ax = test['Price'].plot(legend=True,figsize=(12,6))
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)


# In[174]:


from statsmodels.tools.eval_measures import rmse

error = rmse(test['Price'], predictions)
print(f'RMSE Error: {error:11.10}')


# In[175]:


model = sm.tsa.statespace.SARIMAX(df4['Price'],order=(0,1,2),seasonal_order=(2,1,0,7))
results = model.fit()
fcast = results.predict(len(df4),len(df4)+7,typ='levels').rename('Forecast')


# In[176]:


ax = df4['Price'].plot(legend=True,figsize=(12,6))
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)

