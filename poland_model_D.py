import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statistics import mean
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras import regularizers
import numpy as np 
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import load_model


#csv reading and date parsing
dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d')
poland_covid= pd.read_csv("poland_covid.csv", parse_dates=['DATE_REPORTED'],date_parser=dateparse)
norway_covid= pd.read_csv("norway_covid.csv", parse_dates=['DATE_REPORTED'],date_parser=dateparse)
france_covid= pd.read_csv("france_covid.csv", parse_dates=['DATE_REPORTED'],date_parser=dateparse)


poland_covid.index = pd.DatetimeIndex(poland_covid.DATE_REPORTED)
norway_covid.index = pd.DatetimeIndex(norway_covid.DATE_REPORTED)
france_covid.index = pd.DatetimeIndex(france_covid.DATE_REPORTED)


#new date related columns creation
poland_covid["MONTH"] = pd.DatetimeIndex(poland_covid['DATE_REPORTED']).month
poland_covid["DAY_OF_YEAR"] = pd.DatetimeIndex(poland_covid['DATE_REPORTED']).dayofyear
poland_covid["DAY_OF_MONTH"] = pd.DatetimeIndex(poland_covid['DATE_REPORTED']).day
poland_covid["DAY_OF_WEEK"] = pd.DatetimeIndex(poland_covid['DATE_REPORTED']).dayofweek

norway_covid["MONTH"] = pd.DatetimeIndex(norway_covid['DATE_REPORTED']).month
norway_covid["DAY_OF_YEAR"] = pd.DatetimeIndex(norway_covid['DATE_REPORTED']).dayofyear
norway_covid["DAY_OF_MONTH"] = pd.DatetimeIndex(norway_covid['DATE_REPORTED']).day
norway_covid["DAY_OF_WEEK"] = pd.DatetimeIndex(norway_covid['DATE_REPORTED']).dayofweek

france_covid["MONTH"] = pd.DatetimeIndex(france_covid['DATE_REPORTED']).month
france_covid["DAY_OF_YEAR"] = pd.DatetimeIndex(france_covid['DATE_REPORTED']).dayofyear
france_covid["DAY_OF_MONTH"] = pd.DatetimeIndex(france_covid['DATE_REPORTED']).day
france_covid["DAY_OF_WEEK"] = pd.DatetimeIndex(france_covid['DATE_REPORTED']).dayofweek




poland_covid=poland_covid.drop(columns='DATE_REPORTED')
norway_covid=norway_covid.drop(columns='DATE_REPORTED')
france_covid=france_covid.drop(columns='DATE_REPORTED')

#generating vectors with val from prev days
def generate_t_series(input, value_num):
    input=input.to_numpy()
    new_deaths = input[:,4]
    generator = TimeseriesGenerator(new_deaths,new_deaths, 
                               length=value_num,
                               batch_size=len(new_deaths))
    global_index = value_num
    i, t = generator[0]
    for b_row in range(len(t)):
        assert(abs(t[b_row] - new_deaths[global_index]) <= 0.001)          
        global_index += 1
    return i, t
              



X, Y = generate_t_series(poland_covid, 7)
Y = Y.reshape((Y.shape[0],1))


vector_new_deaths=np.concatenate((X, Y), axis=1)

parameters_X=poland_covid[[
"DAY_OF_YEAR",
"DAY_OF_MONTH",
"DAY_OF_WEEK",
"TOTAL_VACCINATIONS",
"PEOPLE_VACCINATED",
"PEOPLE_FULLY_VACCINATED",
"TOTAL_BOOSTERS"]]

#offset i konwersja do numpy w int
parameters_X=parameters_X.iloc[7:,:].to_numpy().astype(int)

#concat, dzien liczony na koncu
vector_new_deaths=np.concatenate((parameters_X, vector_new_deaths), axis=1)

np.set_printoptions(threshold=np.inf)

#print(vector_new_deaths)
X=vector_new_deaths[:,:-1]

scaler_filename = "scaler"
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(vector_new_deaths)
X_scaled=scaled
size=X_scaled.shape[1]
Y=scaled[:,size-1]
X=scaled[:,:-1]
size=X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, shuffle=True)
X_train = np.reshape(X_train, (-1, 1, size))
X_test = np.reshape(X_test, (-1, 1, size))

n_r2_adjusted = X_train.shape[0]
p_r2_adjusted = X_train.shape[2]


# NEURAL NETWORK
model = Sequential()
model.add(LSTM(64,input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(64))
#model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adagrad')
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500)

#training
training = model.fit(X_train, y_train, epochs=5000 , batch_size=5, validation_split=0.2, verbose=2, callbacks=[early_stopping])

#predictions for test data
#model=load_model("covid_test_2k.h5")
predictions= model.predict(X_test)

plt.figure(0)
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('Funkcja straty')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()


#reshaping and inversing normalization
prediction_reshaped = np.zeros((len(predictions), size+1))
testY_reshaped = np.zeros((len(y_test), size+1))

prediction_r = np.reshape(predictions, (len(predictions),))
testY_r = np.reshape(y_test, (len(y_test),))

prediction_reshaped[:,size] = prediction_r
testY_reshaped[:,size] = testY_r

prediction_inversed = scaler.inverse_transform(prediction_reshaped)[:,size]
testY_inversed = scaler.inverse_transform(testY_reshaped)[:,size]


#calculating error rates
mse = mean_squared_error(testY_inversed, prediction_inversed)
mae=mean_absolute_error(testY_inversed, prediction_inversed)
r2=r2_score(testY_inversed,prediction_inversed) 
adjusted_r2 = 1 - ((1 - r2) * (n_r2_adjusted - 1)) / ( n_r2_adjusted - p_r2_adjusted - 1)
mape_err=mean(np.abs((testY_inversed - prediction_inversed) / testY_inversed)) * 100


model.output_shape 

#model.summary()
#model.get_config()
#model.get_weights() 
#saving model
#model.save('covid_test_2k.h5')


plt.figure(1)
plt.subplot(2,1,1)
plt.plot(prediction_inversed[1:100], label='prognozy')
plt.plot(testY_inversed[1:100], label='wartości rzeczywiste')
plt.title('Prognozy w porównaniu z ilością zgonów', fontdict={'fontsize': 18, 'fontweight': 'medium'})
plt.xlabel('próbki 1-100', fontsize=15)
plt.ylabel('ilość zgonów', fontsize=15)
plt.subplot(2,1,2)
plt.plot(prediction_inversed[101:200], label='prognozy')
plt.plot(testY_inversed[101:200], label='wartości rzeczywiste')
plt.title('  ', fontdict={'fontsize': 12, 'fontweight': 'medium'})
plt.xlabel('próbki 101-200', fontsize=15)
plt.ylabel('ilość zgonów', fontsize=15)
plt.legend(loc='upper left', fontsize=13)
plt.show()


print('Test MSE: %.3f' % mse)
print('Test MAE: %.3f' % mae)
print('R^2: %.3f' % r2)
print('adjR^2: %.3f' % adjusted_r2)