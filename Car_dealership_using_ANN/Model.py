import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import keras as kr

# import the data frame (csv)
def import_data(file, encode):
    return pd.read_csv(file, encoding = encode)

#Clean data by eliminating unnecessary columns such as name, email, age and country
def clean_data(df, tables_to_drop, output_column):
    df_clean = df.drop(tables_to_drop, axis = 1)
    df_output = df[output_column]
    return df_clean, df_output

#Normalize the input values with numbers between 0-1
def normalize_data(df_input, df_output):
    scaler = MinMaxScaler()
    df_input_normalized = scaler.fit_transform(df_input)
    df_output = df_output.values.reshape(-1,1) 
    df_output_normalized = scaler.fit_transform(df_output)
    return df_input_normalized, df_output_normalized

def train_data(df_input, df_output, dim, plot):
    #Dividing the data into training/test
    df_input_train, df_input_test, df_output_train, df_output_test = train_test_split(df_input, df_output, test_size=.2)

    #Create the model
    model = Sequential()
    model.add(Dense(25, input_dim = dim, activation = 'relu'))
    model.add(Dense(25, activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))

    #Training the model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    history = model.fit(df_input_train, df_output_train, epochs=100, batch_size = 25, verbose = 1, validation_split = 0.2)

    #Graph of performance of our model
    if plot:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model lost progress during training')
        plt.ylabel('Training and validation loss')
        plt.xlabel('Epoch number')
        plt.legend(['Training loss', 'Validation loss'])
        plt.show()

    return model

def training(file, encode):
    data_clients = import_data(file, encode)
    data_clients_clean, data_clients_output = clean_data(data_clients,['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], 'Car Purchase Amount')
    data_clients_clean_normalized, data_clients_output_normalized = normalize_data(data_clients_clean, data_clients_output)
    return train_data(data_clients_clean_normalized, data_clients_output_normalized, len(data_clients_clean.columns), True)

def main():
    trained_model = training('Car_Purchasing_Data.csv','ISO-8859-1')
    client = np.array([[1,35,60000,10000,600000]])
    prediction = trained_model.predict(client)
    print('Purchase amount of this customer is: ',prediction,'$')
    #Errases the saved model
    kr.backend.clear_session()

if __name__ == '__main__':
    main()


