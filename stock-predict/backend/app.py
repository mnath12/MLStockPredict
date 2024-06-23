
from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS, cross_origin # CORS for handling Cross-Origin Resource Sharing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split
# Make sure that you have all these libaries available to run the code successfully


# Create a Flask application instance
app = Flask(__name__)

# Enable CORS for all routes, allowing requests from any origin
CORS(app)
cors = CORS(app, resources={r"/*":{"origins":"http://localhost:5173"}})

# Define a route for handling HTTP GET requests to the root URL
@app.route('/', methods=['GET'])
def get_data():
    data = {
        "message":"API is Running"
    }
    return jsonify(data)
  
# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # In theory, upon getting the CSV data from request get JSON, this should 
        # run all the stuff in the notebook and give the prediction
        # Might want to separate out training somehow but idk how
        # If we keep training on the same page -> implement progress bar on GUI

        # change this to index by dates
        data = request.get_json()

        # Python program to demonstrate
        # writing to file
        
        # Opening a file
        file1 = open('myfile.csv', 'w')
       
        # Writing a string to file
        file1.write(data)
        
        
        # Closing file
        file1.close()
        

        # #print("Data:", data)
        # df = pd.read_csv('myfile.csv', index_col='Date', parse_dates=["Date"])

        # #print("Dataframe:", df)
        # n=df.size
        # print(df)
        # i = int(n/2)
        # sc = MinMaxScaler(feature_range=(0, 1))

        # keys = df.index.values
        # print(len(keys))
        # split_value = keys[0]

        # # here we are seperating the data
        # training_set = df[:keys[60]].iloc[:,1:2].values
        # test_set = df[keys[61]:].iloc[:,1:2].values
        # #training_set, test_set = train_test_split(df, test_size=0.2)
        # training_set_scaled = sc.fit_transform(training_set)

        # X_train = []
        # y_train = []
        # for i in range(5, len(training_set_scaled)):
        #     X_train.append(training_set_scaled[i - 5:i, 0])
        #     y_train.append(training_set_scaled[i, 0])


        # X_train, y_train = np.array(X_train), np.array(y_train)
        # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # regressor = Sequential()
        # # First LSTM layer with Dropout regularisation
        # regressor.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1],1)))
        # regressor.add(Dropout(0.3))

        # regressor.add(LSTM(units=80, return_sequences=True))
        # regressor.add(Dropout(0.1))

        # regressor.add(LSTM(units=50, return_sequences=True))
        # regressor.add(Dropout(0.2))

        # regressor.add(LSTM(units=30))
        # regressor.add(Dropout(0.3))

        # regressor.add(Dense(units=1))


        # regressor.compile(optimizer='adam',loss='mean_squared_error')
        # regressor.fit(X_train, y_train, epochs=50, batch_size=32)

        # dataset_total = pd.concat((df["High"][:keys[60]],df["High"][keys[61]:]),axis=0)
        # print(dataset_total)
        # inputs = dataset_total[len(dataset_total)-len(test_set) - 5:].values
        # inputs = inputs.reshape(-1,1)
        # inputs  = sc.transform(inputs)

        # # making the test data
        # X_test = []
        # for i in range(5,len(inputs)):
        #     X_test.append(inputs[i-5:i,0])
        # X_test = np.array(X_test)
        # X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        # predicted_stock_price = regressor.predict(X_test)
        # predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        # predicted_stock_price.flatten()
        memory = 1
        df = pd.read_csv('myfile.csv', index_col='Date', parse_dates=["Date"])

        print("Dataframe:", df)

        sc = MinMaxScaler(feature_range=(0, 1))

        keys = df.index.values
        n = len(keys)
        print("Number of data points: ", n)
        train_percentage = .8
        train_index = int(.8*n)
        split_value = keys[train_index]


        # here we are seperating the data
        training_set = df[:split_value].iloc[:,1:2].values
        test_set = df[split_value:].iloc[:,1:2].values
        #training_set, test_set = train_test_split(df, test_size=0.2)
        training_set_scaled = sc.fit_transform(training_set)

        X_train = []
        y_train = []
        for i in range(memory, len(training_set_scaled)):
            X_train.append(training_set_scaled[i - memory:i, 0])
            y_train.append(training_set_scaled[i, 0])


        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        regressor = Sequential()
        # First LSTM layer with Dropout regularisation
        regressor.add(LSTM(units=400, return_sequences=True, input_shape=(X_train.shape[1],1)))
        regressor.add(Dropout(0.5))

        regressor.add(LSTM(units=383, return_sequences=True))
        regressor.add(Dropout(0.2))

        regressor.add(LSTM(units=200, return_sequences=True))
        regressor.add(Dropout(0.3))

        regressor.add(LSTM(units=120))
        regressor.add(Dropout(0.27))

        regressor.add(Dense(units=4))


        regressor.compile(optimizer='adam',loss='mean_squared_error')
        regressor.fit(X_train, y_train, epochs=100, batch_size=8)

        dataset_total = pd.concat((df["High"][:split_value],df["High"][split_value:]),axis=0)
        print(dataset_total)
        inputs = dataset_total[len(dataset_total)-len(test_set) - memory:].values
        inputs = inputs.reshape(-1,1)
        inputs  = sc.transform(inputs)

        # making the test data
        X_test = []
        for i in range(memory,len(inputs)):
            X_test.append(inputs[i-memory:i,0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        print("Prediction: ", predicted_stock_price)
        return jsonify({'Prediction': predicted_stock_price[:,0].tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5173)