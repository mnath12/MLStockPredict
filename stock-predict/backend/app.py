
from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS, cross_origin # CORS for handling Cross-Origin Resource Sharing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
# Make sure that you have all these libaries available to run the code successfully


# Create a Flask application instance
app = Flask(__name__)

# Enable CORS for all routes, allowing requests from any origin
CORS(app)
cors = CORS(app, resources={r"/*":{"origins":"http://localhost:5173"}})
sc = MinMaxScaler(feature_range=(0, 1))

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
        split_year = '2024-03-04'
        # change this to index by dates
        data = request.get_json()
        print("Data:", data)
        df = pd.DataFrame([data])
        
        # here we are seperating the data
        training_set = df[:split_year].iloc[:,1:2].values
        training_set_scaled = sc.fit_transform(training_set)
        test_set = df[str(int(split_year) + 1):].iloc[:,1:2].values

        X_train = []
        y_train = []
        for i in range(60, len(training_set_scaled)):
            X_train.append(training_set_scaled[i - 60:i, 0])
            y_train.append(training_set_scaled[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        regressor = Sequential()
        # First LSTM layer with Dropout regularisation
        regressor.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1],1)))
        regressor.add(Dropout(0.3))

        regressor.add(LSTM(units=80, return_sequences=True))
        regressor.add(Dropout(0.1))

        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))

        regressor.add(LSTM(units=30))
        regressor.add(Dropout(0.3))

        regressor.add(Dense(units=1))


        regressor.compile(optimizer='adam',loss='mean_squared_error')
        regressor.fit(X_train, y_train, epochs=50, batch_size=32)

        test_set = df[str(int(split_year) + 1):].iloc[:,1:2].values
        dataset_total = pd.concat((df["High"][:split_year],df["High"][str(int(split_year) + 1):]),axis=0)
        inputs = dataset_total[len(dataset_total)-len(test_set) - 60:].values
        inputs = inputs.reshape(-1,1)
        inputs  = sc.transform(inputs)

        # making the test data
        X_test = []
        for i in range(60,len(inputs)):
            X_test.append(inputs[i-60:i,0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        print("Prediction: ", predicted_stock_price)
        return jsonify({'Prediction': list(predicted_stock_price)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5173)