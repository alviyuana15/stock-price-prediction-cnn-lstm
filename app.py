from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
from werkzeug.utils import secure_filename
from scipy.stats import norm, zscore
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import pandas as pd
import os

# Load model
model = tf.keras.models.load_model('model/stock_cnn_lstm_v2.h5', compile=False)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app,resources={r"/*":{"origins":"*"}})
socketio = SocketIO(app,cors_allowed_origins="*")

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv'}

def cornish_fisher_var(data, alpha=0.05):
    data = data.dropna()
    data = data.clip(lower=data.quantile(0.01), upper=data.quantile(0.99))

    mean_return = data.mean()
    std_dev = data.std()
    skewness = data.skew()
    kurtosis = data.kurtosis()

    z_alpha = norm.ppf(1 - alpha)
    cf_correction = z_alpha + (z_alpha**2 - 1) * skewness / 6 + (z_alpha**3 - 3*z_alpha) * kurtosis / 24 - (2*z_alpha**3 - 5*z_alpha) * (skewness**2) / 36

    var = mean_return + std_dev * cf_correction
    return var

def process_file_risk(filepath):
    df = pd.read_csv(filepath)
    returns = df[['Open_Predicted', 'High_Predicted', 'Low_Predicted', 'Close_Predicted']].pct_change().dropna()
    returns_winsor = returns.clip(lower=returns.quantile(0.01, axis=0),
                                upper=returns.quantile(0.99, axis=0),
                                axis=1)
    mean_vals = returns_winsor.mean()
    std_vals = returns_winsor.std()
    skew_vals = returns_winsor.skew()
    kurt_vals = returns_winsor.kurtosis()
    z_95 = norm.ppf(1 - 0.05)
    z_99 = norm.ppf(1 - 0.01)

    # Cornish-Fisher Correction
    cf_95 = (
        z_95
        + (z_95**2 - 1) * skew_vals / 6
        + (z_95**3 - 3 * z_95) * kurt_vals / 24
        - (2 * z_95**3 - 5 * z_95) * (skew_vals**2) / 36
    )

    cf_99 = (
        z_99
        + (z_99**2 - 1) * skew_vals / 6
        + (z_99**3 - 3 * z_99) * kurt_vals / 24
        - (2 * z_99**3 - 5 * z_99) * (skew_vals**2) / 36
    )

    var_95 = (mean_vals - cf_95 * std_vals)
    var_99 = (mean_vals - cf_99 * std_vals)

    data_final = {}
    for col in returns_winsor.columns:
        data = returns_winsor[col]
        var95 = (var_95[col])
        var99 = (var_99[col])
        data_jittered = data + np.random.normal(0, 0.0005, size=len(data))
        counts, bin_edges = np.histogram(data_jittered, bins=50)
        data_final[col] = {
            "hist": {
                "bin_edges": bin_edges.tolist(), # Convert numpy array to list
                "counts": counts.tolist()        # Convert numpy array to list
            },
            "var_95": var95,
            "var_99": var99
        }

    return data_final

def process_file(filepath):
    df = pd.read_csv(filepath)
    df = df[['Open', 'High', 'Low', 'Close']]
    split_index = int(len(df) * 0.8)
    df_train = df.iloc[:split_index]
    df_test  = df.iloc[split_index:]

    scaler = MinMaxScaler()
    scaler.fit(df_train[['Open', 'High', 'Low', 'Close']])

    train_scaled = scaler.transform(df_train[['Open', 'High', 'Low', 'Close']])
    test_scaled  = scaler.transform(df_test[['Open', 'High', 'Low', 'Close']])

    sequence_length = 10
    X_train, y_train = create_sliding_window(train_scaled, sequence_length)
    X_test, y_test   = create_sliding_window(test_scaled, sequence_length)
    
    min_train = min(len(X_train), len(y_train))
    X_train = X_train[:min_train]
    y_train = y_train[:min_train]

    min_test = min(len(X_test), len(y_test))
    X_test = X_test[:min_test]
    y_test = y_test[:min_test]
    y_pred = model.predict(X_test)

    # Future prediction
    n_days = 7

    last_window = X_test[-1].copy()
    future_preds = []

    for _ in range(n_days):
        input_window = last_window.reshape(1, last_window.shape[0], last_window.shape[1])
        next_pred = model.predict(input_window, verbose=0)[0]
        future_preds.append(next_pred)
        last_window = np.vstack([last_window[1:], next_pred])

    future_preds = np.array(future_preds)
    df_future = pd.DataFrame(future_preds, columns=['Open', 'High', 'Low', 'Close'])

    feature_names = ['Open', 'High', 'Low', 'Close']
    result = {}
    for i in range(4):
        single_result = {
            "feature_names": feature_names[i],
            "y_test": y_test[:, i].tolist(),
            "y_pred": y_pred[:, i].tolist(),
            "y_future": df_future[feature_names[i]].tolist()
        }
        result[feature_names[i]] = single_result

    print(df_future)
    return result

def create_sliding_window(data, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        data = process_file(save_path)
        return jsonify({'message': f'File "{filename}" uploaded successfully', 'data': data}), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/risk', methods=['POST'])
def upload_file_risk():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        data_final = process_file_risk(save_path)
        return jsonify({'message': f'File "{filename}" uploaded successfully', 'data':data_final}), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    socketio.run(app, debug=True,port=5000)