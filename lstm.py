import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def prepare_lstm_sequences(features_df: pd.DataFrame, window_size: int = 60, shift: int = 22):
    sequences, targets = [], []
    for ticker, df_t in features_df.groupby('Ticker'):
        df_t = df_t.sort_values('Date')
        X = df_t.drop(['Date', 'Ticker', 'target'], axis=1).values
        y = df_t['target'].values
        for i in range(window_size, len(X) - shift + 1):
            seq = X[i - window_size:i]
            label = y[i + shift - 1]
            sequences.append(seq)
            targets.append(label)
    return np.array(sequences), np.array(targets)

if __name__ == "__main__":
    from forecast_jpx import features, price_matrix, load_exogenous, load_fundamentals, codes
    cds = codes()
    FUNDAMENTALS = load_fundamentals(cds)
    mat = price_matrix(cds)
    start_date = mat.index.min().strftime("%Y-%m-%d")
    end_date = mat.index.max().strftime("%Y-%m-%d")
    EXOG_DF = load_exogenous(["JPY=X", "^N225", "^VIX", "^TOPX"], start_date, end_date, "1d")
    features_df = features(mat, FUNDAMENTALS, EXOG_DF)

    X_seq, y_seq = prepare_lstm_sequences(features_df, window_size=60, shift=22)
    nsamples, ntimesteps, nfeatures = X_seq.shape
    X_flat = X_seq.reshape(-1, nfeatures)
    scaler = StandardScaler().fit(X_flat)
    X_scaled = scaler.transform(X_flat).reshape(nsamples, ntimesteps, nfeatures)

    split = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(ntimesteps, nfeatures)))
    model.add(LSTM(64))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.1,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=0)

    preds = model.predict(X_test)
    print(f"Test MSE: {mean_squared_error(y_test, preds):.6f}")
    print(f"Test R2 : {r2_score(y_test, preds):.4f}")