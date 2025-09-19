# -*- coding: utf-8 -*-
"""
Adapted for external test set generalization analysis
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.metrics import r2_score, mean_absolute_error
from keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import gc
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ------------------ Create sequences ---------------------
def create_sequences(X, y, seq_length=10):
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
        X_seq.append(X[i-seq_length:i])
        y_seq.append(y.iloc[i])
    return np.array(X_seq), np.array(y_seq)

# ------------------ Feature Engineering ------------------
def feature_engineering(df, mode="basic", scaler=None, encoder=None, fit_scaler=True, fit_encoder=True):
    predictor_cat = df[["Predictor"]].copy()
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    if fit_encoder:
        encoded_predictor = encoder.fit_transform(predictor_cat)
    else:
        encoded_predictor = encoder.transform(predictor_cat)
    encoded_df = pd.DataFrame(encoded_predictor, columns=encoder.get_feature_names_out(["Predictor"]))

    df = df.drop(columns=["Predictor"])
    df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    sensor_columns = ["TFSI", "NCN", "Ni", "Fe"]
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        df[sensor_columns] = scaler.fit_transform(df[sensor_columns])
    else:
        df[sensor_columns] = scaler.transform(df[sensor_columns])

    if mode == "basic":
        return df, scaler, encoder

    if mode == "derivatives":
        for sensor in sensor_columns:
            df[f"deriv_{sensor}"] = df[sensor].diff().fillna(0)
        return df, scaler, encoder

    if mode == "ratios":
        for i in range(len(sensor_columns)):
            for j in range(i + 1, len(sensor_columns)):
                a, b = sensor_columns[i], sensor_columns[j]
                df[f"ratio_{a}_{b}"] = df[a] / (df[b] + 1e-6)
        return df, scaler, encoder

    if mode == "full":
        df["sum_norm"] = df[sensor_columns].sum(axis=1)
        for sensor in sensor_columns:
            df[f"ratio_{sensor}"] = df[sensor] / (df["sum_norm"] + 1e-6)
            df[f"deriv_{sensor}"] = df[sensor].diff().fillna(0)
            df[f"rolling_mean_{sensor}"] = df[sensor].rolling(window=50, min_periods=1).mean()
            df[f"rolling_std_{sensor}"] = df[sensor].rolling(window=50, min_periods=1).std().fillna(0)
            df[f"lag1_{sensor}"] = df[sensor].shift(1).fillna(0)
        for i in range(len(sensor_columns)):
            for j in range(i + 1, len(sensor_columns)):
                a, b = sensor_columns[i], sensor_columns[j]
                df[f"ratio_{a}_{b}"] = df[a] / (df[b] + 1e-6)
                df[f"cross_corr_{a}_{b}"] = df[a] * df[b]
                df[f"poly_{a}_{b}"] = df[a] * (df[b]**2)
        return df, scaler, encoder

# ------------------ Feature Redundancy Removal ------------------
def remove_redundant_features(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    X_reduced = X.drop(columns=to_drop)
    return X_reduced

# ------------------ Prepare Data ------------------
def prepare_data(mode, seq_length=10):
    df = pd.read_csv("water_model_train.csv").dropna()
    df, scaler, encoder = feature_engineering(df, mode=mode)

    X = df.drop(columns=["Gas"])
    y = df["Gas"]

    X = remove_redundant_features(X, threshold=0.9)
    feature_names = X.columns.tolist()

    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    X_seq, y_seq = create_sequences(X, pd.Series(y_scaled), seq_length=seq_length)

    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False)

    X_train_orig = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))
    X_val_orig = pd.DataFrame(X_val.reshape(X_val.shape[0], -1))

    X_train_orig.columns = [f"{feat}_t{t}" for t in range(seq_length) for feat in feature_names]
    X_val_orig.columns = X_train_orig.columns

    return X_train, X_val, y_train, y_val, feature_names, scaler, encoder, X_train_orig, X_val_orig, target_scaler

# ------------------ Build LSTM ------------------
def build_lstm_model(input_shape, units=50, layers=1, learning_rate=0.001, l2_reg=0.0):
    
    units = int(units)
    layers = int(layers)
    model = Sequential()
    model.add(Input(shape=input_shape))

    if layers > 1:
        model.add(LSTM(units=units, return_sequences=True, kernel_regularizer=l2(l2_reg)))
    else:
        model.add(LSTM(units=units, return_sequences=False, kernel_regularizer=l2(l2_reg)))

    for i in range(layers - 1):
        model.add(LSTM(units=units, return_sequences=(i < layers - 2), kernel_regularizer=l2(l2_reg)))

    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

# ------------------ Bayesian Optimization ------------------
def optimize_bayesian(X_train, y_train, X_val, y_val, feature_names, X_train_orig, X_val_orig, seq_length):
    space = [
        Integer(1, 3, name='layers'),
        Integer(10, 50, name='units'),
        Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),
        Real(1e-8, 1e-2, prior='log-uniform', name='l2_reg'),
        Integer(1, len(feature_names), name='feature_index')
    ]

    @use_named_args(space)
    def objective(**params):
        K.clear_session()
        gc.collect()
        top_n = params['feature_index']
        selected_feats = feature_names[:top_n]
        selected_columns = [col for feat in selected_feats for col in X_train_orig.columns if col.startswith(f"{feat}_t")]

        X_train_lstm = X_train_orig[selected_columns].values.reshape((X_train_orig.shape[0], seq_length, len(selected_feats)))
        X_val_lstm = X_val_orig[selected_columns].values.reshape((X_val_orig.shape[0], seq_length, len(selected_feats)))

        model = build_lstm_model((seq_length, len(selected_feats)), units=params['units'], layers=params['layers'], learning_rate=params['learning_rate'], l2_reg=params['l2_reg'])
        model.fit(X_train_lstm, y_train, epochs=5, batch_size=32, validation_data=(X_val_lstm, y_val), verbose=0)

        y_pred = model.predict(X_val_lstm)
        return -r2_score(y_val, y_pred)

    result = gp_minimize(objective, dimensions=space, n_calls=30, n_initial_points=10, random_state=42, verbose=False)

    best_layers, best_units, best_lr, best_l2, best_feat_idx = result.x
    best_feats = feature_names[:best_feat_idx]

    selected_columns = [col for feat in best_feats for col in X_train_orig.columns if col.startswith(f"{feat}_t")]
    X_train_lstm = X_train_orig[selected_columns].values.reshape((X_train_orig.shape[0], seq_length, len(best_feats)))
    X_val_lstm = X_val_orig[selected_columns].values.reshape((X_val_orig.shape[0], seq_length, len(best_feats)))

    best_model = build_lstm_model(
        (seq_length, len(best_feats)),
        units=best_units,
        layers=best_layers,
        learning_rate=best_lr,
        l2_reg=best_l2
    )

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    history = best_model.fit(
        X_train_lstm, y_train,
        validation_data=(X_val_lstm, y_val),
        epochs=30,
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping_callback]
    )

    return {
        "Best Model": best_model,
        "Selected Features": best_feats,
        "layers": best_layers,
        "units": best_units,
        "lr": best_lr,
        "l2": best_l2,
        "history": history
    }, result


# ------------------ Predict on new unseen data ------------------
def predict_on_new_data(mode, scaler, encoder, target_scaler, selected_features, seq_length=200):
    df = pd.read_csv("water_model_test.csv").dropna()
    df, _, _ = feature_engineering(df, mode=mode, scaler=scaler, encoder=encoder, fit_scaler=False, fit_encoder=False)
    X = df.drop(columns=["Gas"])
    y = df["Gas"]

    y_scaled = target_scaler.transform(y.values.reshape(-1, 1)).flatten()

    # Create sequences
    X_seq, y_seq = create_sequences(X, pd.Series(y_scaled), seq_length=seq_length)

    X_seq_df = pd.DataFrame(X_seq.reshape(X_seq.shape[0], -1))
    X_seq_df.columns = [f"{feat}_t{t}" for t in range(seq_length) for feat in X.columns.tolist()]

    selected_columns = [col for feat in selected_features for col in X_seq_df.columns if col.startswith(f"{feat}_t")]
    X_new = X_seq_df[selected_columns].values.reshape((X_seq_df.shape[0], seq_length, len(selected_features)))

    return X_new, y_seq

# ------------------ Save & Plot Results ------------------
def save_and_plot_best_model(best_results, mode, scaler, encoder, target_scaler):
    best_model = best_results['Best Model']
    selected_features = best_results['Selected Features']
    history = best_results.get("history", None)

    # Save model
    joblib.dump(best_model, f'{mode}_waterl_best_model_lstm.joblib')
    print(f"Best model saved as {mode}_waterl_best_model_lstm.joblib")

    # Save training history
    if history is not None:
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(f"{mode}_waterl_training_history.csv", index=False)
        print(f"Training history saved as {mode}_waterl_training_history.csv")

    # Predict on new unseen data
    X_new, y_real_scaled = predict_on_new_data(mode, scaler, encoder, target_scaler, selected_features, seq_length=200)
    y_pred_scaled = best_model.predict(X_new).flatten()

    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_real = target_scaler.inverse_transform(y_real_scaled.reshape(-1, 1)).flatten()

    # Calculate R² and MAE on external test set
    r2_real = r2_score(y_real, y_pred)
    mae_real = mean_absolute_error(y_real, y_pred)
    print(f"{mode} Mode: R² on NEW unseen data (real units) = {r2_real:.4f}")
    print(f"{mode} Mode: MAE on NEW unseen data (real units) = {mae_real:.4f}")

    # Save real vs predicted
    real_vs_predicted_df = pd.DataFrame({
        'Real Gas': y_real,
        'Predicted Gas': y_pred
    })
    real_vs_predicted_df.to_csv(f'{mode}_waterl_real_vs_predicted_lstm.csv', index=False)

    # Save model summary
    summary_data = {
        'Mode': mode,
        'Layers': best_results['layers'],
        'Units per Layer': best_results['units'],
        'Learning Rate': best_results['lr'],
        'L2 Regularization': best_results['l2'],
        'Number of Selected Features': len(selected_features),
    }
    for i, feat in enumerate(selected_features):
        summary_data[f'Selected Feature {i+1}'] = feat
    summary_data['External R2'] = r2_real
    summary_data['External MAE'] = mae_real

    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(f'{mode}_waterl_model_summary_lstm.csv', index=False)
    print(f"Summary saved as {mode}_waterl_model_summary_lstm.csv")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(real_vs_predicted_df['Real Gas'], label="Real Gas", lw=2)
    plt.plot(real_vs_predicted_df['Predicted Gas'], label="Predicted Gas", lw=2)
    plt.title(f"{mode} Mode: True vs Predicted on External Test Set (LSTM)")
    plt.xlabel("Sample")
    plt.ylabel("Gas Concentration [%RH]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{mode}_waterl_true_vs_predicted_lstm.png')
    plt.show()


# ------------------ Model Testing Loop ------------------
def model_testing_loop():
    modes = ['full', 'derivatives', 'ratios', 'basic']
    results = []

    for mode in modes:
        print(f"\nTraining model with {mode} mode")

        X_train, X_val, y_train, y_val, feature_names, scaler, encoder, X_train_orig, X_val_orig, target_scaler = prepare_data(mode=mode, seq_length=200)

        best_results, _ = optimize_bayesian(X_train, y_train, X_val, y_val, feature_names, X_train_orig, X_val_orig, seq_length=200)

        save_and_plot_best_model(best_results, mode, scaler, encoder, target_scaler)

        results.append(best_results)

    return results

# Execute
results = model_testing_loop()
