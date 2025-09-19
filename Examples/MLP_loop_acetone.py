import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.inspection import permutation_importance
from skopt.callbacks import VerboseCallback
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import joblib

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
    # Compute the correlation matrix
    corr_matrix = X.corr().abs()
    
    # Get the upper triangle of the correlation matrix
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    
    # Drop the redundant features
    X_reduced = X.drop(columns=to_drop)
    
    return X_reduced

# ------------------ Load and Prepare Data ------------------
def prepare_data(mode):
    df = pd.read_csv("acetone_model_train.csv").dropna()
    df, scaler, encoder = feature_engineering(df, mode=mode)  # basic, ratios, derivatives, full
    
    X = df.drop(columns=["Gas"])
    y = df["Gas"]

    # Remove redundant features
    X = remove_redundant_features(X, threshold=0.9)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=False)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, encoder

# ------------------ Feature Selection ------------------
def estimate_feature_contributions(model, X, y, selected_features):
    result = permutation_importance(model, X, y, n_repeats=20, random_state=0)
    importances = result.importances_mean
    y_pred = model.predict(X)

    # Sort by absolute importance for selection
    importance_df = pd.DataFrame({"Feature": selected_features, "Importance": importances})
    importance_df["Abs_Importance"] = importance_df["Importance"].abs()  # Use absolute importance
    importance_df = importance_df.sort_values(by="Abs_Importance", ascending=False)

    # Get the sign for interpretation but not for selection
    importance_df["Estimated Sign"] = importance_df["Importance"].apply(lambda x: "Positive" if x > 0 else "Negative")
    
    return importance_df

# ------------------ Bayesian Optimization ------------------
def optimize_bayesian(X_train, y_train, X_test, y_test, all_features):
    # Define the search space for the neural network hyperparameters
    space = [
        Integer(5, 30, name='layer1'),
        Integer(0, 30, name='layer2'),
        Real(0.1, 0.5, prior='uniform', name='alpha'),
        Integer(1, len(all_features), name='feature_index')  # Dynamically selecting top_n_features
    ]

    @use_named_args(space)
    def objective(**params):
        top_n = params['feature_index']
        selected_feats = all_features[:top_n]

        layers = (params['layer1'],) if params['layer2'] == 0 else (params['layer1'], params['layer2'])
        model = MLPRegressor(hidden_layer_sizes=layers, alpha=params['alpha'], max_iter=2000, random_state=42)

        model.fit(X_train[selected_feats], y_train)
        y_pred = model.predict(X_test[selected_feats])
        base_score = r2_score(y_test, y_pred)

        size_penalty = (params['layer1'] + params['layer2']) * 1e-3
        alpha_penalty = (1e-2 / (params['alpha'] + 1e-6)) * 1e-2

        penalized_score = base_score - (size_penalty + alpha_penalty)

        return -penalized_score

    result = gp_minimize(
        objective,
        dimensions=space,
        n_calls=60,
        n_initial_points=30,
        random_state=42,
        verbose=False
    )

    # Get the best hyperparameters
    best_layer1, best_layer2, best_alpha, best_feat_idx = result.x
    best_layers = (best_layer1,) if best_layer2 == 0 else (best_layer1, best_layer2)
    best_feats = all_features[:best_feat_idx]

    # Return the best model found through optimization
    final_model = MLPRegressor(hidden_layer_sizes=best_layers, alpha=best_alpha, max_iter=2000, random_state=42)
    return final_model, best_feats

# ------------------ Vary Number of Estimators ------------------

def vary_estimators(X_train, y_train, X_val, y_val, X_test, y_test, all_features, mode, estimators_range, scaler, encoder):
    # First, we optimize the model using bayesian optimization
    model, selected_features = optimize_bayesian(X_train, y_train, X_test, y_test, all_features)
    
    best_results = None
    all_predictions = []  # To store predictions for each number of estimators

    for estimators in estimators_range:
        print(f"Optimizing with {estimators} estimators")
        
        # Optimize Bagging model with different estimators
        bagging_model = BaggingRegressor(model, n_estimators=estimators, random_state=42)
        bagging_model.fit(X_train[selected_features], y_train)

        # Predict again on new data
        X_test_new, y_test_new = predict_on_new_data("acetone_model_test.csv", scaler, encoder, selected_features, mode)
        y_pred_new = bagging_model.predict(X_test_new)
        
        # Apply smoothing
        y_pred_smoothed = pd.Series(y_pred_new).rolling(window=20, min_periods=1).mean().values
        new_test_r2 = r2_score(y_test_new, y_pred_smoothed)
        
        # Store prediction and R² for later export
        all_predictions.append({
            'Number of Estimators': estimators,
            'New Test R² (Smoothed)': new_test_r2,
            'Smoothed Prediction': y_pred_smoothed.tolist()
        })

        # Evaluate performance
        train_r2 = r2_score(y_train, bagging_model.predict(X_train[selected_features]))
        val_r2 = r2_score(y_val, bagging_model.predict(X_val[selected_features]))
        test_r2 = r2_score(y_test, bagging_model.predict(X_test[selected_features]))
        new_test_r2 = r2_score(y_test_new, bagging_model.predict(X_test_new))

        # Store the results for the current estimator
        results = {
            'Mode': mode,
            'Number of Estimators': estimators,
            'Train R²': train_r2,
            'Validation R²': val_r2,
            'Test R²': test_r2,
            'New Test R²': new_test_r2,
            'Best Model': bagging_model,  # Store the best model
            'Selected Features': selected_features  # Store the selected features
        }

        # If this is the best result so far, save it
        if best_results is None or new_test_r2 > best_results['New Test R²']:
            best_results = results

    return best_results, all_predictions

# ------------------ Model Training ------------------
def train_model_with_estimators(X_train, y_train, X_val, y_val, X_test, y_test, all_features, mode, estimators=10):
    # Prepare data for the given mode
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, encoder = prepare_data(mode=mode)
    model, selected_features = optimize_bayesian(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        all_features=all_features
    )
    # Optimize Bagging model with a dynamic number of estimators
    bagging_model = BaggingRegressor(model, n_estimators=estimators, random_state=42)
    bagging_model.fit(X_train[selected_features], y_train)
    
    # Evaluate
    return bagging_model, selected_features

# ------------------ Predict on New Data ------------------
def predict_on_new_data(path, scaler, encoder, selected_features, mode):
    # Load new data
    df = pd.read_csv(path).dropna()

    # Apply the same feature engineering steps as on the training data
    df, _, _ = feature_engineering(df, mode=mode, scaler=scaler, encoder=encoder, fit_scaler=False, fit_encoder=False)

    # Prepare features (X) and labels (y) for new data
    X = df.drop(columns=["Gas"])
    y = df["Gas"]

    # Ensure all selected features are present in the new data
    for f in selected_features:
        if f not in X.columns:
            X[f] = 0  # Add missing features with value 0 (or handle appropriately)

    return X[selected_features], y

# ------------------ Save & Plot ------------------
def save_and_plot_best_model(best_results, X_train, y_train, X_test, y_test, mode, scaler, encoder):

    best_model = best_results['Best Model']
    joblib.dump(best_model, f'{mode}_acetone_best_model.joblib')
    print(f"Best model saved as {mode}_acetone_best_model.joblib")

    selected_features = best_results['Selected Features']
    best_model.fit(X_train[selected_features], y_train)

    # Predict on new test data
    X_test_new, y_test_new = predict_on_new_data("acetone_model_test.csv", scaler, encoder, selected_features, mode)
    y_pred_new = best_model.predict(X_test_new)

    # Smoothing with a rolling mean (20 samples = 2 seconds at 10 Hz)
    y_pred_smoothed = pd.Series(y_pred_new).rolling(window=20, min_periods=1).mean().values

    # Compute R² using smoothed prediction
    r2_smoothed = r2_score(y_test_new, y_pred_smoothed)
    print(f"{mode} Mode: R² on smoothed predictions = {r2_smoothed:.4f}")

    # Save results
    real_vs_predicted_df = pd.DataFrame({
        'Real Gas': y_test_new.values,
        'Predicted Gas': y_pred_new,
        'Smoothed Predicted Gas': y_pred_smoothed
    }, index=y_test_new.index).sort_index()

    real_vs_predicted_df.to_csv(f'{mode}_acetone_real_vs_predicted.csv', index=False)

    # Plot: True vs Smoothed Predicted
    plt.figure(figsize=(10, 5))
    plt.plot(real_vs_predicted_df['Real Gas'].values, label="True Gas", lw=2)
    plt.plot(real_vs_predicted_df['Smoothed Predicted Gas'].values, label="Smoothed Prediction", lw=2)
    plt.title(f"{mode} Mode: True vs Smoothed Prediction (New Test Data)")
    plt.xlabel("Time [sample index]")
    plt.ylabel("Gas Concentration")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{mode}_acetone_true_vs_predicted.png')
    plt.show()
    plt.close()

    # Residuals of smoothed prediction
    residuals = real_vs_predicted_df['Real Gas'] - real_vs_predicted_df['Smoothed Predicted Gas']

    plt.figure(figsize=(10, 5))
    plt.plot(residuals.values, label="Residuals", lw=2)
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"{mode} Mode: Residuals of Smoothed Prediction (New Test Data)")
    plt.xlabel("Time [sample index]")
    plt.ylabel("Residual (Real - Smoothed Predicted)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{mode}_acetone_residuals.png')
    plt.show()
    
    # Save residuals to CSV
    residuals_df = pd.DataFrame({
        'Real Gas': real_vs_predicted_df['Real Gas'],
        'Smoothed Predicted Gas': real_vs_predicted_df['Smoothed Predicted Gas'],
        'Residual (Real - Smoothed)': residuals
    })
    residuals_df.to_csv(f'{mode}_acetone_residuals.csv', index=False)
    print(f"Residuals saved as {mode}_acetone_residuals.csv")
    
# ------------------ Save Predictions for Each Estimator ------------------
def save_estimators_predictions(all_predictions, mode):
    # Expand predictions into rows
    rows = []
    for entry in all_predictions:
        estimators = entry['Number of Estimators']
        r2 = entry['New Test R² (Smoothed)']
        for i, pred in enumerate(entry['Smoothed Prediction']):
            rows.append({'Number of Estimators': estimators, 'Sample Index': i, 'Smoothed Prediction': pred, 'New Test R² (Smoothed)': r2})
    
    all_predictions_df = pd.DataFrame(rows)
    all_predictions_df.to_csv(f'{mode}_acetone_estimators_predictions.csv', index=False)
    print(f"Predictions for each estimator saved as {mode}_acetone_estimators_predictions.csv")


# ------------------ Model Testing Loop ------------------
def model_testing_loop():
    # Define feature engineering modes
    modes = ['full', 'derivatives', 'ratios', 'basic']
    results = []

    for mode in modes:
        print(f"\nTraining model with {mode} mode")
        
        # Prepare the data before entering the loop
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, encoder = prepare_data(mode=mode)  # Pass mode dynamically
        all_feats = X_train.columns.tolist()

        # Call the vary_estimators function for the current mode
        best_results, all_predictions = vary_estimators(X_train, y_train, X_val, y_val, X_test, y_test, all_feats, mode, estimators_range=range(1, 31), scaler=scaler, encoder=encoder)
        
        # Save the best results for the current mode
        results.append(best_results)

        importance_df = estimate_feature_contributions(best_results['Best Model'], X_test[best_results['Selected Features']], y_test, best_results['Selected Features'])
        importance_df.to_csv(f'{mode}_acetone_selected_features_importance.csv', index=False)
        
        # Save best results for this mode to CSV
        mode_results_df = pd.DataFrame([best_results])
        mode_results_df.to_csv(f'{mode}_acetone_results.csv', index=False)

        # Save the best model's real and predicted gas data and other results
        save_and_plot_best_model(best_results, X_train, y_train, X_test, y_test, mode, scaler, encoder)

        # Save predictions for each estimator
        save_estimators_predictions(all_predictions, mode)

        print(f"Best results for {mode} mode with {best_results['Number of Estimators']} estimators:")
        print(f"Train R²: {best_results['Train R²']:.4f}, Validation R²: {best_results['Validation R²']:.4f}, Test R²: {best_results['Test R²']:.4f}, New Test R²: {best_results['New Test R²']:.4f}")
    
    return results

# ------------------ Execute the loop ------------------
results = model_testing_loop()
