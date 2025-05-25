import pandas as pd
import re
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
from pathlib import Path
import json
import optuna

FEATURE_PATH = Path("features/player_features.csv")
MATCHES_PATH = Path("data/atp_matches_all.csv")
MODEL_PATH_MINUTES = Path("models/minutes_model.cbm")
MODEL_PATH_SCORE = Path("models/score_model.cbm")
METRICS_PATH = Path("models/metrics.json")

EXCLUDED_SCORES = {'0:0', '1:1', '2:2', '3:3', '1:0', '0:1', '0:3'}

def calculate_set_score(score_string):
    if not isinstance(score_string, str):
        return ''
    
    sets = score_string.strip().split()
    player1_wins = 0
    player2_wins = 0
    
    for set_score in sets:
        # Убираем тай-брейк в скобках, если есть
        clean_score = re.sub(r'\(.*?\)', '', set_score)
        try:
            p1, p2 = map(int, clean_score.split('-'))
            if p1 > p2:
                player1_wins += 1
            else:
                player2_wins += 1
        except ValueError:
            # Невалидный формат — пропускаем
            continue

    return f"{player1_wins}:{player2_wins}"

def parse_score(score):
    return str(score)

def load_and_prepare_data():
    features_df = pd.read_csv(FEATURE_PATH).set_index('name')
    matches_df = pd.read_csv(MATCHES_PATH)
    matches_df = matches_df[~matches_df['score'].isin(EXCLUDED_SCORES)].dropna(subset=['minutes'])

    def get_features(name, prefix):
        row = features_df.loc[name]
        return row.add_prefix(f"{prefix}_")

    X_rows, y_minutes, y_score = [], [], []

    for _, row in matches_df.iterrows():
        winner, loser = row['winner_name'], row['loser_name']
        if winner not in features_df.index or loser not in features_df.index:
            continue
        try:
            p1 = get_features(winner, "p1")
            p2 = get_features(loser, "p2")
            X_rows.append(pd.concat([p1, p2]))
            y_minutes.append(row['minutes'])
            y_score.append(parse_score(row['score']))
        except:
            continue

    X = pd.DataFrame(X_rows).fillna(method='ffill')
    return X, pd.Series(y_minutes), pd.Series(y_score)

def evaluate_and_save_metrics(model_name, y_true, y_pred, task):
    metrics = {}
    if task == "regression":
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['R2'] = r2_score(y_true, y_pred)
    else:
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['F1_macro'] = f1_score(y_true, y_pred, average='macro')

    all_metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, 'r') as f:
            all_metrics = json.load(f)

    all_metrics[model_name] = metrics
    with open(METRICS_PATH, 'w') as f:
        json.dump(all_metrics, f, indent=2)

def tune_model(X, y, is_classifier=False):
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
            "depth": trial.suggest_int("depth", 3, 6),
            "l2_leaf_reg": 3,
            "early_stopping_rounds": 50,
            "verbose": 100
        }
        if is_classifier:
            params["loss_function"] = "MultiClass"
            model = CatBoostClassifier(**params)
        else:
            params["loss_function"] = "RMSE"
            model = CatBoostRegressor(**params)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
        cat_features = [col for col in X.columns if X[col].dtype == "object"]
        model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val))
        preds = model.predict(X_val)
        if is_classifier:
            return 1 - accuracy_score(y_val, preds)
        return mean_absolute_error(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)
    return study.best_params

def train_models_with_optuna():
    X, y_minutes, y_score = load_and_prepare_data()

    # Regressor
    best_reg_params = tune_model(X, y_minutes, is_classifier=False)
    reg_model = CatBoostRegressor(**best_reg_params)
    X_train_m, X_val_m, y_train_m, y_val_m = train_test_split(X, y_minutes, test_size=0.2, shuffle=True, random_state=42)
    cat_features = [col for col in X.columns if X[col].dtype == "object"]
    reg_model.fit(X_train_m, y_train_m, cat_features=cat_features, eval_set=(X_val_m, y_val_m))
    reg_model.save_model(MODEL_PATH_MINUTES)
    evaluate_and_save_metrics("minutes_model", y_val_m, reg_model.predict(X_val_m), task="regression")

    # Classifier
    best_clf_params = tune_model(X, y_score, is_classifier=True)
    clf_model = CatBoostClassifier(**best_clf_params)
    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X, y_score, test_size=0.2, shuffle=True, random_state=42)
    clf_model.fit(X_train_s, y_train_s, cat_features=cat_features, eval_set=(X_val_s, y_val_s))
    clf_model.save_model(MODEL_PATH_SCORE)
    evaluate_and_save_metrics("score_model", y_val_s, clf_model.predict(X_val_s), task="classification")

if __name__ == "__main__":
    train_models_with_optuna()