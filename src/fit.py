import pandas as pd
import numpy as np
import json
import re
import itertools
import logging
from pathlib import Path
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error,
    accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
)
from collections import Counter


class ModelTrainer:
    EXCLUDED_SCORES = {'0:0', '1:1', '2:2', '3:3', '1:0', '0:1', '0:3'}

    def __init__(self, features_path, matches_path, model_dir="models", log_path="logs/train.log"):
        self.features_path = Path(features_path)
        self.matches_path = Path(matches_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.metrics_path = self.model_dir / "metrics.json"

        self.model_path_minutes = self.model_dir / "minutes_model.cbm"
        self.model_path_score = self.model_dir / "score_model.cbm"

        # Логгер
        Path(log_path).parent.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)

    def calculate_set_score(self, score_string):
        if not isinstance(score_string, str):
            return ''
        sets = score_string.strip().split()
        player1_wins = 0
        player2_wins = 0

        for set_score in sets:
            clean_score = re.sub(r'\(.*?\)', '', set_score)
            try:
                p1, p2 = map(int, clean_score.split('-'))
                if p1 > p2:
                    player1_wins += 1
                else:
                    player2_wins += 1
            except ValueError:
                continue
        return f"{player1_wins}:{player2_wins}"

    def parse_score(self, score):
        return str(score)

    def get_features(self, name, prefix, features_df):
        row = features_df.loc[name]
        return row.add_prefix(f"{prefix}_")

    def load_and_prepare_data(self):
        logging.info("Загрузка данных...")
        features_df = pd.read_csv(self.features_path).set_index("name")
        matches_df = pd.read_csv(self.matches_path)
        matches_df['score'] = matches_df['score'].apply(self.calculate_set_score)
        matches_df = matches_df[~matches_df["score"].isin(self.EXCLUDED_SCORES)].dropna(subset=["minutes"])

        X_rows, y_minutes, y_score = [], [], []

        for _, row in matches_df.iterrows():
            winner, loser = row["winner_name"], row["loser_name"]
            if winner not in features_df.index or loser not in features_df.index:
                continue
            try:
                p1 = self.get_features(winner, "p1", features_df)
                p2 = self.get_features(loser, "p2", features_df)
                X_rows.append(pd.concat([p1, p2]))
                y_minutes.append(row["minutes"])
                y_score.append(self.parse_score(row["score"]))
            except Exception as e:
                logging.warning(f"Ошибка при обработке матча {winner} vs {loser}: {e}")
                continue

        X = pd.DataFrame(X_rows).fillna(method="ffill")
        return X, pd.Series(y_minutes), pd.Series(y_score)

    def evaluate_and_save_metrics(self, model_name, y_true, y_pred, task):
        logging.info(f"Расчёт метрик для '{model_name}' ({task})")
        metrics = {}

        if task == "regression":
            metrics["MAE"] = mean_absolute_error(y_true, y_pred)
            metrics["MSE"] = mean_squared_error(y_true, y_pred)
            metrics["R2"] = r2_score(y_true, y_pred)
        else:
            metrics["Accuracy"] = accuracy_score(y_true, y_pred)
            metrics["F1_macro"] = f1_score(y_true, y_pred, average="macro")
            metrics["Balanced_Accuracy"] = balanced_accuracy_score(y_true, y_pred)
            try:
                from sklearn.preprocessing import LabelBinarizer
                lb = LabelBinarizer()
                y_true_bin = lb.fit_transform(y_true)
                y_pred_bin = lb.transform(y_pred)
                metrics["ROC_AUC_macro"] = roc_auc_score(y_true_bin, y_pred_bin, average="macro", multi_class="ovo")
            except:
                metrics["ROC_AUC_macro"] = "N/A"

        all_metrics = {}
        if self.metrics_path.exists():
            with open(self.metrics_path, "r") as f:
                all_metrics = json.load(f)

        all_metrics[model_name] = metrics
        with open(self.metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        logging.info(f"Метрики сохранены в {self.metrics_path}")

    def simple_grid_search(self, X, y, is_classifier=False):
        logging.info(f"Grid Search для {'classifier' if is_classifier else 'regressor'}")
        param_grid = {
            "iterations": [300, 500, 700],
            "learning_rate": [0.03],
            "depth": [3]
        }

        best_score = float("inf")
        best_model = None
        best_params = None

        cat_features = [col for col in X.columns if X[col].dtype == "object"]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)

        for combo in itertools.product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), combo))
            params.update({
                "l2_leaf_reg": 3,
                "early_stopping_rounds": 50,
                "verbose": 0,
                "random_seed": 42,
            })

            if is_classifier:
                class_counts = Counter(y_train)
                total = sum(class_counts.values())
                class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
                params["class_weights"] = class_weights
                params["loss_function"] = "MultiClass"
                model = CatBoostClassifier(**params)
            else:
                params["loss_function"] = "RMSE"
                model = CatBoostRegressor(**params)

            model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val))

            preds = model.predict(X_val)
            score = (
                (1 - accuracy_score(y_val, preds)) if is_classifier
                else mean_absolute_error(y_val, preds)
            )

            if score < best_score:
                best_score = score
                best_model = model
                best_params = params

        logging.info(f"🏁 Лучшие параметры: {best_params}")
        return best_model

    def train(self):
        logging.info("Старт обучения моделей...")
        X, y_minutes, y_score = self.load_and_prepare_data()

        logging.info("Обучение модели предсказания минут...")
        reg_model = self.simple_grid_search(X, y_minutes, is_classifier=False)
        reg_model.save_model(self.model_path_minutes)
        self.evaluate_and_save_metrics("minutes_model", y_minutes, reg_model.predict(X), task="regression")

        logging.info("Обучение модели предсказания счёта...")
        clf_model = self.simple_grid_search(X, y_score, is_classifier=True)
        clf_model.save_model(self.model_path_score)
        self.evaluate_and_save_metrics("score_model", y_score, clf_model.predict(X), task="classification")

        logging.info("✅ Обучение завершено. Модели и метрики сохранены.")


if __name__ == "__main__":
    trainer = ModelTrainer(
        features_path="features/player_features.csv",
        matches_path="data/atp_matches_all.csv"
    )
    trainer.train()
