import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor, CatBoostClassifier
import random
import difflib
import logging


class MatchPredictor:
    def __init__(
        self,
        features_path="features/player_features.csv",
        minutes_model_path="models/minutes_model.cbm",
        score_model_path="models/score_model.cbm",
        log_file="logs/player_replacements.log"
    ):
        self.features_path = Path(features_path)
        self.minutes_model_path = Path(minutes_model_path)
        self.score_model_path = Path(score_model_path)
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)

        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self.player_df = self._load_features()
        self.minutes_model = self._load_model(self.minutes_model_path, CatBoostRegressor)
        self.score_model = self._load_model(self.score_model_path, CatBoostClassifier)

    def _load_features(self):
        return pd.read_csv(self.features_path).set_index("name")

    def _load_model(self, model_path: Path, model_class):
        model = model_class()
        model.load_model(model_path)
        return model

    def _suggest_similar_players(self, name: str, n=3, cutoff=0.6):
        return difflib.get_close_matches(name, self.player_df.index.tolist(), n=n, cutoff=cutoff)

    def _pick_random_opponent(self, exclude_name: str = None):
        candidates = [name for name in self.player_df.index if name != exclude_name]
        return random.choice(candidates)

    def predict(self, p1_name: str, p2_name: str):
        player_list = self.player_df.index.tolist()
        p1_known = p1_name in self.player_df.index
        p2_known = p2_name in self.player_df.index
        notes = []

        if not p1_known and not p2_known:
            msg = f"❌ Оба игрока не найдены: '{p1_name}' и '{p2_name}'"
            logging.warning(msg)
            return {
                "player_1": p1_name,
                "player_2": p2_name,
                "predicted_score": "нет данных",
                "predicted_minutes": "нет данных",
                "note": msg
            }

        if not p1_known:
            similar = self._suggest_similar_players(p1_name)
            if similar:
                new_p1 = similar[0]
                log_msg = f"Игрок '{p1_name}' не найден. Заменён на похожего: '{new_p1}'"
                notes.append(f"⚠️ {log_msg}\n🔍 Похожие: {', '.join(similar)}")
            else:
                new_p1 = self._pick_random_opponent(p2_name if p2_known else None)
                log_msg = f"Игрок '{p1_name}' не найден. Похожие не найдены. Заменён на случайного: '{new_p1}'"
                notes.append(f"⚠️ {log_msg}")
            logging.info(log_msg)
            p1_name = new_p1

        if not p2_known:
            similar = self._suggest_similar_players(p2_name)
            if similar:
                new_p2 = similar[0]
                log_msg = f"Игрок '{p2_name}' не найден. Заменён на похожего: '{new_p2}'"
                notes.append(f"⚠️ {log_msg}\n🔍 Похожие: {', '.join(similar)}")
            else:
                new_p2 = self._pick_random_opponent(p1_name)
                log_msg = f"Игрок '{p2_name}' не найден. Похожие не найдены. Заменён на случайного: '{new_p2}'"
                notes.append(f"⚠️ {log_msg}")
            logging.info(log_msg)
            p2_name = new_p2

        # Подготовка фичей
        p1_feats = self.player_df.loc[p1_name].add_prefix("p1_")
        p2_feats = self.player_df.loc[p2_name].add_prefix("p2_")
        X = pd.concat([p1_feats, p2_feats]).to_frame().T.fillna(method="ffill")

        # Предсказания
        pred_minutes = self.minutes_model.predict(X)[0]
        pred_score = self.score_model.predict(X)[0]

        return {
            "player_1": p1_name,
            "player_2": p2_name,
            "predicted_score": str(pred_score),
            "predicted_minutes": round(pred_minutes),
            "note": "\n\n".join(notes) if notes else "✅ Предсказание выполнено на основе модели"
        }
