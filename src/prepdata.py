import pandas as pd
import re
from pathlib import Path


class FeatureBuilder:
    EXCLUDED_SCORES = {'0:0', '1:1', '2:2', '3:3', '1:0', '0:1', '0:3'}

    def __init__(self, input_path="data/atp_matches_all.csv", output_path="features/player_features.csv"):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.df = None

    def parse_score(self, score):
        try:
            sets_won, sets_lost = map(int, score.split(":"))
            return sets_won, sets_lost
        except:
            return None, None

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

    def load_data(self):
        df = pd.read_csv(self.input_path)

        columns = [
            "tourney_date",
            "winner_hand", "winner_ht", "winner_ioc", "winner_age", "winner_name",
            "loser_hand", "loser_ht", "loser_ioc", "loser_age", "loser_name",
            "score", "minutes"
        ]

        df = df[columns].dropna(subset=["minutes", "score"])
        df['score'] = df['score'].apply(self.calculate_set_score)
        df = df[~df['score'].isin(self.EXCLUDED_SCORES)]

        df['tourney_date'] = pd.to_datetime(df['tourney_date'].astype(str), format='%Y%m%d')
        df[['sets_won', 'sets_lost']] = df['score'].apply(lambda x: pd.Series(self.parse_score(x)))

        self.df = df

    def get_latest_static_features(self):
        players = []

        for role in ['winner', 'loser']:
            role_df = self.df[[f'{role}_name', f'{role}_hand', f'{role}_ht', f'{role}_ioc', f'{role}_age', 'tourney_date']].copy()
            role_df.columns = ['name', 'hand', 'ht', 'ioc', 'age', 'tourney_date']
            role_df = role_df.sort_values('tourney_date')
            role_df = role_df.dropna(subset=['hand', 'ht', 'ioc', 'age'])
            role_df = role_df.drop_duplicates(subset='name', keep='last')
            players.append(role_df)

        all_players = pd.concat(players, axis=0)
        all_players = all_players.drop_duplicates(subset='name', keep='last')
        return all_players.reset_index(drop=True)

    def get_aggregated_stats(self):
        df = self.df

        winner_stats = df.groupby('winner_name').agg(
            wins=('winner_name', 'count'),
            avg_minutes=('minutes', 'mean'),
            avg_sets_won=('sets_won', 'mean'),
            avg_sets_lost=('sets_lost', 'mean'),
            avg_opponent_age=('loser_age', 'mean'),
            avg_opponent_ht=('loser_ht', 'mean')
        ).reset_index().rename(columns={'winner_name': 'name'})

        loser_stats = df.groupby('loser_name').agg(
            losses=('loser_name', 'count'),
            avg_minutes_l=('minutes', 'mean'),
            avg_sets_won_l=('sets_lost', 'mean'),
            avg_sets_lost_l=('sets_won', 'mean'),
            avg_opponent_age_l=('winner_age', 'mean'),
            avg_opponent_ht_l=('winner_ht', 'mean')
        ).reset_index().rename(columns={'loser_name': 'name'})

        full = pd.merge(winner_stats, loser_stats, on='name', how='outer').fillna(0)
        full['total_matches'] = full['wins'] + full['losses']
        full['win_rate'] = full['wins'] / full['total_matches'].replace(0, 1)

        full['avg_minutes'] = (
            (full['avg_minutes'] * full['wins'] + full['avg_minutes_l'] * full['losses']) /
            full['total_matches'].replace(0, 1)
        )
        full['avg_sets_won'] = (
            (full['avg_sets_won'] * full['wins'] + full['avg_sets_won_l'] * full['losses']) /
            full['total_matches'].replace(0, 1)
        )
        full['avg_sets_lost'] = (
            (full['avg_sets_lost'] * full['wins'] + full['avg_sets_lost_l'] * full['losses']) /
            full['total_matches'].replace(0, 1)
        )
        full['avg_opponent_age'] = (
            (full['avg_opponent_age'] * full['wins'] + full['avg_opponent_age_l'] * full['losses']) /
            full['total_matches'].replace(0, 1)
        )
        full['avg_opponent_ht'] = (
            (full['avg_opponent_ht'] * full['wins'] + full['avg_opponent_ht_l'] * full['losses']) /
            full['total_matches'].replace(0, 1)
        )

        return full[['name', 'total_matches', 'wins', 'losses', 'win_rate', 
                     'avg_minutes', 'avg_sets_won', 'avg_sets_lost',
                     'avg_opponent_age', 'avg_opponent_ht']]

    def build_features(self):
        print("Загрузка и обработка данных...")
        self.load_data()

        print("Извлечение последних известных статических данных игроков...")
        static = self.get_latest_static_features()

        print("Расчет агрегированных статистик...")
        stats = self.get_aggregated_stats()

        print("Объединение и сохранение признаков...")
        final = pd.merge(static, stats, on='name', how='left')
        final.to_csv(self.output_path, index=False)

        print(f"Признаки сохранены в: {self.output_path}")


if __name__ == "__main__":
    builder = FeatureBuilder()
    builder.build_features()
