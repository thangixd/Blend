from src.Operators.OperatorBase import Operator
from abc import ABC
from pathlib import Path
from src.DBHandler import DBHandler

class Seeker(Operator, ABC):
    def __init__(self, k: int) -> None:
        super().__init__(k)

        self._cached_predicted_runtime = None
        if self.DB.USE_ML_OPTIMIZER:
            from xgboost import XGBRegressor
            self.model = XGBRegressor()
            self.model.load_model(Path(__file__).parent / f"{self.__class__.__name__}_model.json")
        else:
            self.model = None
            self._cached_predicted_runtime = 1

    def _feature_columns(self) -> list:
        raise NotImplementedError

    def _features(self, db: DBHandler) -> list:
        columns = self._feature_columns()
        rows = [tuple(row) for row in zip(*columns)]

        freqs = db.get_token_frequencies(set().union(*columns))
        prod = 1
        for col in columns:
            prod *= sum(freqs[token] for token in set(col) if token in freqs)

        return [len(set(rows)), prod ** (1 / len(columns)), len(columns)]

    def ml_cost(self, db: DBHandler) -> float:
        if self._cached_predicted_runtime is None:
            self._cached_predicted_runtime = float(self.model.predict([self._features(db)])[0])

        return self._cached_predicted_runtime
