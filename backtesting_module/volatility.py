from abc import ABC, abstractmethod
import pandas as pd

class VolatilityModel(ABC):
    """
    All σ forecasters – GARCH, HAR-RV, LSTM, whatever – implement this.
    """

    @abstractmethod
    def warm_up(self, price_series: pd.Series) -> None:
        """Load initial history (called once before the back-test)."""

    @abstractmethod
    def forecast(self, t: pd.Timestamp) -> float:
        """Return annualised σ forecast for time t *using only past data*."""

    @abstractmethod
    def update(self, t: pd.Timestamp, price: float) -> None:
        """Push the new realised price so the model can roll forward."""
