from __future__ import annotations

import polars as pl
import pandas as pd
from data import WallboxesColumns

from statsmodels.tsa.arima.model import ARIMA


def forecast_n_days(
    accumulated: pl.DataFrame, model: dict, DAYS_IN_FUTURE: int, dates: list
):
    model_name = list(model.keys())[0]
    best_fit = list(model.values())[0]
    curr_model = ARIMA(
        accumulated.select(model_name).to_numpy(),
        order=best_fit,
    ).fit()

    forecast = curr_model.predict(
        start=len(accumulated.select(model_name).to_numpy()),
        end=len(accumulated.select(model_name).to_numpy()) + DAYS_IN_FUTURE,
    )

    print(f"Forecast for {model_name}:")
    print(forecast)


def forecast_total_power(df: pl.DataFrame) -> None:
    DAYS_IN_FUTURE = 7

    accumulated = df.with_columns(
        pl.fold(
            acc=pl.lit(0),
            function=lambda acc, col: acc + col,
            exprs=[
                WallboxesColumns.KEBA_ONE,
                WallboxesColumns.KEBA_THREE,
                WallboxesColumns.KEBA_TWO,
                WallboxesColumns.LADEBOX_ONE,
                WallboxesColumns.LADEBOX_TWO,
                WallboxesColumns.LADEBOX_THREE,
                WallboxesColumns.DELTA,
                WallboxesColumns.RAPTION,
            ],
        ).alias("Total Power"),
    ).select(
        [
            WallboxesColumns.TIMESTAMP,
            "Total Power",
            WallboxesColumns.KEBA_ONE,
            WallboxesColumns.KEBA_THREE,
            WallboxesColumns.KEBA_TWO,
            WallboxesColumns.LADEBOX_ONE,
            WallboxesColumns.LADEBOX_TWO,
            WallboxesColumns.LADEBOX_THREE,
            WallboxesColumns.DELTA,
            WallboxesColumns.RAPTION,
        ]
    )

    best_arima_models = [
        {"Total Power": (3, 1, 3)},
        {WallboxesColumns.KEBA_ONE: (1, 1, 0)},
        {WallboxesColumns.KEBA_THREE: (1, 1, 2)},
        {WallboxesColumns.KEBA_TWO: (4, 1, 4)},
        {WallboxesColumns.LADEBOX_ONE: (0, 1, 0)},
        {WallboxesColumns.LADEBOX_TWO: (0, 1, 1)},
        {WallboxesColumns.LADEBOX_THREE: (0, 1, 1)},
        {WallboxesColumns.DELTA: (0, 1, 0)},
        {WallboxesColumns.RAPTION: (2, 0, 4)},
    ]

    last_date = accumulated.select("Timestamp").to_numpy()
    last_date = last_date[-1][0]
    dates = pd.date_range(last_date, periods=DAYS_IN_FUTURE + 1)

    for model in best_arima_models:
        forecast_n_days(accumulated, model, DAYS_IN_FUTURE, dates)
