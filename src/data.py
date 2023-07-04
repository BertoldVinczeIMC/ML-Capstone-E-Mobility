from __future__ import annotations

from typing import Self
from enum import StrEnum
import polars as pl


class DataEnum(StrEnum):
    @classmethod
    def values(cls) -> list[str]:
        return list(map(lambda c: c.value, cls))


class BatteryColumns(DataEnum):
    TIMESTAMP = "Timestamp"
    POWER = "LEM.Overview.Wirkleistung_P"
    ENERGY_CHARGED = "LEM.Overview.Energy_Charged"
    ENERGY_DISCHARGED = "LEM.Overview.Energy_Discharged"
    STATE_OF_CHARGE = "LEM.Overview.Battery_SOC"


class GridColumns(DataEnum):
    TIMESTAMP = "Timestamp"
    POWER = "LEM.Einspeisung_HV_NE6.Wirkleistung_P"


class PhotovoltaicColumns(DataEnum):
    TIMESTAMP = "Timestamp"
    POWER = "LEM.PV_Anlage.Wirkleistung_P"


class WallboxesColumns(DataEnum):
    TIMESTAMP = "Timestamp"
    KEBA_ONE = "LEM.KEBA_P30_1.Wirkleistung_P"
    KEBA_TWO = "LEM.KEBA_P30_2.Wirkleistung_P"
    KEBA_THREE = "LEM.KEBA_P30_3.Wirkleistung_P"
    LADEBOX_ONE = "LEM.Ladebox1.P"
    LADEBOX_TWO = "LEM.Ladebox2.P"
    LADEBOX_THREE = "LEM.Ladebox3.P"
    DELTA = "LEM.Delta_Wallbox.Wirkleistung_P"
    RAPTION = "LEM.Raption_50.Wirkleistung_P"


class Datasets:
    """
    Class wrapper for all datasets

    Access the columns using the enum values
    Access the day of the week using the DayOfWeek column (provided for you already)

    Keep in mind: 1 = Monday, 2 = Tuesday, ..., 7 = Sunday

    Include more columns as needed by editing the enums
    """

    battery: pl.DataFrame
    grid: pl.DataFrame
    photovoltaic: pl.DataFrame
    wallboxes: pl.DataFrame

    def __init__(self):
        self.battery = (
            pl.read_csv(
                "dataset/data_battery_2022-01-01_2023-02-21.csv",
                separator=";",
                dtypes={t: pl.Float32 for t in BatteryColumns.values()}
                | {"Timestamp": pl.Utf8, "DayOfWeek": pl.UInt8},
            )
            .select(BatteryColumns.values())
            .with_columns(
                pl.col("Timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%:z"),
            )
            .with_columns(
                pl.col("Timestamp").dt.weekday().alias("DayOfWeek"),
            )
        ).set_sorted(BatteryColumns.TIMESTAMP)

        self.grid = (
            pl.read_csv(
                "dataset/data_grid_2022-01-01_2023-02-21.csv",
                separator=";",
                dtypes={t: pl.Float32 for t in GridColumns.values()}
                | {"Timestamp": pl.Utf8, "DayOfWeek": pl.UInt8},
            )
            .select(GridColumns.values())
            .with_columns(
                pl.col("Timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%:z"),
            )
            .with_columns(
                pl.col("Timestamp").dt.weekday().alias("DayOfWeek"),
            )
        ).set_sorted(BatteryColumns.TIMESTAMP)

        self.photovoltaic = (
            pl.read_csv(
                "dataset/data_photovoltaic_2022-01-01_2023-02-21.csv",
                separator=";",
                dtypes={t: pl.Float32 for t in PhotovoltaicColumns.values()}
                | {"Timestamp": pl.Utf8, "DayOfWeek": pl.UInt8},
            )
            .select(PhotovoltaicColumns.values())
            .with_columns(
                pl.col("Timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%:z"),
            )
            .with_columns(
                pl.col("Timestamp").dt.weekday().alias("DayOfWeek"),
            )
            .set_sorted(BatteryColumns.TIMESTAMP)
        )

        self.wallboxes = (
            pl.read_csv(
                "dataset/data_wallboxes_2022-01-01_2023-02-21.csv",
                separator=";",
                dtypes={t: pl.Float32 for t in WallboxesColumns.values()}
                | {"Timestamp": pl.Utf8, "DayOfWeek": pl.UInt8},
            )
            .select(WallboxesColumns.values())
            .with_columns(
                pl.col("Timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%:z"),
            )
            .with_columns(
                pl.col("Timestamp").dt.weekday().alias("DayOfWeek"),
            )
            .set_sorted(BatteryColumns.TIMESTAMP)
        )

    def heads(self) -> None:
        print("Battery")
        print(self.battery.head(5))

        print("Grid")
        print(self.grid.head(5))

        print("Photovoltaic")
        print(self.photovoltaic.head(5))

        print("Wallboxes")
        print(self.wallboxes.head(5))


datasets = Datasets()
