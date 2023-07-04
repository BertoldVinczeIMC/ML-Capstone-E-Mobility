from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from aeon.clustering.k_means import TimeSeriesKMeans
from matplotlib import pyplot as plt
from data import WallboxesColumns, datasets
from logging import log
from typing import TypedDict


class Event(TypedDict):
    start: list[float]
    end: list[float]
    data: list[np.ndarray]


def capture_charging_activity(
    df: pl.DataFrame,
    station: WallboxesColumns = WallboxesColumns.KEBA_ONE,
) -> list[Event]:
    """
    Captures the charging activity of a station

    If the power is greater than 1 the car is charging,
    the event ends at the next timestamp where the power is 1.

    Events that are are shorter than a certain duration are discarded.
    """

    np_dataframe = df.select(station, "Timestamp").to_numpy()
    timestamps = df.select("Timestamp").to_numpy()

    """
    Events are tuples of all elements between the start and the end of the event
    configurations for each station
    [detect_charging_value, detect_end_value]
    """
    configurations: dict[WallboxesColumns, tuple[float, float]] = {
        WallboxesColumns.KEBA_ONE: (2, 2),
        WallboxesColumns.KEBA_TWO: (3, 0.5),
        WallboxesColumns.KEBA_THREE: (3, 0.5),
        WallboxesColumns.LADEBOX_ONE: (3, 0.5),
        WallboxesColumns.LADEBOX_TWO: (10, 6),
        WallboxesColumns.LADEBOX_THREE: (10, 6),
        WallboxesColumns.DELTA: (15, 10),
        WallboxesColumns.RAPTION: (30, 25),
    }

    # Get the configuration for the station
    high_threshold, low_threshold = configurations[station]

    events = list[Event]()
    charging = True
    singular_event = list[float]()
    count = 0

    for data in np_dataframe:
        if charging:
            # Append point to the event
            if data[0] > high_threshold:
                charging = True
                singular_event.append(data)

            # End of the event
            elif data[0] < low_threshold:
                charging = False

                # An event is qualified as a dictionary, containing the following 3 keys:
                # start: the timestamp of the first point of the event
                # end: the timestamp of the last point of the event
                # data: the data points of the event
                events.append(
                    {
                        "start": timestamps[count - len(singular_event)],
                        "end": timestamps[count],
                        "data": singular_event,  # type: ignore
                    }
                )

                # Reset the current event
                singular_event = []
        else:
            # Start of the event
            if data[0] > high_threshold:
                charging = True
                singular_event.append(data)

        count += 1

    return events

#this is the function that plots the events and also looks how does the wallbox behave after chargind event
def spot_decrease(
    event, event_timestamp, counter: int, name: str, everything: list
) -> None:
    int_to_day = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    # peak = highest value in the event

    if len(event) == 0:
        return

    # first drop means when we go from peak by at least 2% to the next value
    first_drop = event[0][0] - (event[0][0] * 0.015)
    for i in range(len(event)):
        if event[i][0] < first_drop:
            first_drop = i
            break

    # return the all the points between the start of the event and the first drop

    y_vals = []

    for i in range(len(event)):
        y_vals.append(event[i][0])

    try:
        if name == "LEM.KEBA_P30_1.Wirkleistung_P":
            our_range = y_vals[0:first_drop]
            # plt.figure(figsize=(10, 7))

            # make evenet_timestamp a pd.Timestamp
            event_timestamp = pd.Timestamp(event_timestamp)

            # plt.title(
            #     f"{counter}. event for {name} at {event_timestamp} ({int_to_day[event_timestamp.dayofweek]})"
            # )
            # plt.xlabel("Lifetime of the event")
            # plt.ylabel("Power")
            # # plot our range
            # plt.plot(range(len(our_range)), our_range)
            # plt.show()
            # plt the dropping event

            range_drop_end = y_vals[first_drop:]
            everything.append(range_drop_end)

            # plt.figure(figsize=(10, 7))
            # plt.title(
            #     f"Dropping event for {name} at {event_timestamp} ({int_to_day[event_timestamp.dayofweek]})"
            # )
            # plt.xlabel("Lifetime of the event")
            # plt.ylabel("Power")
            # plt.plot(range(len(range_drop_end)), range_drop_end)
            # plt.show()

    except Exception as e:
        log(level=2, msg=f"Error while plotting: {e}")


if __name__ == "__main__":
    wallboxes = list(WallboxesColumns)
    wallboxes.remove(WallboxesColumns.TIMESTAMP)

    df = datasets.wallboxes

    lengths = []

    # We pre-calculate the longest event for each station
    # for convenience
    longest = {
        WallboxesColumns.DELTA: 266,
        WallboxesColumns.KEBA_ONE: 271,
        WallboxesColumns.KEBA_TWO: 250,
        WallboxesColumns.KEBA_THREE: 1129,
        WallboxesColumns.LADEBOX_ONE: 1634,
        WallboxesColumns.LADEBOX_TWO: 367,
        WallboxesColumns.LADEBOX_THREE: 378,
        WallboxesColumns.RAPTION: 125,
    }

    # Normalised charging events
    daily_events = {wallbox: [] for wallbox in wallboxes}

    # Statistics and normalisation
    for wallbox in wallboxes:
        everything = []
        counter_drops = -1
        # Get the charging events for the station
        events = capture_charging_activity(df=df, station=wallbox)

        # Iterate over the events 
        for event in events:
            counter_drops += 1
            lengths.append(len(event))

            # adding drops to everything inside the fu
            spot_decrease(
                event=event["data"],
                event_timestamp=(event["start"][0]),
                counter=counter_drops,
                name=wallbox,
                everything=everything,
            )

            data = event["data"]

            L = len(data)

            if L < 1:
                continue

            for idx, dp in enumerate(data):
                data[idx] = np.array([dp[0]])

            for i in range(L, longest[wallbox]):
                data.append(np.array([0]))

            _data = np.array(data)

            daily_events[wallbox].append(_data)

    for wallbox in wallboxes:
        kmeans = TimeSeriesKMeans(
            n_clusters=3, metric="dtw", max_iter=5, random_state=0
        )

        fitted = kmeans.fit_predict(np.array(daily_events[wallbox]))

        clusters = [np.array(daily_events[wallbox])[fitted == i] for i in range(3)]

        for idx, cluster in enumerate(clusters):
            plt.figure(figsize=(10, 7))
            plt.title(f"Cluster #{idx+1} for {wallbox}")
            plt.xlabel("Event Duration (in seconds)")
            plt.ylabel("Power (in kW)")

            for line in cluster:
                plt.plot(line, color="red", alpha=0.1)

            plt.savefig(f"plots/spike_clustering/cluster_{idx+1}_{wallbox}.png")
            plt.close()
