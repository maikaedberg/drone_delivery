import math
import constants
import pandas as pd

def erlang_c(s, a):
    if s <= a:
        return 1.0
    tail = (a ** s / math.factorial(s)) * (s / (s - a))
    normalizer = sum(a ** k / math.factorial(k) for k in range(s)) + tail
    return tail / normalizer


def minimum_drones_for_asa(lambda_per_hour, mean_service_minutes, target_asa_minutes):
    mu_per_hour = 60.0 / mean_service_minutes
    offered_load = lambda_per_hour / mu_per_hour
    s = max(1, math.floor(offered_load) + 1)
    while True:
        wait_probability = erlang_c(s, offered_load)
        asa_minutes = 60.0 * wait_probability / (s * mu_per_hour - lambda_per_hour)
        if asa_minutes <= target_asa_minutes:
            return s
        s += 1
def tsf(s, lambda_per_hour, ts_minutes, T_minutes):
    mu = 60 / ts_minutes
    a = lambda_per_hour / mu

    if s <= a:
        return 0.0

    return 1 - erlang_c(s, a) * math.exp(-(s * mu - lambda_per_hour) * T_minutes / 60)


def find_min_qt(n_orders_per_hour, avg_service_time):
    target_asa = constants.TIME_BUCKET
    fleets = minimum_drones_for_asa(n_orders_per_hour, avg_service_time, target_asa)
    return fleets

def analyse_minimum_fleets(fname):
           
    delivery_df = pd.read_csv(fname)
    for restaurant, restaurant_df in delivery_df.groupby(constants.RESTAURANT_NAME):
        # find the minimum number of drones and moped required
        n_orders_per_hour = len(restaurant_df) // constants.TOTAL_WINDOW_HOURS
        avg_service_time_drone = restaurant_df[constants.DRONE_UNAVAILABILITY_TIME].mean()
        avg_service_time_moped = restaurant_df[constants.MOPED_UNAVAILABILITY_TIME].mean()
        min_num_drone_fleet = find_min_qt(n_orders_per_hour, avg_service_time_drone)
        min_num_moped_fleet = find_min_qt(n_orders_per_hour, avg_service_time_moped)

        print(f"Restaurant: {restaurant}, Minimum Drones Required: {min_num_drone_fleet}")
        print(f"Restaurant: {restaurant}, Minimum Mopeds Required: {min_num_moped_fleet}")

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    default="delivery_locations.csv",
    help="Input filename"
)
args = parser.parse_args()

if Path(args.f).name != args.f:
    parser.error("-f must contain a filename only, not a directory path")

input_file = Path(__file__).resolve().parent.parent / "data" / args.f

analyse_minimum_fleets(input_file)