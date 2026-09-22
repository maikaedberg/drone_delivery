import argparse
from pathlib import Path

import gurobipy as gp
import pandas as pd

import constants


def greedy_packing(time_list, time_max):
    order = sorted(
        range(len(time_list)),
        key=lambda i: time_list[i],
        reverse=True
    )

    bins = []
    loads = []

    for i in order:
        for bucket_i, load in enumerate(loads):
            if load + time_list[i] <= time_max:
                bins[bucket_i].append(i)
                loads[bucket_i] += time_list[i]
                break
        else:
            bins.append([i])
            loads.append(time_list[i])

    return bins


def _bucket_orders(restaurant_df):
    total_num_buckets = constants.TOTAL_TIME // constants.TIME_BUCKET
    bucket_size = (
        len(restaurant_df) // total_num_buckets
        + (len(restaurant_df) % total_num_buckets > 0)
    )

    buckets = []
    for bucket_index in range(total_num_buckets):
        start_index = bucket_index * bucket_size
        end_index = (bucket_index + 1) * bucket_size
        bucket = restaurant_df.iloc[start_index:end_index].reset_index(drop=True)
        if len(bucket) > 0:
            buckets.append(bucket)

    return buckets


def filter_impossible_orders(delivery_df, service_time_col):
    filtered_df = delivery_df[
        delivery_df[service_time_col].notna()
        & (delivery_df[service_time_col] <= constants.TIME_BUCKET)
    ]
    dropped_orders = len(delivery_df) - len(filtered_df)
    return filtered_df, dropped_orders

def get_service_time_col(fleet_type, use_no_fly_zone):

    if fleet_type == "drone":
        if not use_no_fly_zone:
            return constants.DRONE_UNAVAILABILITY_TIME
        else:
            return constants.PENALIZED_DRONE_UNAVAILABILITY_TIME
    else:
        return constants.MOPED_UNAVAILABILITY_TIME

def find_minimal_fleet(delivery_df, fleet_type, use_no_fly_zone=True):
    min_fleet = {}

    service_time_col = get_service_time_col(fleet_type, use_no_fly_zone)
    delivery_df, dropped_orders = filter_impossible_orders(
        delivery_df,
        service_time_col
    )
    print(f"WARNING: {dropped_orders} orders were dropped.")

    for restaurant, restaurant_df in delivery_df.groupby(constants.RESTAURANT_NAME):
        print(
            f"Optimizing for restaurant {restaurant} with "
            f"{len(restaurant_df)} orders, fleet_type={fleet_type}"
        )

        for df in _bucket_orders(restaurant_df):
            n_orders = len(df)

            service_time = df[service_time_col].to_list()

            service_time.sort(reverse=True)
            greedy_bins = greedy_packing(service_time, constants.TIME_BUCKET)

            if restaurant in min_fleet:
                if min_fleet[restaurant] > len(greedy_bins):
                    continue

            n_possible_fleets = len(greedy_bins)
            lower_bound = (
                sum(service_time) + constants.TIME_BUCKET - 1
            ) // constants.TIME_BUCKET

            if n_possible_fleets == lower_bound:
                min_num_fleet = lower_bound
            else:
                model = gp.Model("delivery")
                model.Params.OutputFlag = 0
                model.Params.TimeLimit = 20

                x = model.addVars(
                    n_orders,
                    n_possible_fleets,
                    vtype=gp.GRB.BINARY,
                    name="x"
                )
                y = model.addVars(
                    n_possible_fleets,
                    vtype=gp.GRB.BINARY,
                    name="y"
                )

                for order_index in range(n_orders):
                    model.addConstr(
                        gp.quicksum(
                            x[order_index, fleet_index]
                            for fleet_index in range(n_possible_fleets)
                        ) == 1
                    )

                for order_index in range(n_orders):
                    for fleet_index in range(n_possible_fleets):
                        model.addConstr(
                            x[order_index, fleet_index] <= y[fleet_index]
                        )

                for fleet_index in range(n_possible_fleets - 1):
                    model.addConstr(y[fleet_index] >= y[fleet_index + 1])

                model.addConstr(
                    gp.quicksum(
                        y[fleet_index]
                        for fleet_index in range(n_possible_fleets)
                    ) >= lower_bound
                )

                for fleet_index, bucket in enumerate(greedy_bins):
                    y[fleet_index].Start = 1
                    for order_index in bucket:
                        x[order_index, fleet_index].Start = 1

                model.setObjective(
                    gp.quicksum(
                        y[fleet_index]
                        for fleet_index in range(n_possible_fleets)
                    ),
                    gp.GRB.MINIMIZE
                )
                model.Params.BestObjStop = lower_bound
                model.optimize()

                if model.SolCount == 0:
                    raise RuntimeError("Gurobi did not find a feasible solution")

                min_num_fleet = round(model.ObjVal)

            if restaurant in min_fleet:
                min_fleet[restaurant] = max(min_fleet[restaurant], min_num_fleet)
            else:
                min_fleet[restaurant] = min_num_fleet

    return min_fleet


def analyse_minimum_fleets(fname, use_no_fly_zone=True):
    delivery_df = pd.read_csv(fname)
    min_drone_fleet = find_minimal_fleet(
        delivery_df,
        fleet_type="drone",
        use_no_fly_zone=use_no_fly_zone,
    )
    min_moped_fleet = find_minimal_fleet(
        delivery_df,
        fleet_type="moped",
        use_no_fly_zone=use_no_fly_zone,
    )

    for restaurant, min_fleets in min_drone_fleet.items():
        print(f"Restaurant: {restaurant}, Minimum Drones Required: {min_fleets}")
    for restaurant, min_fleets in min_moped_fleet.items():
        print(f"Restaurant: {restaurant}, Minimum Mopeds Required: {min_fleets}")


parser = argparse.ArgumentParser()
parser.add_argument("-f", default="delivery_locations.csv", help="Input filename")
parser.add_argument(
    "--ignore-no-fly-zone",
    action="store_true",
    help="Ignore the no-fly-zone restriction when checking drone feasibility.",
)
args = parser.parse_args()

if Path(args.f).name != args.f:
    parser.error("-f must contain a filename only, not a directory path")

input_file = Path(__file__).resolve().parent.parent / "data" / args.f
analyse_minimum_fleets(input_file, use_no_fly_zone=not args.ignore_no_fly_zone)
