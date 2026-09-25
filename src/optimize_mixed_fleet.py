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


def assign_buckets(delivery_df):
    total_num_buckets = constants.TOTAL_TIME // constants.TIME_BUCKET
    delivery_df = delivery_df.copy()
    delivery_df["_bucket_index"] = pd.Series(
        pd.NA,
        index=delivery_df.index,
        dtype="Int64",
    )

    for _, restaurant_df in delivery_df.groupby(
        constants.RESTAURANT_NAME,
        sort=False,
    ):
        bucket_size = (
            len(restaurant_df) // total_num_buckets
            + (len(restaurant_df) % total_num_buckets > 0)
        )

        for bucket_index in range(total_num_buckets):
            start_index = bucket_index * bucket_size
            end_index = (bucket_index + 1) * bucket_size
            order_indices = restaurant_df.iloc[start_index:end_index].index
            delivery_df.loc[order_indices, "_bucket_index"] = bucket_index

    return delivery_df

def greedy_num_fleets(delivery_df, service_time_col):
    num_fleets = 0
    for _, bucket_df in delivery_df.groupby("_bucket_index"):
        service_time = bucket_df[service_time_col].dropna().to_list()
        greedy_bins = greedy_packing(service_time, constants.TIME_BUCKET)
        num_fleets = max(len(greedy_bins), num_fleets)
    return num_fleets


def greedy_mixed_start(
    delivery_df,
    drone_service_time_col,
    moped_service_time_col,
):
    drone_assignments = {}
    moped_assignments = {}
    n_drones = 0
    n_mopeds = 0

    for _, bucket_df in delivery_df.groupby("_bucket_index"):
        drone_orders = [
            order_index
            for order_index in bucket_df.index
            if pd.notna(bucket_df.loc[order_index, drone_service_time_col])
        ]
        moped_orders = [
            order_index
            for order_index in bucket_df.index
            if order_index not in drone_orders
        ]

        drone_bins = greedy_packing(
            [bucket_df.loc[i, drone_service_time_col] for i in drone_orders],
            constants.TIME_BUCKET,
        )
        moped_bins = greedy_packing(
            [bucket_df.loc[i, moped_service_time_col] for i in moped_orders],
            constants.TIME_BUCKET,
        )

        n_drones = max(n_drones, len(drone_bins))
        n_mopeds = max(n_mopeds, len(moped_bins))

        for fleet_index, bin_orders in enumerate(drone_bins):
            for local_index in bin_orders:
                drone_assignments[drone_orders[local_index]] = fleet_index
        for fleet_index, bin_orders in enumerate(moped_bins):
            for local_index in bin_orders:
                moped_assignments[moped_orders[local_index]] = fleet_index

    return n_drones, n_mopeds, drone_assignments, moped_assignments
        
def filter_impossible_orders(df, drone_service_time_col, moped_service_time_col):
    filtered_df = df[
        (df[drone_service_time_col] <= constants.TIME_BUCKET) |
        (df[moped_service_time_col] <= constants.TIME_BUCKET) 
    ] 
    dropped_orders = len(df) - len(filtered_df)
    return filtered_df, dropped_orders

def get_service_time_col(fleet_type, use_no_fly_zone):

    if fleet_type == constants.FLEET_TYPE_DRONE:
        if not use_no_fly_zone:
            return constants.DRONE_UNAVAILABILITY_TIME
        else:
            return constants.PENALIZED_DRONE_UNAVAILABILITY_TIME
    else:
        return constants.MOPED_UNAVAILABILITY_TIME

def find_minimal_fleet(delivery_df, use_no_fly_zone=True, restaurant=None):
    min_fleet = {}

    drone_service_time_col = get_service_time_col(constants.FLEET_TYPE_DRONE, use_no_fly_zone)
    moped_service_time_col = get_service_time_col(constants.FLEET_TYPE_MOPED, use_no_fly_zone)
    delivery_df, dropped_orders = filter_impossible_orders(
        delivery_df,
        drone_service_time_col, moped_service_time_col
    )

    if restaurant is not None:
        delivery_df = delivery_df[
            delivery_df[constants.RESTAURANT_NAME] == restaurant
        ]
        if delivery_df.empty:
            raise ValueError(f"Restaurant not found: {restaurant}")
    
    print(f"WARNING: {dropped_orders} orders were dropped.")

    for restaurant, restaurant_df in delivery_df.groupby(constants.RESTAURANT_NAME):
        print(
            f"Optimizing for restaurant {restaurant} with {len(restaurant_df)} orders"
        )
        restaurant_df = assign_buckets(restaurant_df).reset_index(drop=True)

        n_orders = len(restaurant_df)

        drone_eligible = restaurant_df[drone_service_time_col].notna().to_list()
        drone_service_time = restaurant_df[drone_service_time_col].fillna(0).to_list()
        moped_service_time = restaurant_df[moped_service_time_col].to_list()
        
        n_possible_drones = greedy_num_fleets(
            restaurant_df,
            drone_service_time_col,
        )
        n_possible_mopeds = greedy_num_fleets(
            restaurant_df,
            moped_service_time_col,
        )
        _, _, drone_start, moped_start = greedy_mixed_start(
            restaurant_df,
            drone_service_time_col,
            moped_service_time_col,
        )
        
        model = gp.Model("delivery")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = 120

        x_drone = model.addVars(
            n_orders, n_possible_drones,
            vtype=gp.GRB.BINARY, name="x_drone"
        )
        x_moped = model.addVars(
            n_orders, n_possible_mopeds,
            vtype=gp.GRB.BINARY, name="x_moped"
        )
        y_drone = model.addVars(
            n_possible_drones,
            vtype=gp.GRB.BINARY, name="y_drone"
        )
        y_moped = model.addVars(
            n_possible_mopeds,
            vtype=gp.GRB.BINARY, name="y_moped"
        )

        for order_index, fleet_index in drone_start.items():
            x_drone[order_index, fleet_index].Start = 1
        for order_index, fleet_index in moped_start.items():
            x_moped[order_index, fleet_index].Start = 1
        for fleet_index in range(n_possible_drones):
            y_drone[fleet_index].Start = int(
                fleet_index < max(drone_start.values(), default=-1) + 1
            )
        for fleet_index in range(n_possible_mopeds):
            y_moped[fleet_index].Start = int(
                fleet_index < max(moped_start.values(), default=-1) + 1
            )

        for order_index in range(n_orders):
            model.addConstr(
                gp.quicksum(
                    x_drone[order_index, fleet_index]
                    for fleet_index in range(n_possible_drones)
                ) + 
                gp.quicksum(
                    x_moped[order_index, fleet_index]
                    for fleet_index in range(n_possible_mopeds)
                ) == 1
            )

        for order_index in range(n_orders):
            for fleet_index in range(n_possible_drones):
                model.addConstr(
                    x_drone[order_index, fleet_index] <= y_drone[fleet_index]
                )
            for fleet_index in range(n_possible_mopeds):
                model.addConstr(
                    x_moped[order_index, fleet_index] <= y_moped[fleet_index]
                )
            if not drone_eligible[order_index]:
                for fleet_index in range(n_possible_drones):
                    model.addConstr(x_drone[order_index, fleet_index] == 0)

        for _, bucket_df in restaurant_df.groupby("_bucket_index"):
            for fleet_index in range(n_possible_drones):
                model.addConstr(
                    gp.quicksum(
                        drone_service_time[order_index]
                        * x_drone[order_index, fleet_index]
                        for order_index in bucket_df.index
                    )
                    <= constants.TIME_BUCKET * y_drone[fleet_index]
                )
            for fleet_index in range(n_possible_mopeds):
                model.addConstr(
                    gp.quicksum(
                        moped_service_time[order_index]
                        * x_moped[order_index, fleet_index]
                        for order_index in bucket_df.index
                    )
                    <= constants.TIME_BUCKET * y_moped[fleet_index]
                )

        for fleet_index in range(n_possible_drones - 1):
            model.addConstr(y_drone[fleet_index] >= y_drone[fleet_index + 1])
        for fleet_index in range(n_possible_mopeds - 1):
            model.addConstr(y_moped[fleet_index] >= y_moped[fleet_index + 1])

        model.setObjective(
            gp.quicksum(
                y_drone[fleet_index] * constants.DRONE_COST_PER_HOUR
                for fleet_index in range(n_possible_drones)
            ) +
            gp.quicksum(
                y_moped[fleet_index] * constants.MOPED_COST_PER_HOUR
                for fleet_index in range(n_possible_mopeds)
            ) ,
            gp.GRB.MINIMIZE
        )
        model.optimize()

        if model.SolCount == 0:
            raise RuntimeError("Gurobi did not find a feasible solution")

        if model.Status == gp.GRB.TIME_LIMIT:
            print(
                f"Time limit reached for {restaurant}; "
                f"best cost: {model.ObjVal}, bound: {model.ObjBound}"
            )

        num_drones = round(
            sum(y_drone[fleet_index].X for fleet_index in range(n_possible_drones))
        )
        num_mopeds = round(
            sum(y_moped[fleet_index].X for fleet_index in range(n_possible_mopeds))
        )

        min_fleet[restaurant] = {
            "drones": num_drones,
            "mopeds": num_mopeds,
            "hourly_cost": round(model.ObjVal),
        }

    return min_fleet


def analyse_minimum_fleets(fname, use_no_fly_zone=True, restaurant=None):
    delivery_df = pd.read_csv(fname)
    min_mixed_fleet = find_minimal_fleet(
        delivery_df,
        use_no_fly_zone=use_no_fly_zone,
        restaurant=restaurant,
    )

    for restaurant, fleet in min_mixed_fleet.items():
        print(
            f"Restaurant: {restaurant}, Drones: {fleet['drones']}, "
            f"Mopeds: {fleet['mopeds']}, "
            f"Hourly Cost: {fleet['hourly_cost']} SEK"
        )


parser = argparse.ArgumentParser()
parser.add_argument("-f", default="delivery_locations.csv", help="Input filename")
parser.add_argument(
    "--ignore-no-fly-zone",
    action="store_true",
    help="Ignore the no-fly-zone restriction when checking drone feasibility.",
)
parser.add_argument(
    "--restaurant",
    help="Optimize only this restaurant.",
)
args = parser.parse_args()

if Path(args.f).name != args.f:
    parser.error("-f must contain a filename only, not a directory path")

input_file = Path(__file__).resolve().parent.parent / "data" / args.f
analyse_minimum_fleets(
    input_file,
    use_no_fly_zone=not args.ignore_no_fly_zone,
    restaurant=args.restaurant,
)
