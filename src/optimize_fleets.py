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

def greedy_fleet_start(delivery_df, service_time_col):
    num_fleets = 0
    assignments = {}

    for _, bucket_df in delivery_df.groupby("_bucket_index"):
        service_time = bucket_df[service_time_col].to_list()
        greedy_bins = greedy_packing(service_time, constants.TIME_BUCKET)
        num_fleets = max(len(greedy_bins), num_fleets)

        for fleet_index, bin_orders in enumerate(greedy_bins):
            for order_index in bin_orders:
                assignments[bucket_df.index[order_index]] = fleet_index

    return num_fleets, assignments
        
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

def find_minimal_fleet(
    delivery_df,
    fleet_type,
    use_no_fly_zone=True,
    restaurant=None,
):
    min_fleet = {}

    service_time_col = get_service_time_col(fleet_type, use_no_fly_zone)
    delivery_df, dropped_orders = filter_impossible_orders(
        delivery_df,
        service_time_col
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
            f"Optimizing for restaurant {restaurant} with "
            f"{len(restaurant_df)} orders, fleet_type={fleet_type}"
        )
        restaurant_df = assign_buckets(restaurant_df).reset_index(drop=True)

        n_orders = len(restaurant_df)

        service_time = restaurant_df[service_time_col].to_list()

        n_possible_fleets, greedy_assignments = greedy_fleet_start(
            restaurant_df,
            service_time_col,
        )

        lower_bound = (
            max(
                (
                    bucket_df[service_time_col].sum()
                    + constants.TIME_BUCKET
                    - 1
                )
                // constants.TIME_BUCKET
                for _, bucket_df in restaurant_df.groupby("_bucket_index")
            )
        )


        model = gp.Model("delivery")
        model.Params.OutputFlag = 0

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

        for order_index, fleet_index in greedy_assignments.items():
            x[order_index, fleet_index].Start = 1
        for fleet_index in range(n_possible_fleets):
            y[fleet_index].Start = int(
                fleet_index < max(greedy_assignments.values()) + 1
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

        for _, bucket_df in restaurant_df.groupby("_bucket_index"):
            for fleet_index in range(n_possible_fleets):
                model.addConstr(
                    gp.quicksum(
                        service_time[order_index]
                        * x[order_index, fleet_index]
                        for order_index in bucket_df.index
                    )
                    <= constants.TIME_BUCKET * y[fleet_index]
                )

        for fleet_index in range(n_possible_fleets - 1):
            model.addConstr(y[fleet_index] >= y[fleet_index + 1])

        model.addConstr(
            gp.quicksum(
                y[fleet_index]
                for fleet_index in range(n_possible_fleets)
            ) >= lower_bound
        )

        model.setObjective(
            gp.quicksum(
                y[fleet_index]
                for fleet_index in range(n_possible_fleets)
            ),
            gp.GRB.MINIMIZE
        )
        model.Params.BestObjStop = lower_bound
        model.Params.TimeLimit = 60  # seconds

        model.optimize()

        if model.SolCount == 0:
            raise RuntimeError("No feasible solution found")

        if model.Status == gp.GRB.TIME_LIMIT:
            print(
                f"Time limit reached for {restaurant}; "
                f"best feasible fleet count: {model.ObjVal}, "
                f"bound: {model.ObjBound}"
            )

        min_num_fleet = round(model.ObjVal)

        min_fleet[restaurant] = min_num_fleet

    return min_fleet


def analyse_minimum_fleets(fname, use_no_fly_zone=True, restaurant=None):
    delivery_df = pd.read_csv(fname)
    min_drone_fleet = find_minimal_fleet(
        delivery_df,
        fleet_type="drone",
        use_no_fly_zone=use_no_fly_zone,
        restaurant=restaurant,
    )
    min_moped_fleet = find_minimal_fleet(
        delivery_df,
        fleet_type="moped",
        use_no_fly_zone=use_no_fly_zone,
        restaurant=restaurant,
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
