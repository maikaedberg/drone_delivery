import argparse
from pathlib import Path

import gurobipy as gp
import pandas as pd

import constants


def _drone_time_for_order(order):
    if (
        constants.PENALIZED_DRONE_UNAVAILABILITY_TIME in order.index
        and pd.notna(order[constants.PENALIZED_DRONE_UNAVAILABILITY_TIME])
    ):
        return float(order[constants.PENALIZED_DRONE_UNAVAILABILITY_TIME])

    if pd.notna(order.get(constants.DRONE_UNAVAILABILITY_TIME)):
        return float(order[constants.DRONE_UNAVAILABILITY_TIME])

    if pd.notna(order.get(constants.DRONE_DELIVERY_DISTANCE)):
        return float(order[constants.DRONE_DELIVERY_DISTANCE]) / constants.SPEED_DRONE * 60

    return 0.0


def _moped_time_for_order(order):
    if pd.notna(order.get(constants.MOPED_UNAVAILABILITY_TIME)):
        return float(order[constants.MOPED_UNAVAILABILITY_TIME])

    if pd.notna(order.get(constants.MOPED_DELIVERY_DISTANCE)):
        return float(order[constants.MOPED_DELIVERY_DISTANCE]) / constants.SPEED_MOPED * 60

    return 0.0


def _drone_unavailability_from_geodesic(geodesic_distance_km):
    flight_distance_km = float(geodesic_distance_km or 0.0) * 2
    flight_time = flight_distance_km / constants.SPEED_DRONE * 60
    charge_time = (flight_distance_km / constants.FULL_CHARGE_DIST) * constants.FULL_CHARGE_TIME
    return flight_time + charge_time + 1


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


def _count_moped_fleets(restaurant_df):
    service_times = sorted(
        restaurant_df[constants.MOPED_UNAVAILABILITY_TIME].tolist(),
        reverse=True,
    )
    loads = []
    bins = []

    for service_time in service_times:
        for index, load in enumerate(loads):
            if load + service_time <= constants.TIME_BUCKET:
                bins[index].append(service_time)
                loads[index] += service_time
                break
        else:
            bins.append([service_time])
            loads.append(service_time)

    return len(bins)


def find_minimal_mixed_fleet(delivery_df, use_no_fly_zone=True):
    mixed_fleet = {}

    for restaurant, restaurant_df in delivery_df.groupby(constants.RESTAURANT_NAME):
        if use_no_fly_zone and (restaurant_df[constants.NO_FLY_STATUS] == constants.STATUS_NOGO).all():
            moped_fleet_count = _count_moped_fleets(restaurant_df)
            mixed_fleet[restaurant] = {
                "drones": 0,
                "mopeds": moped_fleet_count,
                "hourly_cost": round(moped_fleet_count * constants.MOPED_COST_PER_HOUR),
            }
            continue

        if not use_no_fly_zone:
            restaurant_df = restaurant_df.copy()
            restaurant_df[constants.NO_FLY_STATUS] = constants.STATUS_CLEAR
            restaurant_df[constants.DRONE_DELIVERY_DISTANCE] = restaurant_df[constants.GEODESIC_DIST].fillna(0)
            restaurant_df[constants.DRONE_UNAVAILABILITY_TIME] = restaurant_df[constants.GEODESIC_DIST].apply(
                _drone_unavailability_from_geodesic
            )
            restaurant_df[constants.PENALIZED_DRONE_UNAVAILABILITY_TIME] = restaurant_df[
                constants.DRONE_UNAVAILABILITY_TIME
            ]

        buckets = [_bucket.reset_index(drop=True) for _bucket in _bucket_orders(restaurant_df)]
        max_orders_in_bucket = max(len(bucket) for bucket in buckets)
        model = gp.Model(f"mixed_delivery_{restaurant}")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = 20

        drone_fleet = model.addVars(
            max_orders_in_bucket,
            vtype=gp.GRB.BINARY,
            name="drone_fleet",
        )
        moped_fleet = model.addVars(
            max_orders_in_bucket,
            vtype=gp.GRB.BINARY,
            name="moped_fleet",
        )

        for fleet in (drone_fleet, moped_fleet):
            for fleet_index in range(max_orders_in_bucket - 1):
                model.addConstr(
                    fleet[fleet_index] >= fleet[fleet_index + 1]
                )

        for bucket_index, bucket in enumerate(buckets):
            drone_assignment = model.addVars(
                len(bucket),
                max_orders_in_bucket,
                vtype=gp.GRB.BINARY,
                name=f"drone_assignment_{bucket_index}",
            )
            moped_assignment = model.addVars(
                len(bucket),
                max_orders_in_bucket,
                vtype=gp.GRB.BINARY,
                name=f"moped_assignment_{bucket_index}",
            )

            for order_position, order in bucket.iterrows():
                model.addConstr(
                    gp.quicksum(
                        drone_assignment[order_position, fleet_index]
                        + moped_assignment[order_position, fleet_index]
                        for fleet_index in range(max_orders_in_bucket)
                    ) == 1
                )

                if use_no_fly_zone and order[constants.NO_FLY_STATUS] == constants.STATUS_NOGO:
                    for fleet_index in range(max_orders_in_bucket):
                        model.addConstr(drone_assignment[order_position, fleet_index] == 0)

            for fleet_index in range(max_orders_in_bucket):
                model.addConstr(
                    gp.quicksum(
                        drone_assignment[order_index, fleet_index]
                        * _drone_time_for_order(bucket.iloc[order_index])
                        for order_index in range(len(bucket))
                    ) <= constants.TIME_BUCKET * drone_fleet[fleet_index]
                )
                model.addConstr(
                    gp.quicksum(
                        moped_assignment[order_index, fleet_index]
                        * _moped_time_for_order(bucket.iloc[order_index])
                        for order_index in range(len(bucket))
                    ) <= constants.TIME_BUCKET * moped_fleet[fleet_index]
                )

        model.setObjective(
            constants.DRONE_COST_PER_HOUR * drone_fleet.sum()
            + constants.MOPED_COST_PER_HOUR * moped_fleet.sum(),
            gp.GRB.MINIMIZE,
        )
        model.optimize()

        if model.SolCount == 0:
            raise RuntimeError(f"Gurobi did not find a feasible solution for {restaurant}")

        num_drones = round(sum(drone_fleet[index].X for index in range(max_orders_in_bucket)))
        num_mopeds = round(sum(moped_fleet[index].X for index in range(max_orders_in_bucket)))
        mixed_fleet[restaurant] = {
            "drones": num_drones,
            "mopeds": num_mopeds,
            "hourly_cost": round(model.ObjVal),
        }

    return mixed_fleet


def analyse_mixed_fleet(fname, use_no_fly_zone=True):
    delivery_df = pd.read_csv(fname)
    mixed_fleet = find_minimal_mixed_fleet(
        delivery_df,
        use_no_fly_zone=use_no_fly_zone,
    )

    for restaurant, fleet in mixed_fleet.items():
        print(
            f"Restaurant: {restaurant}, Drones: {fleet['drones']}, "
            f"Mopeds: {fleet['mopeds']}, Hourly Cost: {fleet['hourly_cost']} SEK"
        )

parser = argparse.ArgumentParser()
parser.add_argument("-f", default="delivery_locations.csv", help="Input filename")
parser.add_argument(
    "--ignore-no-fly-zone",
    action="store_true",
    help="Ignore the no-fly-zone restriction when assigning drones.",
)
args = parser.parse_args()

if Path(args.f).name != args.f:
    parser.error("-f must contain a filename only, not a directory path")

input_file = Path(__file__).resolve().parent.parent / "data" / args.f
analyse_mixed_fleet(input_file, use_no_fly_zone=not args.ignore_no_fly_zone)

