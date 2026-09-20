import argparse
from pathlib import Path

import gurobipy as gp
import pandas as pd

import constants


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


def find_minimal_mixed_fleet(delivery_df):
    mixed_fleet = {}

    for restaurant, restaurant_df in delivery_df.groupby(constants.RESTAURANT_NAME):
        buckets = _bucket_orders(restaurant_df)
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

            for order_index, order in bucket.iterrows():
                model.addConstr(
                    gp.quicksum(
                        drone_assignment[order_index, fleet_index]
                        + moped_assignment[order_index, fleet_index]
                        for fleet_index in range(max_orders_in_bucket)
                    ) == 1
                )

                if order[constants.NO_FLY_STATUS] == constants.STATUS_NOGO:
                    for fleet_index in range(max_orders_in_bucket):
                        model.addConstr(drone_assignment[order_index, fleet_index] == 0)

            for fleet_index in range(max_orders_in_bucket):
                model.addConstr(
                    gp.quicksum(
                        drone_assignment[order_index, fleet_index]
                        * (
                            bucket.iloc[order_index][constants.DRONE_UNAVAILABILITY_TIME]
                            if pd.notna(
                                bucket.iloc[order_index][constants.DRONE_UNAVAILABILITY_TIME]
                            )
                            else 0
                        )
                        for order_index in range(len(bucket))
                    ) <= constants.TIME_BUCKET * drone_fleet[fleet_index]
                )
                model.addConstr(
                    gp.quicksum(
                        moped_assignment[order_index, fleet_index]
                        * bucket.iloc[order_index][constants.MOPED_UNAVAILABILITY_TIME]
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


def analyse_mixed_fleet(fname):
    delivery_df = pd.read_csv(fname)
    mixed_fleet = find_minimal_mixed_fleet(delivery_df)

    for restaurant, fleet in mixed_fleet.items():
        print(
            f"Restaurant: {restaurant}, Drones: {fleet['drones']}, "
            f"Mopeds: {fleet['mopeds']}, Hourly Cost: {fleet['hourly_cost']} SEK"
        )


parser = argparse.ArgumentParser()
parser.add_argument("-f", default="delivery_locations.csv", help="Input filename")
args = parser.parse_args()

if Path(args.f).name != args.f:
    parser.error("-f must contain a filename only, not a directory path")

input_file = Path(__file__).resolve().parent.parent / "data" / args.f
analyse_mixed_fleet(input_file)