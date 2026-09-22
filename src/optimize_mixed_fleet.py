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
        bucket = restaurant_df.iloc[start_index:end_index].copy()
        bucket["_order_index"] = bucket.index
        bucket = bucket.reset_index(drop=True)
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


def find_minimal_mixed_fleet(delivery_df, use_no_fly_zone=True, give_assignment=False):
    mixed_fleet = {}
    drone_time_col = (
        constants.PENALIZED_DRONE_UNAVAILABILITY_TIME
        if use_no_fly_zone
        else constants.DRONE_UNAVAILABILITY_TIME
    )

    for restaurant, restaurant_df in delivery_df.groupby(constants.RESTAURANT_NAME):
        print(
            f"Optimizing for restaurant {restaurant} with "
            f"{len(restaurant_df)} orders"
        )

        restaurant_result = {
            "drones": 0,
            "mopeds": 0,
            "hourly_cost": 0,
        }
        assignments = []

        for bucket in _bucket_orders(restaurant_df):
            n_orders = len(bucket)
            n_possible_fleets = n_orders
            model = gp.Model(f"mixed_delivery_{restaurant}")
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = 20

            drone_fleet = model.addVars(
                n_possible_fleets,
                vtype=gp.GRB.BINARY,
                name="drone_fleet",
            )
            moped_fleet = model.addVars(
                n_possible_fleets,
                vtype=gp.GRB.BINARY,
                name="moped_fleet",
            )

            for fleet in (drone_fleet, moped_fleet):
                for fleet_index in range(n_possible_fleets - 1):
                    model.addConstr(fleet[fleet_index] >= fleet[fleet_index + 1])

            drone_assignment = model.addVars(
                n_orders,
                n_possible_fleets,
                vtype=gp.GRB.BINARY,
                name="drone_assignment",
            )
            moped_assignment = model.addVars(
                n_orders,
                n_possible_fleets,
                vtype=gp.GRB.BINARY,
                name="moped_assignment",
            )
            drone_times = bucket[drone_time_col].fillna(0).tolist()
            drone_eligible = bucket[drone_time_col].notna().tolist()
            moped_times = bucket[constants.MOPED_UNAVAILABILITY_TIME].tolist()

            for order_index in range(n_orders):
                model.addConstr(
                    gp.quicksum(
                        drone_assignment[order_index, fleet_index]
                        + moped_assignment[order_index, fleet_index]
                        for fleet_index in range(n_possible_fleets)
                    ) == 1
                )

                if use_no_fly_zone and not drone_eligible[order_index]:
                    for fleet_index in range(n_possible_fleets):
                        model.addConstr(
                            drone_assignment[order_index, fleet_index] == 0
                        )

            for fleet_index in range(n_possible_fleets):
                model.addConstr(
                    gp.quicksum(
                        drone_assignment[order_index, fleet_index]
                        * drone_times[order_index]
                        for order_index in range(n_orders)
                    ) <= constants.TIME_BUCKET * drone_fleet[fleet_index]
                )
                model.addConstr(
                    gp.quicksum(
                        moped_assignment[order_index, fleet_index]
                        * moped_times[order_index]
                        for order_index in range(n_orders)
                    ) <= constants.TIME_BUCKET * moped_fleet[fleet_index]
                )

            model.setObjective(
                constants.DRONE_COST_PER_HOUR * drone_fleet.sum()
                + constants.MOPED_COST_PER_HOUR * moped_fleet.sum(),
                gp.GRB.MINIMIZE,
            )
            model.optimize()

            if model.SolCount == 0:
                raise RuntimeError(
                    f"Gurobi did not find a feasible solution for {restaurant}"
                )

            num_drones = round(
                sum(
                    drone_fleet[index].X
                    for index in range(n_possible_fleets)
                )
            )
            num_mopeds = round(
                sum(
                    moped_fleet[index].X
                    for index in range(n_possible_fleets)
                )
            )
            restaurant_result["drones"] = max(
                restaurant_result["drones"],
                num_drones,
            )
            restaurant_result["mopeds"] = max(
                restaurant_result["mopeds"],
                num_mopeds,
            )
            restaurant_result["hourly_cost"] = round(
                restaurant_result["drones"] * constants.DRONE_COST_PER_HOUR
                + restaurant_result["mopeds"] * constants.MOPED_COST_PER_HOUR
            )

            if give_assignment:
                for order_index, order in bucket.iterrows():
                    vehicle = "moped"
                    for fleet_index in range(n_possible_fleets):
                        if drone_assignment[order_index, fleet_index].X > 0.5:
                            vehicle = "drone"
                            break
                    assignments.append(
                        {
                            "order_index": order["_order_index"],
                            "vehicle": vehicle,
                        }
                    )

                if give_assignment:
                    mixed_fleet[restaurant] = {
                        **restaurant_result,
                        "assignments": assignments,
                    }
                else:
                    mixed_fleet[restaurant] = restaurant_result

    return mixed_fleet


def analyse_mixed_fleet(
    fname,
    use_no_fly_zone=True,
    give_assignment=False,
    assignment_file="assignments.csv",
):
    delivery_df = pd.read_csv(fname)
    mixed_fleet = find_minimal_mixed_fleet(
        delivery_df,
        use_no_fly_zone=use_no_fly_zone,
        give_assignment=give_assignment,
    )

    if give_assignment:
        assignment_rows = []
        for restaurant, fleet in mixed_fleet.items():
            for assignment in fleet["assignments"]:
                order = delivery_df.loc[assignment["order_index"]]
                assignment_rows.append(
                    {
                        "restaurant": restaurant,
                        "order_index": assignment["order_index"],
                        "latitude": order[constants.LATITUDE],
                        "longitude": order[constants.LONGITUDE],
                        "vehicle": assignment["vehicle"],
                    }
                )
        pd.DataFrame(assignment_rows).sort_values(
            [constants.RESTAURANT_NAME, "order_index"]
        ).to_csv(assignment_file, index=False)
        print(f"Assignments written to {assignment_file}")

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
parser.add_argument(
    "--give-assignment",
    action="store_true",
    help="Write each order's vehicle assignment to a CSV file.",
)
parser.add_argument(
    "--assignment-file",
    default="assignments.csv",
    help="Output filename for order assignments.",
)
args = parser.parse_args()

if Path(args.f).name != args.f:
    parser.error("-f must contain a filename only, not a directory path")
if Path(args.assignment_file).name != args.assignment_file:
    parser.error("--assignment-file must contain a filename only, not a directory path")

input_file = Path(__file__).resolve().parent.parent / "data" / args.f
assignment_file = Path(__file__).resolve().parent.parent / "data" / args.assignment_file
analyse_mixed_fleet(
    input_file,
    use_no_fly_zone=not args.ignore_no_fly_zone,
    give_assignment=args.give_assignment,
    assignment_file=assignment_file,
)

