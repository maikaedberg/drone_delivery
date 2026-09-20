import pandas as pd
import gurobipy as gp
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

# FLEET MINIMIZATION MODEL
def find_minimal_fleet(delivery_df, fleet_type):
    min_fleet = {}

    if fleet_type == "drone":
        drop_indices = [i for i in delivery_df.index
                    if delivery_df.loc[i, constants.DRONE_UNAVAILABILITY_TIME] > constants.TIME_BUCKET or 
                    delivery_df.loc[i, constants.NO_FLY_STATUS] == constants.STATUS_NOGO]
        delivery_df = delivery_df.drop(drop_indices).reset_index(drop=True)
        print("WARNING: {} orders were dropped due to no-fly zones or exceeding the time bucket.".format(len(drop_indices)))
    elif fleet_type == "moped":
        drop_indices = [i for i in delivery_df.index
                    if delivery_df.loc[i, constants.MOPED_UNAVAILABILITY_TIME] > constants.TIME_BUCKET]
        delivery_df = delivery_df.drop(drop_indices).reset_index(drop=True)
        print("WARNING: {} orders were dropped due to exceeding the time bucket.".format(len(drop_indices)))
    else:
        raise ValueError("Invalid fleet type. Choose either 'drone' or 'moped'.")
    
    print("Remaining orders: {}".format(len(delivery_df)))

    total_num_buckets = constants.TOTAL_TIME // constants.TIME_BUCKET
    for restaurant, restaurant_df in delivery_df.groupby(constants.RESTAURANT_NAME):

        print(f"Optimizing for restaurant {restaurant} with {len(restaurant_df)} orders, fleet_type={fleet_type}")
        size_bucket = len(restaurant_df) // total_num_buckets + (len(restaurant_df) % total_num_buckets > 0)

        for t_i in range(total_num_buckets):

            # get the t_i : 
            start_index = t_i * size_bucket
            end_index = (t_i + 1) * size_bucket
            df = restaurant_df.iloc[start_index:end_index].reset_index(drop=True)

            n_orders = len(df)

            if n_orders == 0:
                continue

            if fleet_type == "drone":
                service_time = df[constants.DRONE_UNAVAILABILITY_TIME].tolist()
            else:
                service_time = df[constants.MOPED_UNAVAILABILITY_TIME].tolist()

            service_time.sort(reverse=True)

            greedy_bins = greedy_packing(
                service_time,
                constants.TIME_BUCKET
            )
            if restaurant in min_fleet:
                if min_fleet[restaurant] > len(greedy_bins):
                    continue

            n_possible_fleets = len(greedy_bins)

            # Valid lower bound: total required time divided by capacity
            lower_bound = (
                sum(service_time) + constants.TIME_BUCKET - 1
            ) // constants.TIME_BUCKET

            # The greedy solution is already provably optimal
            if n_possible_fleets == lower_bound:
                min_num_fleet = lower_bound
            else:
                m = gp.Model("delivery")
                m.Params.OutputFlag = 0
                m.Params.TimeLimit = 20

                x = m.addVars(
                    n_orders,
                    n_possible_fleets,
                    vtype=gp.GRB.BINARY,
                    name="x"
                )
                y = m.addVars(
                    n_possible_fleets,
                    vtype=gp.GRB.BINARY,
                    name="y"
                )

                for i in range(n_orders):
                    m.addConstr(
                        gp.quicksum(
                            x[i, d] for d in range(n_possible_fleets)
                        ) == 1
                    )

                for i in range(n_orders):
                    for d in range(n_possible_fleets):
                        m.addConstr(x[i, d] <= y[d])

                for d in range(n_possible_fleets - 1):
                    m.addConstr(y[d] >= y[d + 1])

                m.addConstr(
                    gp.quicksum(y[d] for d in range(n_possible_fleets))
                    >= lower_bound
                )

                # Supply the greedy solution as a feasible starting solution
                for d, bucket in enumerate(greedy_bins):
                    y[d].Start = 1
                    for i in bucket:
                        x[i, d].Start = 1

                m.setObjective(
                    gp.quicksum(y[d] for d in range(n_possible_fleets)),
                    gp.GRB.MINIMIZE
                )
                m.Params.BestObjStop = lower_bound
                m.optimize()

                if m.SolCount == 0:
                    raise RuntimeError("Gurobi did not find a feasible solution")

                min_num_fleet = round(m.ObjVal)

            if restaurant in min_fleet:
                min_fleet[restaurant] = max(
                    min_fleet[restaurant],
                    min_num_fleet
                )
            else:
                min_fleet[restaurant] = min_num_fleet

    return min_fleet

def analyse_minimum_fleets(fname):
           
    delivery_df = pd.read_csv(fname)

    # find the minimum number of drones and moped required
    min_drone_fleet = find_minimal_fleet(delivery_df, fleet_type="drone")
    min_moped_fleet = find_minimal_fleet(delivery_df, fleet_type="moped")

    for restaurant, min_fleets in min_drone_fleet.items():
        print(f"Restaurant: {restaurant}, Minimum Drones Required: {min_fleets}")
    for restaurant, min_fleets in min_moped_fleet.items():
        print(f"Restaurant: {restaurant}, Minimum Mopeds Required: {min_fleets}")

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