import pandas as pd
import numpy as np
import requests

import random
from geopy.distance import geodesic

from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union 

import constants 


def generate_random_points_with_exclusions(main_coords, num_points):
    """
    Generates random (latitude, longitude) coordinates within a main polygon,
    excluding specified sub-areas.

    :param main_coords: List of (lon, lat) tuples for the main area.
    :param exclusion_zones: List of lists of (lon, lat) tuples to exclude.
    :param num_points: Integer, number of valid points to generate.
    :return: List of (latitude, longitude) tuples.
    """
    # 1. Create the main polygon
    valid_area = Polygon(main_coords)

    # 2. Subtract each exclusion zone from the main polygon
    #for zone in exclusion_zones:
    #    exclusion_poly = Polygon(zone)
    #    valid_area = valid_area.difference(exclusion_poly)

    # 3. Get the bounding box of the new, modified area
    min_x, min_y, max_x, max_y = valid_area.bounds

    valid_points = []

    while len(valid_points) < num_points:
        # Generate point within the bounding box
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))

        # Check if the modified polygon contains the point
        if valid_area.contains(random_point):
            # Append as (Latitude, Longitude)
            valid_points.append((random_point.y, random_point.x))

    return valid_points

def generate_random_points_in_shape(main_shape, exclusion_zones, num_points):
    """
    Generates random (latitude, longitude) coordinates within a Shapely geometry,
    excluding specified sub-areas.

    :param main_shape: A Shapely Polygon or MultiPolygon object.
    :param exclusion_zones: List of lists of (lon, lat) tuples to exclude.
    :param num_points: Integer, number of valid points to generate.
    :return: List of (latitude, longitude) tuples.
    """
    # 1. Set the valid area directly to the passed Shapely shape
    valid_area = main_shape

    # 2. Subtract each exclusion zone
    for zone in exclusion_zones:
        exclusion_poly = Polygon(zone)
        valid_area = valid_area.difference(exclusion_poly)

    # 3. Get the bounding box of the entire combined area
    min_x, min_y, max_x, max_y = valid_area.bounds

    valid_points = []

    while len(valid_points) < num_points:
        # Generate point within the master bounding box
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))

        # Check if the combined (and subtracted) geometry contains the point
        if valid_area.contains(random_point):
            valid_points.append((random_point.y, random_point.x))

    return valid_points

def generate_random_points_stockholm_innercity(N, area="all"):
    # 1. Convert district coordinates into Shapely Polygons
    if area == "sodermalm":
        districts = [Polygon(constants.SODERMALM_POLYGON)]
    elif area == "all":
        districts = [
            Polygon(constants.SODERMALM_POLYGON),
            Polygon(constants.OSTERMALM_POLYGON),
            Polygon(constants.NORRMALM_POLYGON),
            Polygon(constants.KUNGSHOLMEN_POLYGON),
            Polygon(constants.GAMLA_STAN_POLYGON)
        ]
    else:
        raise ValueError("Invalid area specified. Choose 'sodermalm' or 'all'.")
    
    # 2. Merge them into a single "Inner City" geometry using unary_union
    inner_city_shape = unary_union(districts)

    # 3. Generate N random points across all of inner-city Stockholm, excluding no-fly zones
    return generate_random_points_in_shape(inner_city_shape, constants.EXCLUSION_NO_FLY_ZONE, N)

def get_road_distance(restaurants, delivery_locations, restaurants_dict):

    key_path = Path(__file__).resolve().parent.parent / "key" / "ors_api_key.txt"
    with open(key_path, "r") as f:
        API_KEY = f.read().strip()

    url = "https://api.heigit.org/openrouteservice/v2/matrix/cycling-regular"

    headers = {
        "Authorization": API_KEY,
        "Content-Type": "application/json"
    }

    distances = []

    # Group deliveries by their chosen restaurant
    restaurant_groups = {}

    for i, restaurant in enumerate(restaurants):
        restaurant_groups.setdefault(restaurant, []).append(i)

    for restaurant, indices in restaurant_groups.items():

        origin_lat, origin_lon = restaurants_dict[restaurant]

        # OBS: coordinates are expected in lon, lat
        coordinates = [[origin_lon, origin_lat]]
        coordinates += [[delivery_locations[i][1], delivery_locations[i][0]] for i in indices]

        body = {
            "locations": coordinates,
            "sources": [0],
            "destinations": list(range(1, len(coordinates))),
            "metrics": ["distance"]
        }

        response = requests.post(url, headers = headers, json = body)
        response.raise_for_status()

        result = response.json()

        # Put each distance back at its original delivery index
        for i, distance in zip(indices, result["distances"][0]):
            if distance is None:
                raise ValueError(f"Distance for delivery index {i} is None. Check the API response.")
            distances.append(distance / 1000)

    return distances

def get_geodesic_distance(origin_lat, origin_lon, delivery_locations):
    distances = [
        geodesic((origin_lat, origin_lon), (lat, lon)).kilometers
        for lat, lon in delivery_locations
    ]
    return distances

def get_moped_delivery_time(moped_delivery_distances):
    return [d / constants.SPEED_MOPED * 60 for d in moped_delivery_distances]

def get_drone_delivery_time(drone_delivery_distances):
    return [d / constants.SPEED_DRONE * 60 for d in drone_delivery_distances]

def get_nearest_restaurant_dist(order_coord, restaurants):

    distances = {
        name: geodesic((resto_lat, resto_lon), order_coord).kilometers
        for name, (resto_lat, resto_lon) in restaurants.items()
    }

    nearest = min(distances, key=distances.get)
    return nearest, distances[nearest]

def get_moped_unavailability_time(moped_travel_distances):
    return [d / constants.SPEED_MOPED * 60 for d in moped_travel_distances]

def calculate_drone_unavailability_time(drone_flight_distance):
    flight_time = drone_flight_distance / constants.SPEED_DRONE * 60  # in minutes
    charge_time = (drone_flight_distance / constants.FULL_CHARGE_DIST) * constants.FULL_CHARGE_TIME  # in minutes
    # Additional time for loading, takeoff, and landing
    additional_time = 1  # in minutes
    total_unavailability_time = flight_time + charge_time + additional_time
    return total_unavailability_time

def get_drone_unavailability_time(drone_flight_distances):
    return [calculate_drone_unavailability_time(d) for d in drone_flight_distances]

def check_nofly_status(resto_lat, resto_lon, order_lat, order_lon, exclusion_zones):
    restaurant_point = Point(resto_lon, resto_lat)
    order_point = Point(order_lon, order_lat)
    flight_path = LineString([(resto_lon, resto_lat), (order_lon, order_lat)])
    exclusion_polygons = [Polygon(zone) for zone in exclusion_zones]

    for poly in exclusion_polygons:
        if poly.contains(restaurant_point) or poly.touches(restaurant_point):
            return constants.STATUS_NOGO
        if poly.contains(order_point) or poly.touches(order_point):
            return constants.STATUS_NOGO
        if flight_path.intersects(poly):
            return constants.STATUS_INTERSECT
    return constants.STATUS_CLEAR

def generate_delivery_data(N, restaurants_dict, area, fname):
    # input 
    #  N: number of delivery locations to generate
    #  restaurants_dict: dictionary of restaurant names and their coordinates
    #  area: area name (sodermalm | all)
    #  fname: output file name

    # step 1: generate N locations random as delivery locations
    delivery_locations = generate_random_points_stockholm_innercity(N, area)
    nearest = [
        get_nearest_restaurant_dist((lat, lon), restaurants_dict)
        for lat, lon in delivery_locations
    ]

    # step 2: get the nearest restaurant and geodesic distance for each delivery location
    restaurants, geodesic_dists = map(list, zip(*nearest))

    # step 3: get the moped and drone delivery distances
    moped_delivery_distances = get_road_distance(restaurants, delivery_locations, restaurants_dict  )

    no_fly_status = []
    for resto, (order_lat, order_lon) in zip(restaurants, delivery_locations):
        resto_lat, resto_lon = restaurants_dict[resto]
        exclusion_polygons = constants.EXCLUSION_NO_FLY_ZONE
        no_fly_status.append(check_nofly_status(resto_lat, resto_lon, order_lat, order_lon, exclusion_polygons))

    drone_delivery_distances = geodesic_dists.copy()
    penalized_drone_distances = geodesic_dists.copy()
    for i in range(N):
        if no_fly_status[i] == constants.STATUS_NOGO:
            drone_delivery_distances[i] = np.nan
            penalized_drone_distances[i] = geodesic_dists[i] + constants.NO_FLY_PENALTY_DISTANCE_KM
        elif no_fly_status[i] == constants.STATUS_INTERSECT:
            drone_delivery_distances[i] += 1
            penalized_drone_distances[i] = geodesic_dists[i] + constants.NO_FLY_PENALTY_DISTANCE_KM
        else:
            penalized_drone_distances[i] = geodesic_dists[i]

    moped_travel_distances = [d*2 for d in moped_delivery_distances]
    drone_flight_distances = [d*2 for d in drone_delivery_distances]
    penalized_drone_flight_distances = [d*2 for d in penalized_drone_distances]

    # Step 4: Generate moped and drone delivery times
    moped_delivery_time = get_moped_delivery_time(moped_delivery_distances)
    drone_delivery_time = get_drone_delivery_time(drone_delivery_distances)
    penalized_drone_delivery_time = get_drone_delivery_time(penalized_drone_distances)

    # Step 5: Generate unavailability time
    moped_unavailability_time = get_moped_unavailability_time(moped_travel_distances)
    drone_unavailability_time = get_drone_unavailability_time(drone_flight_distances)
    penalized_drone_unavailability_time = get_drone_unavailability_time(penalized_drone_flight_distances)

    # Step 6: Save the delivery locations to a CSV file
    delivery_df = pd.DataFrame({
        constants.LATITUDE: [lat for lat, _ in delivery_locations],
        constants.LONGITUDE: [lon for _, lon in delivery_locations],
        constants.GEODESIC_DIST: geodesic_dists,
        constants.RESTAURANT_NAME: restaurants,
        constants.MOPED_DELIVERY_DISTANCE: moped_delivery_distances,
        constants.DRONE_DELIVERY_DISTANCE: drone_delivery_distances,
        constants.MOPED_TRAVEL_DISTANCE: moped_travel_distances,
        constants.DRONE_TRAVEL_DISTANCE: drone_flight_distances,
        constants.MOPED_DELIVERY_TIME: moped_delivery_time,
        constants.DRONE_DELIVERY_TIME: drone_delivery_time,
        constants.MOPED_UNAVAILABILITY_TIME: moped_unavailability_time,
        constants.DRONE_UNAVAILABILITY_TIME: drone_unavailability_time,
        constants.PENALIZED_DISTANCE: penalized_drone_distances,
        constants.PENALIZED_DRONE_UNAVAILABILITY_TIME: penalized_drone_unavailability_time,
        constants.NO_FLY_STATUS: no_fly_status
    })

    delivery_df.to_csv(fname, index=False)

# usage: python3 generate_data.py -N {number_of_points} -A {sodermalm | all} -O {output_file}
# get flags

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, required=True,
                    help="Number of delivery locations to generate")
parser.add_argument("-a", choices=["sodermalm", "södermalm", "all"], required=False, default="all",
                    help="Area to consider")
parser.add_argument(
    "-f",
    default="delivery_locations.csv",
    help="Output filename only"
)
args = parser.parse_args()

if Path(args.f).name != args.f:
    parser.error("-f must contain a filename only, not a directory path")

if args.a == "södermalm":
    args.a = "sodermalm"
if args.a == "sodermalm":
    restaurants_dict = {
        k: v for k, v in constants.RESTAURANTS_DICT.items()
        if "södermalm" in k.lower()
    }
elif args.a == "all":
    restaurants_dict = constants.RESTAURANTS_DICT
else:
    raise ValueError("Invalid area specified. Choose 'sodermalm' or 'all'.")

output_file = Path(__file__).resolve().parent.parent / "data" / args.f

generate_delivery_data(args.n, restaurants_dict, args.a, output_file)

