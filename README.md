# SF2866 Project 1
This is an attempt to use drones for food delivery in the Stockholm Area.

## OpenRoute Service
We use openroute service to calculate the distance of roads to delivery locations. Generate your own API key and put it to `key/ors_api_key.txt` to test it out yourself :)

## Gurobi
Set up your license with [Gurobi](https://www.gurobi.com) to be able to test out our optimization scripts.

## Code
All codes needs to be run from the `drone_delivery/src` directory.

### Generate Data

To generate the data for the single restaurant case:
```
    python3 generate_data.py -n 400 -a södermalm -f deliveries_sodermalm.csv
```

To generate the data for the single restaurant case:
```
    python3 generate_data.py -n 2400 -a all -f delivery_locations.csv
```

### Data
To get the data from Table 1: Single Restaurant MILP results, run
```
    python3 optimize_fleets.py -f deliveries_sodermalm.csv --ignore-no-fly-zone
```
To get the data from Table 2: Multiple Restaurant MILP results without no-fly-zones, run
```
    python3 optimize_fleets.py -f delivery_locations.csv --ignore-no-fly-zone
```
To get the data from Table 3: 
```
    python3 optimize_mixed_fleet.py -f deliveries_sodermalm.csv --ignore-no-fly-zone
```
To get the data from Table 4: 
```
    python3 optimize_mixed_fleet.py -f delivery_locations.csv --ignore-no-fly-zone
```
To get the data from Table 6: 
```
   python3 optimize_mixed_fleet.py -f delivery_locations.csv
```


