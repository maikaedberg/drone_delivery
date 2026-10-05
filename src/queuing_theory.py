import pandas as pd

import math

def erlang_c(s, a):
    if s <= a:
        return 1.0
    tail = (a ** s / math.factorial(s)) * (s / (s - a))
    normalizer = sum(a ** k / math.factorial(k) for k in range(s)) + tail
    return tail / normalizer

def asa(num_drones, orders_per_hour, unavailability_minutes):
    mu = 60 / unavailability_minutes
    a = orders_per_hour / mu

    if num_drones <= a:
        return float("inf")  # unstable system

    return 60 * erlang_c(num_drones, a) / (
        num_drones * mu - orders_per_hour
    )

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

#drones tsf plot söder

import numpy as np
import matplotlib.pyplot as plt

T = np.linspace(0, 40, 300)

df = pd.read_csv("../data/deliveries_sodermalm.csv")
drone_service_time = df['drone_unavailability_time'].mean()

plt.plot(T, [tsf(17, 100, drone_service_time, t) for t in T], label="17 drones")
plt.plot(T, [tsf(16, 100, drone_service_time, t) for t in T], label="16 drones")
plt.plot(T, [tsf(15, 100, drone_service_time, t) for t in T], label="15 drones")
plt.plot(T, [tsf(14, 100, drone_service_time, t) for t in T], label="14 drones")

plt.xlabel("Waiting-time limit T (minutes)",fontsize=14)
plt.ylabel("TSF",fontsize=14)
plt.title("TSF Drones Södermalm",fontsize=16)
plt.tick_params(axis="both", labelsize=12)
plt.ylim(0, 1)
plt.xlim(0, 20)
plt.grid(alpha=0.3)
plt.legend()
plt.show()

#mopeds tsf plot söder

import numpy as np
import matplotlib.pyplot as plt

T = np.linspace(0, 20, 300)
moped_service_time = df['moped_unavailability_time'].mean()

plt.plot(T, [tsf(28, 100, moped_service_time, t) for t in T], label="28 mopeds")
plt.plot(T, [tsf(27, 100, moped_service_time, t) for t in T], label="27 mopeds")
plt.plot(T, [tsf(26, 100, moped_service_time, t) for t in T], label="26 mopeds")
plt.plot(T, [tsf(25, 100, moped_service_time, t) for t in T], label="25 mopeds")

plt.xlabel("Waiting-time limit T (minutes)",fontsize=14)
plt.ylabel("TSF", fontsize=14)
plt.title("TSF mopeds Södermalm", fontsize=16)
plt.ylim(0, 1)
plt.xlim(0, 20)
plt.grid(alpha=0.3)
plt.tick_params(axis="both", labelsize=12)
plt.legend()
plt.show()