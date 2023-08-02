import numpy as np
import matplotlib.pyplot as plt
from tsmoothie.smoother import *
from tsmoothie.utils_func import sim_randomwalk

# generate 3 randomwalks timeseries of lenght 100
np.random.seed(123)
data = sim_randomwalk(n_series=3, timesteps=100, 
                      process_noise=10, measure_noise=30)

# operate smoothing
smoother = KalmanSmoother(component='level_trend', 
                          component_noise={'level':0.1, 'trend':0.1})
smoother.smooth(data)

# generate intervals
low, up = smoother.get_intervals('kalman_interval', confidence=0.05)

# plot the first smoothed timeseries with intervals
plt.figure(figsize=(11,6))
plt.plot(smoother.smooth_data[0], linewidth=3, color='blue')
plt.plot(smoother.data[0], '.k')
plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3)