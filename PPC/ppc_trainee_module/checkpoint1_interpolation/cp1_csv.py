import pandas as pd
import scipy
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/home/arnavv/python_prac/PPC/loop_track_waypoints.csv')
x = df['X'].values
y = df['Y'].values

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'o', label='Waypoints')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Path Plot from CSV')
plt.axis('equal')  # Ensures correct aspect ratio
plt.grid(True)
plt.legend()
plt.show()

# Interpolation
# For direct use of scipy.interpolate.interp1d, the points must be unique!
# So interpolating with respect to the index of the dataframe = t

t = np.arange(len(x))

# Like the interpolating function
fx = interp1d(t, x, kind='cubic')
fy = interp1d(t, y, kind='cubic')

# Now we define the interpolated set of x and y values

t_interp = np.linspace(0, len(x) - 1, num=1000)  # 1000 points between 0 and len(x)-1
x_interp = fx(t_interp)
y_interp = fy(t_interp)
plt.figure(figsize=(8, 6))  
plt.plot(x, y, 'o', label='Waypoints')
plt.plot(x_interp, y_interp, '-', label='Interpolated Path')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Interpolated Path from Waypoints')
plt.axis('equal')  # Ensures correct aspect ratio
plt.grid(True)
plt.legend()
plt.show()