#BLOCK 1

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math

#BLOCK 2

waypoints = np.load('/home/arnavv/PPC/ppc_trainee_module/checkpoint4_the_final_implementation/waypoints.npy')
yellow_cones = np.load('/home/arnavv/PPC/ppc_trainee_module/checkpoint4_the_final_implementation/yellow_cones.npy')
blue_cones = np.load('/home/arnavv/PPC/ppc_trainee_module/checkpoint4_the_final_implementation/blue_cones.npy')
# This is the map which was obtained from optimising the waypoints

plt.plot(waypoints[:,0], waypoints[:,1], 'k--', label="Track")
plt.axis("equal")
plt.title("Waypoints")
plt.legend()
plt.grid()
plt.show()
sum_v = 0
prev_v = 0

#BLOCK 3

class Vehicle:
    def __init__(self, x=0, y=0, yaw=0 , v=0.0, L=2.5):  # Initial position, yaw, velocity, and wheelbase

        self.x = waypoints[0][0]             #  IMP ! : Change the initial pose of the car to determine the starting position
        self.y = waypoints[0][1]
        self.yaw = math.atan2(waypoints[1][1]-waypoints[0][1],waypoints[1][0]-waypoints[0][0])
        self.v = 1
        self.L = L  # Wheelbase

    def update(self, throttle, delta, dt=0.1):
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / self.L * np.tan(delta) * dt
        self.v += throttle * dt
        self.v = max(0.0, self.v)  # No reverse

#BLOCK 4

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def compute_control(vehicle, waypoints, last_target_idx, k=1.3, k_soft=1e-2):
    """

    Args:
        vehicle: Vehicle object
        waypoints: Nx2 array
        last_target_idx: previously selected waypoint index
        k: control gain for Stanley
        k_soft: softening term to avoid division by zero

    Returns:
        throttle: float
        steer: float
        target_idx: int
    """
    global sum_v, prev_v

    #front wheel positions
    fx = vehicle.x + vehicle.L * np.cos(vehicle.yaw)
    fy = vehicle.y + vehicle.L * np.sin(vehicle.yaw)

    #to detect next waypoint:
    dists = np.hypot(waypoints[:, 0] - fx, waypoints[:, 1] - fy)
    target_idx = np.argmin(dists)

    # Smooth heading vector by looking ahead
    '''if target_idx < len(waypoints) - 2:
        next_pt = waypoints[target_idx + 1]
    else:
        next_pt = waypoints[target_idx]'''
    
    last_vector = [(waypoints[last_target_idx][0]-fx),waypoints[last_target_idx][1]-fy]
    last_angle = math.atan2((waypoints[last_target_idx][1]-fy),waypoints[last_target_idx][0]-fx) - vehicle.yaw
    if abs(last_angle) < np.radians(45):
        target_idx = last_target_idx
    else:
        target_idx = (last_target_idx + 1)%len(waypoints)
    
    pt = waypoints[target_idx]
    #print(target_idx)
    theta = np.arctan2(pt[1]- waypoints[last_target_idx][1],pt[0]-waypoints[last_target_idx][0])
    
    #dist = math.sqrt((pt[0]-vehicle.x)**2 + (pt[1]-vehicle.y)**2)
    # Step 3: Compute heading of path at that point
    #theta = math.atan2(waypoints[target_idx][1]-waypoints[last_target_idx][1], waypoints[target_idx][0]-waypoints[last_target_idx][0])


    # Step 4: Compute heading error
    head_err = normalize_angle(theta - vehicle.yaw)



    # Step 5: Compute signed cross-track error
    e = (fx - waypoints[target_idx][0]) * math.sin(theta) - (fy - waypoints[target_idx][1]) * math.cos(theta)
    

    # Limit steering to realistic bounds
    max_steer = np.radians(30)
    steer_angle =   head_err +  1.5* math.atan2((k * e), (vehicle.v + k_soft))
    steer = np.clip(steer_angle, -max_steer, max_steer)
    print(steer)


    # Throttle control (simple proportional control or replace with PID)
    target_v = 5.0  # Target velocity You can change this velocity as per need

    # You must keep in mind that different target velocities will require retuning the steering gains

    
    sum_v = sum_v + (target_v - vehicle.v)
    

    throttle =  0.5 * (target_v - vehicle.v)  #+ sum_v * 0.05  + (prev_v - vehicle.v)/0.05

    prev_v = vehicle.v

    return throttle, steer, target_idx



#BLOCK 5

vehicle = Vehicle()
history = {'x': [], 'y': []}
target_idx = 0

fig, ax = plt.subplots(figsize=(6,6))
# ax.set_xlim(-12, 12)
# ax.set_ylim(-12, 12)
track_line, = ax.plot(waypoints[:,0], waypoints[:,1], 'k--')
blue_cone_dots = ax.scatter(blue_cones[:,0], blue_cones[:,1], c='blue', s=20, label="Blue Cones")
yellow_cone_dots = ax.scatter(yellow_cones[:,0], yellow_cones[:,1], c='yellow', s=20, label="Yellow Cones")
car_dot, = ax.plot([], [], 'bo', markersize=6)
path_line, = ax.plot([], [], 'b-', linewidth=1)

def init():
    car_dot.set_data([], [])
    path_line.set_data([], [])
    return car_dot, path_line, blue_cone_dots, yellow_cone_dots

def animate(i):
    global target_idx
    throttle, steer, target_idx = compute_control(vehicle, waypoints, target_idx)
    vehicle.update(throttle, steer)

    history['x'].append(vehicle.x)
    history['y'].append(vehicle.y)

    car_dot.set_data([vehicle.x], [vehicle.y])
    path_line.set_data(history['x'], history['y'])
    return car_dot, path_line, blue_cone_dots, yellow_cone_dots


ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=300, interval=50, blit=True)
#plt.close()  # Prevent double display in some notebooks

from IPython.display import HTML
#HTML(ani.to_jshtml())
plt.show()
