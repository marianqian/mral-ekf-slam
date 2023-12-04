from robot import Robot 
import numpy as np 
import plot 
import matplotlib.pyplot as plt



#initialize robot and parameters 
#alphas = np.array([0.11, 0.01, 0.18, 0.08, 0, 0]) #motion model noise parameters 
#Qt = np.array([[11.7, 0], [0, 0.18]])
Qt = np.array([[0.01,0],
               [0,0.01]])

Rt = 5*np.array([[0.1,0,0],
               [0,0.01,0],
               [0,0,0.01]])
dt = 1 
fov = 80 

#starting state of robot 
x_init = [0,0,0.5*np.pi]

r = Robot(init=x_init, Rt=Rt, Qt=Qt, dt=dt, fov=fov) 

#Generate landmarks 
n = 5 # number of STATIC landmarks
mapsize = 40
landmark_xy = mapsize*(np.random.rand(n,2)-0.5)
ls = landmark_xy


# In[Generate inputs and measurements]
#generate trajectory -- from EKF SLAM repo 
steps = 10
stepsize = 3
curviness = 0.5

#true state of robot (the robot recieves controls and makes noisy movement)
x_true = [x_init]
#list of noisy measurements PER robot movement 
obs = []

# generate input sequence -- sequence of movement vectors 
u = np.zeros((steps,2))
u[:,0] = stepsize
u[4:12,1] = curviness
u[18:26,1] = curviness


for movement, t in zip(u,range(steps)):
    # process robot movement
    x_true.append(r.move(movement))
    obs.append(r.sense(ls))

plot.plotMap(ls, x_true, r, mapsize)

x_pred = [x_init]

for i, (movement, measurement) in enumerate(zip(u, obs)): 
    r.predict(movement)
    x_pred.append(r.robot_state_mean)
    
    plot.plotEstimate(x_pred, r.robot_state_cov, r, mapsize) 
    r.update(measurement) 
    x_pred.append(r.robot_state_mean)
    plot.plotEstimate(x_pred, r.robot_state_cov, r, mapsize) 
    #plot.plotMeasurement(r.robot_state_mean[:3], r.robot_state_cov, measurement, n)

    print("step {} state{}".format(i, r.robot_state_mean)) 
    print("landmarks ", r.n_landmarks)
    
     
    #plot.plotError(np.array(x_pred), np.array(x_true)[:len(x_pred[0::2][:])]) 




plt.show() 




'''
move1 = np.array([1, 0])
meas1 = np.array([[3, 0], [1, np.pi/2]])
move2 = [1, 0]
meas2 = np.array([[3, 0]]) 


r.predict(move1) 
r.update(meas1) 
r.predict(move2) 
r.update(meas2) 
print(r.robot_state_mean) 
print(r.robot_state_cov)
'''


#slam 


#calculate error 