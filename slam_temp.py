#robot predicts (motion model) with some movement 
# robot corrects map using landmark observations from sensors 

# a map class 
# robot class that updates map class? robot contains trajectory and all movement steps 
# use robot class from EKF SLAM repo to generate noisy data 
# robot class in general that holds own state mean, cov and landmark mean and cov 

import numpy as np 

deltaT = 0.02 
alphas = np.array([0.11, 0.01, 0.18, 0.08, 0, 0]) #motion model noise parameters 

Qt = np.array([[11.7, 0], [0, 0.18]])

n_robots = 1 
robot_num = 1

#barcodes, landmark_groundtruth, robots 
#robots, timestpes 

#robot state  
robot_state_mean = np.zeros(3)
robot_state_cov = np.ones((3, 3)) * 0.001 

measurementIndex = 1
meas_count = 0 
n_landmarks = 0 

#codeDict -- we don't need between barcodes and landmarks? 

#TODO: add in movement changes to vectors  
u_t = [1, 0]
#u_t = [change in translation, change in rotation]
dt = 1 





theta = robot_state_mean[2] 
rot = u_t[1] * dt #dt = change in time; change in rotation
half_rot = rot / 2 
trans = u_t[0] * dt #change in translation

#calculate pose update from odometry 
pose_update = [trans * np.cos(theta+half_rot), trans * np.sin(theta+half_rot), rot]

#calculate updated state mean 
F_x = np.append(np.eye(3),np.zeros((3,robot_state_mean.shape[0]-3)),axis=1)
state_mean_bar = robot_state_mean + (F_x.T @ pose_update)
state_mean_bar[2] = (state_mean_bar[2] + 2 * np.pi) % (2 * np.pi)

#calculate movement Jacobian 
g_t = np.array([[0, 0, trans * -1 * np.sin(theta+half_rot)],
                [0, 0, trans * np.cos(theta+half_rot)], 
                [0, 0, 0]])

G_t = np.eye(robot_state_mean.shape[0]) + F_x.T @ g_t @ F_x 

#calculate motion covariance in control space 
M_t = np.array([[(alphas[0] * np.abs(u_t[0]) + alphas[1] * np.abs(u_t[1]))**2, 0],
                [0, (alphas[2] * np.abs(u_t[0]) + alphas[3] * np.abs(u_t[1]))**2]])

print(G_t)
print(F_x)
print(M_t)
#calculate Jacobian to transform motion covariance to state space 
V_t = np.array([[np.cos(theta + half_rot), -0.5 * np.sin(theta + half_rot)],
                [np.sin(theta + half_rot), 0.5 * np.cos(theta + half_rot)],
                [0, 1]])
print(V_t) 
#update state covariance 
R_t = V_t @ M_t @ V_t.T 
print(R_t)

state_cov_bar = (G_t @ robot_state_cov @ G_t.T) + (F_x.T @ R_t @ F_x)

#updated robot state -- state_mean_bar and state_cov_bar 
print(state_mean_bar)
print(state_cov_bar)





#correction step -- looping through measurements
#let z be list of measurements -- 
#TODO: add measurements 
#z = np.array([[translation from robot, rotation from robot], [x2, y2] ... ])
z = np.array([[3, 0], [1, np.pi/2]])

for k in range(z.shape[0]):
    predZ = np.zeros((2, n_landmarks+1)) 
    pred_psi = np.zeros((n_landmarks+1, 2, 2))
    predH = np.zeros((n_landmarks+1, 2, 2*(n_landmarks+1)+3))
    pi_k = np.zeros((n_landmarks+1, 1))

    #temporary new landmark at observed position 
    #temp_mark = np.array([state_mean_bar[0] + z[k][0] * np.cos(z[k][1] + state_mean_bar[2]),
    #                    state_mean_bar[1] + z[k][0] * np.sin(z[k][1] + state_mean_bar[2])])
    temp_mark = np.array([z[k][0],z[k][1]])
    print(temp_mark.shape)
    print(state_mean_bar.shape)
    state_mean_temp = np.append(state_mean_bar, temp_mark, axis=0)

    #TODO: double check axis 
    #state_cov_temp = np.array([[state_cov_bar, np.zeros((np.shape(state_cov_bar)[0], 2))],
    #                            [np.zeros((2, np.shape(state_cov_bar)[1] + 2))]])
    state_cov_temp = np.append(state_cov_bar, np.zeros((np.shape(state_cov_bar)[0], 2)), axis=1)
    state_cov_temp = np.append(state_cov_temp, np.zeros((2, np.shape(state_cov_bar)[1] + 2)), axis=0)
              

    #intialize covariance for new landmark proportional to range measurement squared 
    for i in range(state_cov_temp.shape[0]-2, state_cov_temp.shape[0]):
        state_cov_temp[i, i] = z[k][0]**2/130 

    #loop over all landmarks and compute likelihood of correspondence with new landmark 
    max_j = -1 
    min_pi = 10 * np.ones(2) 

    for i in range(1, n_landmarks+2): #change index (j - 1)
        j = i - 1 
        delta = np.array([state_mean_temp[2*j+3] - state_mean_temp[0],
                        state_mean_temp[2*j+4] - state_mean_temp[1]])
        q = delta.T @ delta 
        r = np.sqrt(q)

        print(r)

        predZ[:, j] = np.array([r, (((np.arctan2(delta[1], delta[0] - state_mean_temp[2])) + 2 * np.pi) % (2 * np.pi))])
        F_xj = np.zeros((5, 2 * (n_landmarks+1) + 3))
        F_xj[:3, :3] = np.eye(3) 
        F_xj[3:, 3+2*j:3+2*j+2] = np.eye(2)
        
        h_t = np.array([[-delta[0]/r, -delta[1]/r, 0, delta[0]/r, delta[1]/r],
                        [delta[1]/q, -delta[0]/q, -1, -delta[1]/q, delta[0]/q]])

        predH[j, :, :] = h_t @ F_xj 
        pred_psi[j, :, :] = np.squeeze(predH[j,:,:]) @ state_cov_temp @ np.squeeze(predH[j,:,:]).T + Qt
        
        if j < n_landmarks: 
            pi_k[j] = (z[k]-predZ[:, j]).T @ np.linalg.inv(np.squeeze(pred_psi[j, :, :])) @ (z[k]-predZ[:, j])
        else:
            pi_k[j] = 0.84 #min mahalanobis distance to add landmark to map 
        if pi_k[j] < min_pi[0]: 
            min_pi[1] = min_pi[0]
            max_j = j 
            min_pi[0] = pi_k[j]


    H = np.squeeze(predH[max_j, :, :])
    
    #best association must be significantly better than the second best, otws thrown out 
    if min_pi[1] / min_pi[0] > 1.6: 
        meas_count += 1 

        #if landmark added, expand state and covariance matrices 
        if max_j >= n_landmarks: 
            state_mean_bar = state_mean_temp
            state_cov_bar = state_cov_temp
            n_landmarks += 1 
        
        else: 
            #if measurement associated to exisiting landmark - truncate H 
            H = H[:, :n_landmarks*2+3]
            K = state_cov_bar @ H.T @ np.linalg.inv(np.squeeze(pred_psi[max_j, :, :]))
            state_mean_bar = state_mean_bar + K @ (z[k]-predZ[:, max_j])
            state_mean_bar[2] = (state_mean_bar[2] + 2 * np.pi) % (2 * np.pi)
            state_cov_bar = (np.eye(state_cov_bar.shape[0]) - (K @ H)) @ state_cov_bar


robot_state_cov = state_cov_bar
robot_state_mean = state_mean_bar 
print(robot_state_mean)
print(robot_state_cov)
print(n_landmarks)




###################



#TODO: add in movement changes to vectors  
u_t = [1, 0]
#u_t = [change in translation, change in rotation]
dt = 1 


theta = robot_state_mean[2] 
rot = u_t[1] * dt #dt = change in time; change in rotation
half_rot = rot / 2 
trans = u_t[0] * dt #change in translation

#calculate pose update from odometry 
pose_update = [trans * np.cos(theta+half_rot), trans * np.sin(theta+half_rot), rot]

#calculate updated state mean 
F_x = np.append(np.eye(3),np.zeros((3,robot_state_mean.shape[0]-3)),axis=1)
state_mean_bar = robot_state_mean + (F_x.T @ pose_update)
state_mean_bar[2] = (state_mean_bar[2] + 2 * np.pi) % (2 * np.pi)

#calculate movement Jacobian 
g_t = np.array([[0, 0, trans * -1 * np.sin(theta+half_rot)],
                [0, 0, trans * np.cos(theta+half_rot)], 
                [0, 0, 0]])

G_t = np.eye(robot_state_mean.shape[0]) + F_x.T @ g_t @ F_x 

#calculate motion covariance in control space 
M_t = np.array([[(alphas[0] * np.abs(u_t[0]) + alphas[1] * np.abs(u_t[1]))**2, 0],
                [0, (alphas[2] * np.abs(u_t[0]) + alphas[3] * np.abs(u_t[1]))**2]])

print(G_t)
print(F_x)
print(M_t)
#calculate Jacobian to transform motion covariance to state space 
V_t = np.array([[np.cos(theta + half_rot), -0.5 * np.sin(theta + half_rot)],
                [np.sin(theta + half_rot), 0.5 * np.cos(theta + half_rot)],
                [0, 1]])
print(V_t) 
#update state covariance 
R_t = V_t @ M_t @ V_t.T 
print(R_t)

state_cov_bar = (G_t @ robot_state_cov @ G_t.T) + (F_x.T @ R_t @ F_x)

#updated robot state -- state_mean_bar and state_cov_bar 
print(state_mean_bar)
print(state_cov_bar)





#correction step -- looping through measurements
#let z be list of measurements -- 
#TODO: add measurements 
#z = np.array([[translation from robot, rotation from robot], [x2, y2] ... ])
z = np.array([[3, 0]])

for k in range(z.shape[0]):
    predZ = np.zeros((2, n_landmarks+1)) 
    pred_psi = np.zeros((n_landmarks+1, 2, 2))
    predH = np.zeros((n_landmarks+1, 2, 2*(n_landmarks+1)+3))
    pi_k = np.zeros((n_landmarks+1, 1))

    #temporary new landmark at observed position 
    #temp_mark = np.array([state_mean_bar[0] + z[k][0] * np.cos(z[k][1] + state_mean_bar[2]),
    #                    state_mean_bar[1] + z[k][0] * np.sin(z[k][1] + state_mean_bar[2])])
    temp_mark = np.array([z[k][0],z[k][1]])
    print(temp_mark.shape)
    print(state_mean_bar.shape)
    state_mean_temp = np.append(state_mean_bar, temp_mark, axis=0)

    #TODO: double check axis 
    #state_cov_temp = np.array([[state_cov_bar, np.zeros((np.shape(state_cov_bar)[0], 2))],
    #                            [np.zeros((2, np.shape(state_cov_bar)[1] + 2))]])
    state_cov_temp = np.append(state_cov_bar, np.zeros((np.shape(state_cov_bar)[0], 2)), axis=1)
    state_cov_temp = np.append(state_cov_temp, np.zeros((2, np.shape(state_cov_bar)[1] + 2)), axis=0)
              

    #intialize covariance for new landmark proportional to range measurement squared 
    for i in range(state_cov_temp.shape[0]-2, state_cov_temp.shape[0]):
        state_cov_temp[i, i] = z[k][0]**2/130 

    #loop over all landmarks and compute likelihood of correspondence with new landmark 
    max_j = -1 
    min_pi = 10 * np.ones(2) 

    for i in range(1, n_landmarks+2): #change index (j - 1)
        j = i - 1 
        delta = np.array([state_mean_temp[2*j+3] - state_mean_temp[0],
                        state_mean_temp[2*j+4] - state_mean_temp[1]])
        q = delta.T @ delta 
        r = np.sqrt(q)

        print(r)

        predZ[:, j] = np.array([r, (((np.arctan2(delta[1], delta[0] - state_mean_temp[2])) + 2 * np.pi) % (2 * np.pi))])
        F_xj = np.zeros((5, 2 * (n_landmarks+1) + 3))
        F_xj[:3, :3] = np.eye(3) 
        F_xj[3:, 3+2*j:3+2*j+2] = np.eye(2)
        
        h_t = np.array([[-delta[0]/r, -delta[1]/r, 0, delta[0]/r, delta[1]/r],
                        [delta[1]/q, -delta[0]/q, -1, -delta[1]/q, delta[0]/q]])

        predH[j, :, :] = h_t @ F_xj 
        pred_psi[j, :, :] = np.squeeze(predH[j,:,:]) @ state_cov_temp @ np.squeeze(predH[j,:,:]).T + Qt
        
        if j < n_landmarks: 
            pi_k[j] = (z[k]-predZ[:, j]).T @ np.linalg.inv(np.squeeze(pred_psi[j, :, :])) @ (z[k]-predZ[:, j])
        else:
            pi_k[j] = 0.84 #min mahalanobis distance to add landmark to map 
        if pi_k[j] < min_pi[0]: 
            min_pi[1] = min_pi[0]
            max_j = j 
            min_pi[0] = pi_k[j]


    H = np.squeeze(predH[max_j, :, :])
    
    #best association must be significantly better than the second best, otws thrown out 
    if min_pi[1] / min_pi[0] > 1.6: 
        meas_count += 1 

        #if landmark added, expand state and covariance matrices 
        if max_j >= n_landmarks: 
            state_mean_bar = state_mean_temp
            state_cov_bar = state_cov_temp
            n_landmarks += 1 
        
        else: 
            #if measurement associated to exisiting landmark - truncate H 
            H = H[:, :n_landmarks*2+3]
            K = state_cov_bar @ H.T @ np.linalg.inv(np.squeeze(pred_psi[max_j, :, :]))
            state_mean_bar = state_mean_bar + K @ (z[k]-predZ[:, max_j])
            state_mean_bar[2] = (state_mean_bar[2] + 2 * np.pi) % (2 * np.pi)
            state_cov_bar = (np.eye(state_cov_bar.shape[0]) - (K @ H)) @ state_cov_bar


robot_state_cov = state_cov_bar
robot_state_mean = state_mean_bar 
print(robot_state_mean)
print(robot_state_cov)
print(n_landmarks)