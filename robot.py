import numpy as np 


class Robot: 
    def __init__(self, init, Qt, Rt, dt, fov):
        self.robot_state_mean = np.zeros(3) 
        self.Qt = Qt #measurement model noise parameters 
        #self.alphas = alphas #motion model noise parameters -- impacts covariance matrix of robot state  
        self.Rt = Rt 
        self.n_landmarks = 0 

        self.robot_state_mean = np.zeros(3) 
        self.robot_state_cov = np.ones((3, 3)) * 0.001 

        self.time = 0 
        self.dt = dt 
        self.x_true = init 

        self.fov = fov #field of view

    def move(self, u): 
        #makes noisy movement -- from EKF SLAM repo 
        motion_noise = np.matmul(np.random.randn(1,3),self.Rt)[0]
        [dtrans, rot] = u[:2] + motion_noise[:2]
        drot1 = rot / 2 
        x = self.x_true
        x_new = x[0] + dtrans*np.cos(x[2]+drot1)
        y_new = x[1] + dtrans*np.sin(x[2]+drot1)
        theta_new = (x[2] + rot + np.pi) % (2*np.pi) - np.pi
        
        self.x_true = [x_new, y_new, theta_new]
        
        return self.x_true #true state of robot 



    def sense(self,lt):
        # Make noisy observation of subset of landmarks in field of view
        
        x = self.x_true
        observation = np.empty((0,2))
        
        fovL = (x[2]+self.fov/2+2*np.pi)%(2*np.pi)
        fovR = (x[2]-self.fov/2+2*np.pi)%(2*np.pi)
        
        for landmark in lt:
            rel_angle = np.arctan2((landmark[1]-x[1]),(landmark[0]-x[0]))
            rel_angle_2pi = (np.arctan2((landmark[1]-x[1]),(landmark[0]-x[0]))+2*np.pi)%(2*np.pi)
            # TODO: re-include and debug field of view constraints
            if (fovL - rel_angle_2pi + np.pi) % (2*np.pi) - np.pi > 0 and (fovR - rel_angle_2pi + np.pi) % (2*np.pi) - np.pi < 0:
                meas_range = np.sqrt(np.power(landmark[1]-x[1],2)+np.power(landmark[0]-x[0],2)) + self.Qt[0][0]*np.random.randn(1)
                meas_bearing = (rel_angle - x[2] + self.Qt[1][1]*np.random.randn(1) + np.pi)%(2*np.pi)-np.pi
                observation = np.append(observation,[[meas_range[0], meas_bearing[0]]],axis=0)

                #meas_range = np.sqrt(np.power(landmark[1]-x[1],2)+np.power(landmark[0]-x[0],2)) 
                #meas_bearing = (rel_angle - x[2] + np.pi)%(2*np.pi)-np.pi
                #observation = np.append(observation,[[meas_range, meas_bearing]],axis=0)

        return observation


    def predict(self, u_t): 
        #u_t: movement vector (2,1) [change in translation, change in rotation (radians)]
                
        theta = self.robot_state_mean[2] 
        rot = u_t[1] * self.dt #dt = change in time; change in rotation
        half_rot = rot / 2 
        trans = u_t[0] * self.dt #change in translation
                
        #calculate pose update from odometry 
        pose_update = [trans * np.cos(theta+half_rot), trans * np.sin(theta+half_rot), rot]

        #calculate updated state mean 
        F_x = np.append(np.eye(3),np.zeros((3,self.robot_state_mean.shape[0]-3)),axis=1)
        state_mean_bar = self.robot_state_mean + (F_x.T @ pose_update)
        state_mean_bar[2] = (state_mean_bar[2] + 2 * np.pi) % (2 * np.pi)

        #calculate movement Jacobian 
        g_t = np.array([[0, 0, trans * -1 * np.sin(theta+half_rot)],
                        [0, 0, trans * np.cos(theta+half_rot)], 
                        [0, 0, 0]])

        G_t = np.eye(self.robot_state_mean.shape[0]) + F_x.T @ g_t @ F_x 

        #to account for additional noise or uncertainty
        #alphas dependent on robot, so using predefined Rt matrix from EKF SLAM robot.py 
        '''#calculate motion covariance in control space 
        
        M_t = np.array([[(self.alphas[0] * np.abs(u_t[0]) + self.alphas[1] * np.abs(u_t[1]))**2, 0],
                        [0, (self.alphas[2] * np.abs(u_t[0]) + self.alphas[3] * np.abs(u_t[1]))**2]])

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
        '''
        print(self.Rt)

        state_cov_bar = (G_t @ self.robot_state_cov @ G_t.T) + (F_x.T @ self.Rt @ F_x)

        #updated robot state -- state_mean_bar and state_cov_bar 
        print(state_mean_bar)
        print(state_cov_bar)
        self.robot_state_cov = state_cov_bar 
        self.robot_state_mean = state_mean_bar



    def update(self, z): 
        #z: list of observations robot perceives after moving (N, 2) where N is number of observations 
        # [x of landmark, y of landmark] -- can change to be translation/rotation from robot
        
        #state_mean_bar = self.robot_state_mean
        state_cov_bar = self.robot_state_cov 

        for k in range(z.shape[0]):
            predZ = np.zeros((2, self.n_landmarks+1)) 
            pred_psi = np.zeros((self.n_landmarks+1, 2, 2))
            predH = np.zeros((self.n_landmarks+1, 2, 2*(self.n_landmarks+1)+3))
            pi_k = np.zeros((self.n_landmarks+1, 1))

            #temporary new landmark at observed position 

            #translation/rotation from robot 
            temp_mark = np.array([self.robot_state_mean[0] + z[k][0] * np.cos(z[k][1] + self.robot_state_mean[2]),
                                self.robot_state_mean[1] + z[k][0] * np.sin(z[k][1] + self.robot_state_mean[2])])
            
            #x, y of robot 
            #temp_mark = np.array([z[k][0],z[k][1]])

            print(temp_mark.shape)
            print(self.robot_state_mean.shape)
            state_mean_temp = np.append(self.robot_state_mean, temp_mark, axis=0)

            #TODO: double check axis 
            #state_cov_temp = np.array([[state_cov_bar, np.zeros((np.shape(state_cov_bar)[0], 2))],
            #                            [np.zeros((2, np.shape(state_cov_bar)[1] + 2))]])
            state_cov_temp = np.append(self.robot_state_cov, np.zeros((np.shape(self.robot_state_cov)[0], 2)), axis=1)
            state_cov_temp = np.append(state_cov_temp, np.zeros((2, np.shape(self.robot_state_cov)[1]+2)), axis=0)
                    

            #intialize covariance for new landmark proportional to range measurement squared 
            for i in range(state_cov_temp.shape[0]-2, state_cov_temp.shape[0]):
                state_cov_temp[i, i] = z[k][0]**2/130 

            #loop over all landmarks and compute likelihood of correspondence with new landmark 
            max_j = -1 
            min_pi = 10 * np.ones(2) 

            for i in range(1, self.n_landmarks+2): #change index (j - 1)
                j = i - 1 
                delta = np.array([state_mean_temp[2*j+3] - state_mean_temp[0],
                                state_mean_temp[2*j+4] - state_mean_temp[1]])
                q = delta.T @ delta 
                r = np.sqrt(q)

                print(r)

                predZ[:, j] = np.array([r, (((np.arctan2(delta[1], delta[0] - state_mean_temp[2])) + 2 * np.pi) % (2 * np.pi))])
                F_xj = np.zeros((5, 2*(self.n_landmarks+1)+3))
                F_xj[:3, :3] = np.eye(3) 
                F_xj[3:, 3+2*j:3+2*j+2] = np.eye(2)
                
                h_t = np.array([[-delta[0]/r, -delta[1]/r, 0, delta[0]/r, delta[1]/r],
                                [delta[1]/q, -delta[0]/q, -1, -delta[1]/q, delta[0]/q]])

                predH[j, :, :] = h_t @ F_xj 
                pred_psi[j, :, :] = np.squeeze(predH[j,:,:]) @ state_cov_temp @ np.squeeze(predH[j,:,:]).T + self.Qt
                
                if j < self.n_landmarks: 
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
                #if landmark added, expand state and covariance matrices 
                if max_j >= self.n_landmarks: 
                    self.robot_state_mean = state_mean_temp
                    self.robot_state_cov = state_cov_temp
                    self.n_landmarks += 1 
                
                else: 
                    #if measurement associated to exisiting landmark - truncate H 
                    H = H[:, :self.n_landmarks*2+3]
                    K = self.robot_state_cov @ H.T @ np.linalg.inv(np.squeeze(pred_psi[max_j, :, :]))
                    self.robot_state_mean = self.robot_state_mean + K @ (z[k]-predZ[:, max_j])
                    self.robot_state_mean[2] = (self.robot_state_mean[2] + 2 * np.pi) % (2 * np.pi)
                    self.robot_state_cov = (np.eye(self.robot_state_cov.shape[0]) - (K @ H)) @ self.robot_state_cov


        #self.robot_state_cov = state_cov_bar
        #self.robot_state_mean = state_mean_bar 
        print(self.robot_state_mean)
        print(self.robot_state_cov)
        print(self.n_landmarks)


#generate aspects of map (i.e. landmarks)
    