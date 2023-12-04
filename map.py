import numpy as np

"""
Contains Map class for the map of landmarks and current robot position 

Accounts for data association when seeing a new observation 
and uknown number of landmarks at beginning 

Translated from MATLAB: http://andrewjkramer.net/intro-to-the-ekf-step-4/
Documentation and structure based on: https://github.com/Attila94/EKF-SLAM
"""

class Map:
    """
    Class for map containing landmarks and current robot position   

    Attributes
    ----------
    state_mean: numpy array 
        mean of map (containing landmarks and current robot position), state vector of length 3 + (2 * n_landmarks) 
        initial state_mean defined as [x, y, heading] when observed landmarks is 0 
        state_mean will increase in length by 2 every time we add a new landmark to the map: append [x, y] of the landmark to end of array 
    state_cov: numpy array 
        covariance of all the landmarks and current robot position 
        covariance matrix of 3 + (2 * n_landmarks) by 3 + (2 * n_landmarks) with same structure as state_mean 
        first 3x3 submatrix are the covariances of the robot position 
    n_landmarks: int
        number of unique landmarks we have observed so far 
    
    """
    # TODO: tune default alpha values
    def __init__(self,
                 alphas=np.array([0.11, 0.01, 0.18, 0.08, 0.0, 0.0]),
                 Q_t=np.array([[11.7, 0.0],
                               [0.0, 0.18]])):
        self.alphas = alphas # motion model noise parameters
        self.Q_t = Q_t # measurement model noise parameters 
        self.state_mean = np.zeros(3)
        self.state_cov = 0.001 * np.ones((3, 3))
        self.n_landmarks = 0

    # odometry and measurement sample, update robot's pose 
    def __predict(self, u_t, dt):
        """Updates robot position given change in polar coordinates 

        Parameters 
        ----------
        u_t: list or numpy array 
            movement vector: [change in theta/rotation, change in translation/distance]
        dt: int 
            change in time since previous robot movement 

        Output
        ----------
        Updates self.state_mean and self.state_cov of the robot position (x, y, heading) 
        based on given changes in robot position 

        """

        n = len(self.state_mean)
        theta = self.state_mean[2]
        dtheta = dt * u_t[1] #change in theta
        dhalf_theta = dtheta / 2
        dtrans = dt * u_t[0] #change in translation 

        #calculate pose update from odometry (motion model)
        pose_update = np.array([dtrans * np.cos(theta + dhalf_theta),
                                dtrans * np.sin(theta + dhalf_theta),
                                dtheta])

        #updated state mean 
        F_x = np.append(np.eye(3),np.zeros((3,n-3)),axis=1)

        state_mean_bar = self.state_mean + (F_x.T).dot(pose_update)
        state_mean_bar[2] = (state_mean_bar[2] + 2 * np.pi) % (2 * np.pi)

        #calculate movement Jacobian 
        g_t = np.array([[0,0,dtrans*-np.sin(theta + dhalf_theta)],
                        [0,0,dtrans*np.cos(theta + dhalf_theta)],
                        [0,0,0]])
        G_t = np.eye(n) + (F_x.T).dot(g_t).dot(F_x)

        #calculate motion covariance in control space 
        M_t = np.array([[(self.alphas[0] * abs(u_t[0]) + self.alphas[1] * abs(u_t[1]))**2, 0],
                        [0, (self.alphas[2] * abs(u_t[0]) + self.alphas[3] * abs(u_t[1]))**2]])

        #calculate Jacobian to transform motion covariance to state space 
        V_t = np.array([[np.cos(theta + dhalf_theta), -0.5 * np.sin(theta + dhalf_theta)],
                        [np.sin(theta + dhalf_theta), 0.5 * np.cos(theta + dhalf_theta)],
                        [0, 1]])
        print(G_t)
        print(F_x)
        print(M_t)
        print(V_t) 
        #update state covariance 
        R_t = V_t.dot(M_t).dot(V_t.T)
        state_cov_bar = (G_t.dot(self.state_cov).dot(G_t.T)) + F_x.T.dot(R_t).dot(F_x)
        print(R_t)
        #updated state mean after the prediction 
        self.state_mean = state_mean_bar
        
        self.state_cov = state_cov_bar
        print(state_mean_bar)
        print(state_cov_bar) 

    

    def __update(self, z, dt):
        """Updates map of landmarks given potential observations 

        Parameters 
        ----------
        z: list of lists or 2D numpy array 
            list of potential measurements of landmarks
            each potential landmark is represented as a list of [x, y] from the current robot position 
            *****have to change to input cartesian coordinates (from the robot position?) 

        Output
        ----------
        Updates self.state_mean, self.state_cov, self.n_landmarks with new updated positions of landmarks 
        If loop closure occured, then self.loop_closure will be True 

        """
        #z: potential measurements of landmarks 
        state_mean_bar = self.state_mean
        state_cov_bar = self.state_cov
        n_landmarks = self.n_landmarks


        #loop over every measurement 
        for k in range(np.shape(z)[0]):
            #z[k]: [x, y]

            pred_z = np.zeros((2, n_landmarks+1))
            pred_psi = np.zeros((n_landmarks+1, 2, 2))
            pred_H = np.zeros((n_landmarks+1, 2, 2 * (n_landmarks+1)+3))
            pi_k = np.zeros((n_landmarks+1, 1))

            # x,y,heading of landmark based on robot's current position 
            #create temporary new landmark at observed position 
            temp_mark = np.array([z[k][0],z[k][1]])


            # TODO: possibly fix axis
            state_mean_temp = np.append(state_mean_bar, temp_mark, axis=0)
            state_cov_temp = np.append(state_cov_bar, np.zeros((np.shape(state_cov_bar)[0], 2)), axis=1)
            state_cov_temp = np.append(state_cov_temp, np.zeros((2, np.shape(state_cov_bar)[1] + 2)), axis=0)
                   

            #initialize state covariance for new landmark proportional to range measurement squared
            for ii in range(np.shape(state_cov_temp)[0] - 2, np.shape(state_cov_temp)[0]):
                state_cov_temp[ii][ii] = (z[k][0]**2) / 130

            
            #index for landmark with maximum association 
            max_j = -1
            min_pi = 10 * np.ones((2,1))

            #loop over all landmarks and compute likelihood of correspondence with new landmark 
            for j in range(n_landmarks + 1):

                delta = np.array([state_mean_temp[2*j+3] - state_mean_temp[0],
                                  state_mean_temp[2*j+4] - state_mean_temp[1]])
                print(delta)
                q = delta.dot(delta)
                r = np.sqrt(q)

                temp_theta = np.arctan2(delta[1], delta[0] - state_mean_temp[2])
                temp_theta = (temp_theta + 2 * np.pi) % (2 * np.pi)

                pred_z[:,j] = np.array([r, temp_theta], dtype=object)

                F_xj = np.zeros((5, 2 * (n_landmarks + 1) + 3))
                F_xj[0:3,0:3] = np.eye(3)
                F_xj[3:5,2*j+3:2*j+5] = np.eye(2)

                h_t = np.array([[-delta[0]/r, -delta[1]/r,  0,   delta[0]/r, delta[1]/r],
                                [delta[1]/q,  -delta[0]/q,   -1,  -delta[1]/q, delta[0]/q]])

                pred_H[j,:,:] = h_t @ F_xj
                print(h_t) 
                print(F_xj)
                print("r", r)
                print("q", q)
                pred_psi[j,:,:] = np.squeeze(pred_H[j,:,:]) @ state_cov_temp @ \
                                  np.transpose(np.squeeze(pred_H[j,:,:])) + self.Q_t
                print("J ", j)
                print("pred psi ", pred_psi[j,:,:])
                print((pred_H[j,:,:])) 
                print(state_cov_temp) 


                if j < n_landmarks:
                    
                    pi_k[j] = (np.transpose(z[k]-pred_z[:,j]) \
                                @ np.linalg.inv(np.squeeze(pred_psi[j,:,:]))) \
                                @ (z[k]-pred_z[:,j])
                    print((np.transpose(z[k]-pred_z[:,j])))
                    print("pred", ((pred_psi[j,:,:])))
                    print((z[k]-pred_z[:,j]))
                    print("j < n_landmarks j {} pi_k {}".format(j, pi_k[j]))
                else:
                    pi_k[j] = 0.84; # alpha: min mahalanobis distance to
                                    #        add landmark to map
                print("j{}, pi_k {}".format(j, pi_k[j]))

                #tracking two best associations 
                if pi_k[j] < min_pi[0]:
                    min_pi[1] = min_pi[0]
                    max_j = j
                    min_pi[0] = pi_k[j]
        
            H = np.squeeze(pred_H[max_j,:,:])

            #best association must be significantly better than second better than second best 
            #otws, measurement is thrown out 
            print("max j", max_j) 
            print("n landmarks ", n_landmarks)
            print(min_pi)
            if (min_pi[1] / min_pi[0] > 1.6):
                if max_j >= n_landmarks:
                    #new landmark is added, expand state and covariance matrices
                    state_mean_bar = state_mean_temp
                    state_cov_bar = state_cov_temp
                    n_landmarks += 1
                    

                else:
                    #if measurement is associated with existing landmark, truncate h matrix to prevent dim. mismatch
                    H = H[:,0:2 * n_landmarks + 3]

                    K = state_cov_bar @ H.T @ np.linalg.inv(np.squeeze(pred_psi[max_j,:,:]))

                    state_mean_bar = state_mean_bar + K @ (z[k] - pred_z[:,max_j])

                    state_mean_bar[2] = (self.state_mean[2] + 2 * np.pi) % (2 * np.pi)

                    state_cov_bar = (np.eye(np.shape(state_cov_bar)[0]) - K @ H) @ state_cov_bar

        #update state mean and covariance (map itself) 
        self.state_mean = state_mean_bar
        self.state_cov = state_cov_bar
        self.n_landmarks = n_landmarks

    def update_map(self, movement, measurements, dt):
        """Updates map with given robot movement and landmark measurements after change in time dt 

        Parameters 
        ----------
        movement: list or numpy array 
            movement vector: [change in theta/rotation, change in translation/distance]
            *****change??
        measurement: list of lists or 2D numpy array 
            list of potential measurements of landmarks
            each potential landmark is represented as a list of [range, bearing] from the current robot position 
            *****have to change to input cartesian coordinates (from the robot position?) 
        dt: int 
            change in time since previous robot movement and observation 

        Output
        ----------
        Executes prediction of robot position and update of landmark positions
        Updates the map of robot position and landmark postions 

        """
        self.__predict(movement, dt)
        self.__update(measurements, dt)

    def get_state(self, get_cov=False):
        if get_cov:
            return self.state_mean, self.state_cov
        else:
            return self.state_mean
    
    

if __name__ == "__main__":
    map = Map()
    # print(map.get_state())
    move1 = np.array([1, 0])
    meas1 = np.array([[3, 0], [1, np.pi/2]])
    map.update_map(move1, meas1, 1)
    print(map.get_state(get_cov=True)) 
    #move2 = np.array([1, 0])
    #meas2 = np.array([[1.001, 0]])
    move2 = [1, 0]
    meas2 = [[4, 0]]
    map.update_map(move2, meas2, 1)
    print(map.get_state(get_cov=True ))
