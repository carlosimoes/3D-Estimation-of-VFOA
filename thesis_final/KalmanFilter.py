import numpy as np

class KalmanFilter3D(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None, dt = None):

        if F is None and dt is not None:
            self.F = np.array([[1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1]])
        else:
            self.F = F
        self.n = self.F.shape[1] # Get size of lines F
        self.H=np.eye(self.n)
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0
        self.dt= dt
        self.x_prev=self.x
        self.x_atual=self.x
        self.velo=np.zeros((3, 1))
        self.velo_ant=np.zeros((1, 3))
        self.result=np.zeros((3, 1))
        self.initR=0
    
    def calc_velocity(self):
        self.velo=((self.x_atual-self.x_prev)/self.dt)
        self.x=np.append(self.result,self.velo_ant).reshape(6,1)
        self.velo_ant=self.velo
         
    def real_time_R_update(self):
        self.s_atual=1/(1+np.exp(-np.absolute(self.velo)))
        
        if self.initR is 0:
            self.Ratual=self.s_atual*100
            self.R_atual=self.Ratual
        else:
            self.R_atual=self.Ratual=np.zeros((1, 3))
            for i in range(self.s_atual.shape[0]):
                if self.s_atual[i]>0.94:
                    r_=10
                else:
                    r_=100*self.s_atual[i]
                self.R_atual[0][i]=r_

                if self.s_atual[i]<self.s_prev[0][i]:
                    self.Ratual[0][i]=0.3*self.R_atual[0][i]+(1-0.3)*self.R_prev[0][i]
                else:
                    self.Ratual[0][i]=self.R_atual[0][i]
        self.s_prev=self.s_atual.reshape(1,3)
        self.R_prev=self.R_atual      
        self.R = np.diag([self.Ratual[0][0],self.Ratual[0][1], self.Ratual[0][2], 1, 1, 1])

    def predict(self, u = 0):
        self.initR=self.initR+1
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        self.result=self.x.T[0][0:3]
        return self.x

    def Kalman(self,z):
        if self.initR is 0:
            self.x_atual=z
            self.velo=np.zeros((1, 3))
            self.x=np.append(self.x_atual,self.velo).reshape(6,1)
            
        else:
            self.x_atual=z
            self.calc_velocity()
        # print("-----------------------", self.initR,"---------------")
        # print("X_prev: ",self.x, "\nVelocidade: ",self.velo,"\nP prev: ", self.P)
        self.real_time_R_update()
        # print("R: ", self.R)
        # print()
        self.predict()
        result=self.update(z=np.append(self.x_atual,self.velo).reshape(6,1))
        # print("X new:",self.x, "\nP new: ", self.P)
        self.x_prev=self.x_atual
        return(result[0:3].reshape(1,3).tolist()[0])
        
        



    def example():
        dt = 0.0330    

        kf = KalmanFilter3D(dt=dt)
        measurements =np.loadtxt('data.txt')
        predictions=[]
        for z in measurements:
            predictions.append(kf.Kalman(z))
        error=np.sum(np.power(predictions-measurements,2), axis=0)
        print(error)
        f = open("pred.txt", "w")
        for pred in predictions:
            f.write(str(pred)+'\n')
        f.close()
        import matplotlib.pyplot as plt
        plt.plot(range(len(measurements)), measurements, label = 'Measurements')
        plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
        plt.legend()
        plt.show()

        

if __name__ == '__main__':
    KalmanFilter3D.example()