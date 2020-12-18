#import packages
import pandas as pd
import numpy as np
from scipy.linalg import svd

class LeeCarter():
    def __init__(self, no_latent_factors=1):
        self.ax = None
        self.bxs = pd.DataFrame()
        self.kts = pd.DataFrame()

        self.training_data = None
        self.no_latent_factors = no_latent_factors
      
    def fit(self, log_mortality):
            
        log_mortality_mean = log_mortality.mean(axis=1)
        train_mort = log_mortality.sub(log_mortality_mean, axis=0)
        train_mort.replace([np.inf, -np.inf], np.nan, inplace=True)

        #fit LC model by using Singular Value Decomposition
        U, S, VT = svd(train_mort)

        kts = {}
        bxs = {}

        for i in range(self.no_latent_factors):
            bx = U[:, i] * S[i]
            kt = VT[i]

            c1 = kt.mean()
            c2 = bx.sum()
            bx /= c2
            kt = (kt-c1)*c2
            kts[i] = kt
            bxs[i] = bx

        self.ax = pd.Series(log_mortality_mean, index=log_mortality.index)
        self.bxs = pd.DataFrame(bxs, index=log_mortality.index) 
        self.kts = pd.DataFrame(kts, index=log_mortality.columns)
                  
        print("Model fitted.")
    
    def predict(self, log_mortality):
                    
        log_mortality_mean = log_mortality.mean(axis=1)
        train_mort = log_mortality.sub(log_mortality_mean, axis=0)
        train_mort.replace([np.inf, -np.inf], np.nan, inplace=True)

        predicted = self.bxs @ self.kts.T
        predicted = predicted.add(log_mortality_mean, axis=0)

        return predicted