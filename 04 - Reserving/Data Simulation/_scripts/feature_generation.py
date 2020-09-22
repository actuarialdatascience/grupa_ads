def feature_generation(V,lob,inflation,seed1):

### Definition of the function Feature.Generation that generates features ###
#############################################################################

### Define the function Feature.Generation
### We have the following inputs:
### V = totally expected number of claims
### LoB.dist = categorical distribution for the allocation of the claims to the 4 lines of business
### inflation = growth parameters (per LoB) for the numbers of claims in the 12 accident years
### seed1 = set the seed for reproducibility

    import numpy as np
    import pandas as pd
    import scipy as sp
    from scipy import stats
    import math
    import random

# Weights in LoB.dist have to be nonnegative
    if len([num for num in lob if num < 0]) > 0:
        print("ERROR: The weights determining the distribution amongst the lines of business cannot be negative.")
        features=pd.DataFrame(np.zeros((0,7)),columns=['ClNr', 'LoB', 'cc', 'AY', 'AQ', 'age', 'inj_part'])
        return features

# Preparation for the data set "features" 
# Determine the number of claims per line of business
    vdist = np.zeros((13,4))
    random.seed(seed1)
    rozkl=np.random.multinomial(V, lob, size=1)
# Determine the number of claims per accident year (for all lines of business)
    vdist[0]=rozkl
    random.seed(seed1+1)
    vmean=np.array([inflation,]*11)
    vsd=np.array([np.abs(inflation),]*11)
    vdist[2:13,:4]=np.random.normal(vmean,vsd)
    vdist[2:13,:4]=np.cumsum(vdist[2:13,:4],axis=0)
    vdist[1:13,:4]=np.exp(vdist[1:13,:4])
    vdist_pom=vdist[1:13,:4].transpose()/vdist[1:13,:4].sum(axis=0)[:,None]
    vdist[1:13,:4]=vdist_pom.transpose()
    vdist_pom2=vdist[1:13,:4].transpose()*vdist[0,:4][:,None]
    vdist[1:13,:4]=vdist_pom2.transpose()
    random.seed(seed1+2)
    vdist[1:13,:4]=np.random.poisson(vdist[1:13,:4],(12,4))
    vdist[0,:4]=vdist[1:13,:4].sum(axis=0)[:,None].transpose()

# Create the array where we will store the observations
    features=pd.DataFrame(np.zeros((int(vdist[0,:4].sum()),7)),columns=['ClNr', 'LoB', 'cc', 'AY', 'AQ', 'age', 'inj_part'])

# Store line of business
    features.LoB=np.repeat(np.array([1,2,3,4]),vdist[0,:4].astype(int),axis=0).transpose()

# Store accident year
    a=np.concatenate(vdist[1:13,:4].transpose())[np.where(np.concatenate(vdist[1:13,:4].transpose())>0)]
    b=np.where(np.concatenate(vdist[1:13,:4].transpose())>0)[0]%12
    features.AY=(np.repeat(b,a.astype(int),axis=0)+1994).transpose()

# Add artificial observations that prevent data sets below from being empty
    features = features.append({'ClNr':-1,'LoB':1,'cc':0,'AY':1994,'AQ':0,'age':0,'inj_part':0}, ignore_index=True)
    features = features.append({'ClNr':-1,'LoB':1,'cc':0,'AY':1994,'AQ':0,'age':0,'inj_part':0}, ignore_index=True)
    features = features.append({'ClNr':-1,'LoB':3,'cc':0,'AY':1994,'AQ':0,'age':0,'inj_part':0}, ignore_index=True)
    features = features.append({'ClNr':-1,'LoB':3,'cc':0,'AY':1994,'AQ':0,'age':0,'inj_part':0}, ignore_index=True)


#feature generation for LoB 1 and 2

# Generate observations from a multivariate normal distribution
    random.seed(seed1+3)
    Sigma=pd.DataFrame(pd.read_table('./dane/Simulation.Machine.V1/Feature.Generation.Parameters/LoB1and2/Covariances/Covariance.txt', sep='\t'))
    help=np.random.multivariate_normal(mean=[0,0,0,0], cov=Sigma, size=len(features[features.LoB<=2]))
# Transform marginals such that they have a uniform distribution on [0,1]
    help=sp.stats.norm.cdf(help)

#Transform marginals such that they have the appropriate distribution

#Claims code (cc)
# Source the parameters
    param=pd.DataFrame(pd.read_table('./dane/Simulation.Machine.V1/Feature.Generation.Parameters/LoB1and2/Parameters/cc.txt', sep='\t'))
    help[:,0]=np.ceil(f_cc(np.array(help[:,0]),param.loc[0,'x'],param.loc[1,'x'],param.loc[2,'x']))
    translator=pd.DataFrame(pd.read_table('./dane/Simulation.Machine.V1/Feature.Generation.Parameters/LoB1and2/Translators/cc.txt', sep='\t'))
    help[:,0]=np.array(translator.loc[help[:,0]-1,'x']).transpose()

# Accident quarter (AQ)
    help[:,1]=np.ceil(help[:,1]*4)

# Age of the injured (age) 
    param2=pd.DataFrame(pd.read_table('./dane/Simulation.Machine.V1/Feature.Generation.Parameters/LoB1and2/Parameters/age.txt', sep='\t'))
    help[np.where(help[:,2]<=param2.loc[4,'x'])[0],2]=np.ceil(f_age1(np.array(help[np.where(help[:,2]<=param2.loc[4,'x'])[0],2]),param2.loc[0,'x'],param2.loc[1,'x'],param2.loc[3,'x']))
    help[np.where(help[:,2]<=1)[0],2]=np.ceil(f_age2(np.array(help[np.where(help[:,2]<=1)[0],2]),param2.loc[0,'x'],param2.loc[1,'x'],param2.loc[2,'x'],param2.loc[3,'x']))

# Injured Part (inj_part)
    param3=pd.DataFrame(pd.read_table('./dane/Simulation.Machine.V1/Feature.Generation.Parameters/LoB1and2/Parameters/inj_part.txt', sep='\t'))
    help[:,3]=np.ceil(f_inj_part(help[:,3],param3.loc[0,'x'],param3.loc[1,'x'],param3.loc[2,'x']))
    translator2=pd.DataFrame(pd.read_table('./dane/Simulation.Machine.V1/Feature.Generation.Parameters/LoB1and2/Translators/inj_part.txt', sep='\t'))
    help[:,3]=np.array(translator2.loc[help[:,3]-1,'x']).transpose()

# Store cc, AQ, age and inj_part in data set features
    features.at[np.where(features.LoB <= 2)[0],'cc']=help[:,0]
    features.at[np.where(features.LoB <= 2)[0],'AQ']=help[:,1]
    features.at[np.where(features.LoB <= 2)[0],'age']=help[:,2]
    features.at[np.where(features.LoB <= 2)[0],'inj_part']=help[:,3]


# Feature Generation for LoB 3 and 4
    random.seed(seed1+4)
    Sigma=pd.DataFrame(pd.read_table('./dane/Simulation.Machine.V1/Feature.Generation.Parameters/LoB3and4/Covariances/Covariance.txt', sep='\t'))
    help=np.random.multivariate_normal(mean=[0,0,0,0], cov=Sigma, size=len(features[features.LoB>2]))
    help=sp.stats.norm.cdf(help)

#Transform marginals such that they have the appropriate distribution
#Claims code (cc)
# Source the parameters
    param=pd.DataFrame(pd.read_table('./dane/Simulation.Machine.V1/Feature.Generation.Parameters/LoB3and4/Parameters/cc.txt', sep='\t'))
    help[:,0]=np.ceil(f_cc(np.array(help[:,0]),param.loc[0,'x'],param.loc[1,'x'],param.loc[2,'x']))
    translator=pd.DataFrame(pd.read_table('./dane/Simulation.Machine.V1/Feature.Generation.Parameters/LoB3and4/Translators/cc.txt', sep='\t'))
    help[:,0]=np.array(translator.loc[help[:,0]-1,'x']).transpose()

# Accident quarter (AQ)
    help[:,1]=np.ceil(help[:,1]*4)

# Age of the injured (age) 
    param2=pd.DataFrame(pd.read_table('./dane/Simulation.Machine.V1/Feature.Generation.Parameters/LoB3and4/Parameters/age.txt', sep='\t'))
    help[np.where(help[:,2]<=param2.loc[4,'x'])[0],2]=np.ceil(f_age1(np.array(help[np.where(help[:,2]<=param2.loc[4,'x'])[0],2]),param2.loc[0,'x'],param2.loc[1,'x'],param2.loc[3,'x']))
    help[np.where(help[:,2]<=1)[0],2]=np.ceil(f_age2(np.array(help[np.where(help[:,2]<=1)[0],2]),param2.loc[0,'x'],param2.loc[1,'x'],param2.loc[2,'x'],param2.loc[3,'x']))

# Injured Part (inj_part)
    param3=pd.DataFrame(pd.read_table('./dane/Simulation.Machine.V1/Feature.Generation.Parameters/LoB3and4/Parameters/inj_part.txt', sep='\t'))
    help[:,3]=np.ceil(f_inj_part(help[:,3],param3.loc[0,'x'],param3.loc[1,'x'],param3.loc[2,'x']))
    translator2=pd.DataFrame(pd.read_table('./dane/Simulation.Machine.V1/Feature.Generation.Parameters/LoB3and4/Translators/inj_part.txt', sep='\t'))
    help[:,3]=np.array(translator2.loc[help[:,3]-1,'x']).transpose()

# Store cc, AQ, age and inj_part in data set features
    features.at[np.where(features.LoB > 2)[0],'cc']=help[:,0]
    features.at[np.where(features.LoB > 2)[0],'AQ']=help[:,1]
    features.at[np.where(features.LoB > 2)[0],'age']=help[:,2]
    features.at[np.where(features.LoB > 2)[0],'inj_part']=help[:,3]


# Order the data: first random order then order according to the accident year AY 
    random.seed(seed1+5)
    order1=random.sample(list(range(0,len(features))),len(features))
    features=features.reindex(order1)
    features=features.sort_values(by='AY')

# Remove the artificial observations
    features=features.drop(features[features.ClNr==-1].index)

# Number the claims from 1 to nrow(features)
    features.at[:,'ClNr']=(np.array(range(0,len(features)))+1)

# Adjust the rownames
    features.index=(np.array(range(0,len(features)))+1)

# Output
    print(features)
    return features
  



def f_cc(x,alpha,beta,const):
    import math
    import numpy as np
    return (np.exp(alpha*x/const+math.log(beta))-beta)/alpha

def f_age1(x,alpha,beta,const):
    import math
    import numpy as np
    return np.sqrt(42*x/(beta*const)+math.pow(21,2)*math.pow(alpha,2)/math.pow(beta,2))-21*alpha/beta+14

def f_age2(x,alpha,beta,gamma,const):
    import math
    import numpy as np
    return -np.sqrt(-70*x/(gamma*const)+math.pow(35,2)/math.pow(gamma,2)*math.pow((alpha+beta),2)+21*70*alpha/gamma+10.5*70*beta/gamma)+35*alpha/gamma+35*beta/gamma+35

def f_inj_part(x,alpha,beta,const):
    import math
    import numpy as np
    return (np.exp(alpha*x/const+math.log(beta))-beta)/alpha
    


