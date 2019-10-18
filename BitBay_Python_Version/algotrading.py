
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy import stats
import random
import os
import matplotlib.pyplot as plt

path = 'c:\\Users\\Qiao Yu\\desktop\\BTCpredictor'
os.chdir(path)
os.getcwd()

df = pd.read_csv('okcoin5s.csv',header = None)
prices = list(df[1])
askVolume = df[2]
bidVolume = df[3]
# turns 5s to 10s steps, abstract all the even position nums;
prices = prices[0::2] 
askVolume = askVolume[0::2]
bidVolume = bidVolume[0::2]

b = 20000
askVolume = askVolume[b:]
bidVolume = bidVolume[b:]

prices1 = prices[0:b]
prices2 = prices[b:b*2]
prices3 = prices[b*2:]

# step 1: creating intervals S_j

# create list of all 720*10s, 360*10s and 180*10s intervals
# each item is (interval of prices, NEXT TEN SEC interval price change)

priceDiff = np.diff(prices)
validIntSize = len(prices1)-750  #valid interval size
interval720s = np.zeros([validIntSize,720+1])
interval360s = np.zeros([validIntSize,360+1])
interval180s = np.zeros([validIntSize,180+1])

for i in range(validIntSize):   
    interval180s[i] = prices1[i:i+180]+[priceDiff[i+179]]
    interval360s[i] = prices1[i:i+360]+[priceDiff[i+359]] 
    interval720s[i] = prices1[i:i+720]+[priceDiff[i+719]]  

# we k-means cluster all 3 interval lists to get the 20 best patterns
# for each of the interval lists;
random.seed(15)
clusters=100
estimator = KMeans(n_clusters=100, n_init=10, max_iter=10000,   n_jobs=-1)
kmeans180s1= estimator.fit(interval180s).cluster_centers_ # 29.4s
kmeans360s1= estimator.fit(interval360s).cluster_centers_ # 35.7s
kmeans720s1= estimator.fit(interval720s).cluster_centers_ #73s

# consider: for speed, use similarity instead of L2 norm for kmeans
# regularize so the mean = 0 and std =1
# don't regularize the price jump (at the last index)
kmeans180s1[:,0:180] = stats.zscore(kmeans180s1[:,0:180])
kmeans360s1[:,0:360] = stats.zscore(kmeans360s1[:,0:360])
kmeans720s1[:,0:720] = stats.zscore(kmeans720s1[:,0:720])


# use sample entropy to choose interesting/effective patterns 
r180s = np.zeros(clusters)
r360s = np.zeros(clusters)
r720s = np.zeros(clusters)
for i in range(clusters):
    r180s[i] = round(0.2 * np.std(kmeans180s1[i,0:180], ddof = 1), 2)
    r360s[i] = round(0.2 * np.std(kmeans360s1[i,0:360], ddof = 1), 2)
    r720s[i] = round(0.2 * np.std(kmeans720s1[i,0:720], ddof = 1), 2)


entropy180 = np.zeros(clusters)
entropy360 = np.zeros(clusters)
entropy720 = np.zeros(clusters)
m = 2
for i in range(clusters):
	entropy180[i] = samplEn(kmeans180s1[i,0:180],m,r180s[i])
	entropy360[i] = samplEn(kmeans360s1[i,0:360],m,r360s[i])
	entropy720[i] = samplEn(kmeans720s1[i,0:720],m,r720s[i])
 

# sort by top 20 highest sample entropy and save these
IDX180= [i[0] for i in sorted(enumerate(entropy180),reverse = True, key=lambda x:x[1])][0:20]
IDX360= [i[0] for i in sorted(enumerate(entropy360),reverse = True, key=lambda x:x[1])][0:20]
IDX720= [i[0] for i in sorted(enumerate(entropy720),reverse = True, key=lambda x:x[1])][0:20]

kmeans180s=kmeans180s1[IDX180,:]
kmeans360s=kmeans360s1[IDX360,:]
kmeans720s=kmeans720s1[IDX720,:]

print('Finished clustering and normalizing')

#step 2: predicting average price change dp_j and learning parameters w_i
#using Bayesian regression
#equation:
#dp = w0 + w1*dp1 + w2*dp2 + w3*dp3 + w4*r
numFeatures = 3
start = 730
numPoints = len(prices2) - start
regressorX = np.zeros((numPoints, numFeatures))
regressorY = np.zeros(numPoints)
price180 = np.zeros((numPoints, 180))
price360 = np.zeros((numPoints, 360))
price720 = np.zeros((numPoints, 720))

for i in range(start, len(prices2)):
    price180[i-start] = stats.zscore(prices2[i-180:i], ddof=1)      
    price360[i-start] = stats.zscore(prices2[i-360:i], ddof=1)      
    price720[i-start] = stats.zscore(prices2[i-720:i], ddof=1)

dp1 = [0]*numPoints
dp2 = [0]*numPoints
dp3 = [0]*numPoints

for i in range(numPoints):    
    dp1[i] = bayesian(price180[i], kmeans180s)
    dp2[i] = bayesian(price360[i], kmeans360s)
    dp3[i] = bayesian(price720[i], kmeans720s)

regressorX =  np.transpose(np.array((dp1,dp2,dp3)))

regressorY = [0]*numPoints
for i in range(numPoints):     
    regressorY[i] = prices2[i+start]-prices2[i+start-1]  

'''
run Rundeopt;

% retrieve weights 
theta = zeros(numFeatures, 1);
for k=1:numFeatures
  theta(k) = FVr_x(k);
end
theta0 = FVr_x(k+1);

print('Finished regression, ready to trade');
'''

#  Start trading with last list of prices

[error, prob, sell, buy, cash] = btrade(prices3, theta0, theta, fee = 1)
# set up plots
result_plots(prices3, buy, sell, prob, cash, error)











