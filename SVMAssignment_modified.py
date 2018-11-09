import numpy as np
import random as rnd
import math as mf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(100)
classA = np.concatenate((np.random.randn(10,2)*0.2+[-2,-2],np.random.randn(10,2)*0.2+[2,2]));
classB = np.random.randn(20,2)*.3+[0,0];
inputs = np.concatenate((classA,classB));
targets = np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])));
N = inputs.shape[0];
permute = list(range(N));
rnd.shuffle(permute);
inputs = inputs[permute,:];
targets = targets[permute]

# Kernel Function
def linear_kernel(x, y):
    return np.dot(x, y)


def polynomial_kernel(x, y, p=4):
    return np.power((np.dot(x, y) + 1), p)


def radial_basis_function_kernel(x, y, sigma=1):
    diff = np.subtract(x, y)
    return mf.exp((-np.dot(diff, diff)) / (2 * sigma * sigma))

t = targets;




P=np.zeros((N,N));
for i in range(0,N):
    for j in range(0,N):
        P[i][j] = t[i]*t[j]*np.dot(inputs[i],inputs[j]);


start = np.zeros(N);

def objective(alpha):
   result_i = 0
   for i in range(0, N):
       # result_j = 0
       for j in range(0, N):
           # result_j = result_j + alpha[i] * alpha[j] * P[i][j]
           result_i += 0.5 * alpha[i] * alpha[j] * P[i][j]
   result2 = np.sum(alpha)
   return result_i - result2

def zerofun(alpha):
    return np.dot(alpha,targets)

C = None
B = [(0, C) for b in range(N)]

ret = minimize(objective, start, bounds=B, constraints={'type': 'eq', 'fun': zerofun});
alpha = ret['x'];

"""extInputs = [];
extTargets = [];
extalpha= []

for i in range(N):
    if alpha[i] > 1e-5:
        extInputs.append(inputs[i]);
        extTargets.append(targets[i]);
        extalpha.append(alpha[i]);"""

 # Filter non-zero vectors
support_vectors = []
for i in range(N):
    if alpha[i] > 1.e-5:
        support_vectors.append((alpha[i], inputs[i][0], inputs[i][1], targets[i]))
        
#calculate b usning equation
b = 0;
"""for i in range(N):
    if 1e-5 < alpha[i] < C:
        for j in range(N):
            b += alpha[j]*targets[j]*np.dot(inputs[i],inputs[j]);
        b -= targets[i];"""
        
for j in range(N):
    if 1.e-5 < alpha[j]:
        b = np.sum(alpha*targets*np.dot([support_vectors[0][1:3]],inputs.transpose()));
    
b -= support_vectors[0][3];


    
               
           
def indicator(s, x, y, kernel):
    ind = 0.0
    for i in range(len(s)):
        ind += s[i][0] * s[i][3] * kernel([x, y], [s[i][1], s[i][2]])
    ind -= b
    return ind
           
    
#plotting data
plt.plot([p[0]for p in classA],[p[1] for p in classA],'b.')   
plt.plot([p[0]for p in classB],[p[1] for p in classB],'r.')
plt.axis('equal')
#plt.show()    

#plotting decision boundary    

xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)

grid = np.array([[indicator(support_vectors,x, y, polynomial_kernel)
                  for x in xgrid]
                 for y in ygrid])

plt.contour(xgrid, ygrid, grid,
            (-1.0, 0.0, 1.0),
            colors=('red', 'black', 'blue'),
            linewidths=(1, 3, 1))
plt.show()  
plt.savefig('svmplot.pdf')     