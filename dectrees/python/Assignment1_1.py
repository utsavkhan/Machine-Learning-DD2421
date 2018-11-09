import monkdata as m
import dtree as dt
import drawtree_qt4 as draw
import random
import matplotlib.pyplot as plt
import numpy as np
import pylab


#print(dt.entropy(m.monk3));
"""
print(dt.averageGain(m.monk1,m.attributes[0]));
print(dt.averageGain(m.monk1,m.attributes[1]));
print(dt.averageGain(m.monk1,m.attributes[2]));
print(dt.averageGain(m.monk1,m.attributes[3]));
print(dt.averageGain(m.monk1,m.attributes[4]));
print(dt.averageGain(m.monk1,m.attributes[5]));


print(dt.averageGain(m.monk2,m.attributes[0]));
print(dt.averageGain(m.monk2,m.attributes[1]));
print(dt.averageGain(m.monk2,m.attributes[2]));
print(dt.averageGain(m.monk2,m.attributes[3]));
print(dt.averageGain(m.monk2,m.attributes[4]));
print(dt.averageGain(m.monk2,m.attributes[5]));


print(dt.averageGain(m.monk3,m.attributes[0]));
print(dt.averageGain(m.monk3,m.attributes[1]));
print(dt.averageGain(m.monk3,m.attributes[2]));
print(dt.averageGain(m.monk3,m.attributes[3]));
print(dt.averageGain(m.monk3,m.attributes[4]));
print(dt.averageGain(m.monk3,m.attributes[5]));

"""

#print(dt.select(m.monk1,m.attributes[0],1))

t=dt.buildTree(m.monk2, m.attributes);
print("training error =",1-dt.check(t,m.monk2))
print("test error =",1-dt.check(t,m.monk2test))
draw.drawTree(t)


def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


minAdded = np.array([])

numberOfIterations = 100;

for z in range(0,numberOfIterations):
    minofmin = np.array([])
    for n in frange(0.3,0.8,0.1):
        monk1train, monk1val = partition(m.monk1, n)
        t=dt.buildTree(monk1train, m.attributes);
        t_allPrunes = dt.allPruned(t)
        
        minValue = np.array([]);
        
        for i in range(0,len(t_allPrunes)-1):
            minValue = np.append(minValue,1-dt.check(t_allPrunes[i],monk1val));
            
        minofmin = np.append(minofmin,min(minValue))
    minAdded = np.append(minAdded,minofmin,axis=0)

print(minAdded)
minAdded = np.reshape(minAdded,(numberOfIterations,6))
minSum = np.sum(minAdded,axis=0)/numberOfIterations
print(minSum)



pylab.figure(1)
pylab.plot([0.3,0.4,0.5,0.6,0.7,0.8],minSum,'-ro',label='Mean Error of Monk1')
pylab.ylabel('Means of Errors')
pylab.xlabel('Fractions')
pylab.title('Means of Errors vs Fractions')
pylab.legend(loc='upper right')
pylab.show()


minVar = np.var(minAdded,axis=0);
pylab.figure(2)
pylab.plot([0.3,0.4,0.5,0.6,0.7,0.8],minVar,'-bo',label='Variance of Error of Monk1')
pylab.ylabel('Variance of Errors')
pylab.xlabel('Fractions')
pylab.title('Variance of Errors vs Fractions')
pylab.legend(loc='upper right')
pylab.show()


#draw.drawTree(dt.allPruned(t)[3])