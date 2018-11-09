import monkdata as m
import dtree as dt
#import drawtree_qt4 as dtree

print (dt.entropy(m.monk3))
print (dt.averageGain(m.monk1 , m.attributes[0]))

t1=dt.buildTree(m.monk1, m.attributes);
t2=dt.buildTree(m.monk2, m.attributes);
t3=dt.buildTree(m.monk3, m.attributes);

print("Error for monk1 data set" , 1-dt.check(t1, m.monk1test))
print("Error for monk2 data set" , 1-dt.check(t2, m.monk1test))
print("Error for monk3 data set" , 1-dt.check(t3, m.monk1test))


import random
def partition(data, fraction):
         ldata = list(data)
         random.shuffle(ldata)
         breakPoint = int(len(ldata) * fraction)
         return ldata[:breakPoint], ldata[breakPoint:]
monk1train, monk1val = partition(m.monk1, 0.6)

monk1traintree= dt.buildTree(monk1train,m.attributes)
pruned_monk1 = dt.allPruned(monk1traintree)

for i in range(0,len(pruned_monk1)):
    print("Error for tree number",i,"=",1-dt.check(pruned_monk1[i],monk1val))