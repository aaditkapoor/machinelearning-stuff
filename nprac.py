import numpy as np

x = np.array([ [0,0,1], [0,1,1] , [1,0,0] ] )
y = np.array([ [0,11,0]]).T

syn0 = 2 * np.random.random((3,4)) - 1
syn1 = 2 * np.random.random((4,1)) - 1


for j in range(100):
    l1 = 1/(1+np.exp(-(np.dot(x, syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1, syn1))))

    l2_d = y - l2 * (l2 * (1-l2))
    l1_d  = l2_d.dot(syn1.T) * (l1 * (1-l1))

    syn1 += l1.T.dot(l2_d)
    syn0 += x.T.dot(l1_d)

 


print (l1)
