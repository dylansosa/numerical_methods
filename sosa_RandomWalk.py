"""
Dylan S
Numerical and Scientific Methods
Day 25
Random Walks
"""
import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import animation
import random
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

random.seed(None)

nsteps = 10000 #10,000
nwalks = 5
rlist = np.zeros(nwalks)

plt.title('Random Walk with 10,000 Steps')
plt.xlabel('x')
plt.ylabel('y')
Rrms = 0
for n in range(nwalks):
    ypos = 0
    xpos = 0
    xpos_list = np.zeros(nsteps)
    ypos_list = np.zeros(nsteps)
    for i in range(nsteps):
        direction = random.random()
        if direction > 0.5:
            x = random.random()
            if x > 0.5:
                xpos += 1 #left or right
            else:
                xpos -= 1
        if direction < 0.5:
            y = random.random()
            if y > 0.5:
                ypos += 1 #up or down
            else:
                ypos -= 1
        ypos_list[i] = ypos
        xpos_list[i] = xpos
    print("Walker {}\'s Final Position is:".format(n+1),m.sqrt(xpos**2+ypos**2))
    Rrms = Rrms + m.sqrt(xpos**2+ypos**2)**2
    plt.plot(xpos_list, ypos_list,'')
Rrms = m.sqrt(Rrms/nwalks)
Rrms_nsteps = m.sqrt(nsteps)

print('\nExpected root-mean-square of the displacement is:', Rrms_nsteps,'\nObserved root-mean-square of the displacement is:', Rrms)
plt.show()
