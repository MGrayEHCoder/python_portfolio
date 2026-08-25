#Monte Carlo Simulation of Ising Model
#Michael Gray
#10/22/24


from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

Gridsize = 32
spinState = np.ones((Gridsize,Gridsize))

def InitializeState(spinState):
    
    for x in range(len(spinState[0])):
        for y in range(len(spinState[0])):
            randnum = random.random()
            if randnum < 0.5:
                spinState[x][y] = -1
    
    return spinState


def UpdateSpin(x,y,spinState,T):
    spin = spinState[x][y]
    
    Energy = -spin*(spinState[x][(y+1)%Gridsize] + spinState[x][(y-1)%Gridsize] + spinState[(x+1)%Gridsize][y] + spinState[(x-1)%Gridsize][y])
    

    if (-2*Energy > 0):
        prob = np.exp(-(1/T)*-2*Energy)
        if random.random() < prob:
            spin = -spin
    else:
        spin = -spin
    
    
    return spin

def calcStateEnergy(spinState):
    Et = 0
    for iy in range(Gridsize):
        for ix in range(Gridsize):
            Et += -spinState[ix][iy]*(spinState[(ix+1)%Gridsize][iy] + spinState[ix][(iy-1)%Gridsize])
    return Et

def UpdateState(spinState,T):
    for i in range(1000):
        randx = random.randint(0,Gridsize-1)
        randy = random.randint(0,Gridsize-1)
        spinState[randx][randy] = UpdateSpin(randx,randy,spinState,T)
    Et = calcStateEnergy(spinState)
    return spinState, Et


def CalcStatesE(T,timeSteps = 100,spinState = np.ones((Gridsize,Gridsize))):
    spinState  = InitializeState(spinState)
    spinStates = np.zeros((timeSteps,Gridsize,Gridsize))
    spinStates[0] = spinState
    Ets = np.zeros(timeSteps)
    Ets[0] = calcStateEnergy(spinState)
    for i in range(timeSteps-1):
        spinStates[i+1],Ets[i+1] = UpdateState(spinStates[i],T)
    
    return Ets


def getData(timesSteps):
    Ts = np.arange(1.5,3.5,0.2)
    df = pd.DataFrame()
    df['Steps'] = np.linspace(1,timesSteps*1000-1,timesSteps)
    for T in Ts:
        Ets = CalcStatesE(T,timesSteps)
        df[f'Energy @ {T}'] = Ets
    
    df.to_csv(DATA_DIR / 'IsingModelData.csv')



getData(2000)    
'''
fig, ax = plt.subplots()

ts = np.linspace(0,1,timeSteps)


img = ax.imshow(spinStates[0],cmap = 'binary')

def update(frame):
    img.set_data(spinStates[frame])
    return img,

ani = animation.FuncAnimation(fig, update,frames = range(timeSteps),blit = False,interval = 50)

ani.save('IsingModel.gif')

fig, ax = plt.subplots()
ax.plot(ts,Ets)
fig.savefig('TotalEnergyVsTime.png')

'''

