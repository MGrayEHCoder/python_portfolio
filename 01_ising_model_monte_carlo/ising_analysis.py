#Ising Model Analysis
#Michael Gray
#11/12/2024

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / 'IsingModelData.csv')
Ts = np.arange(1.5,3.5,0.2)

def plotEtsVsSteps():
    fig,ax = plt.subplots()
    Steps = df['Steps']
    for T in Ts:
        Ets = df[f'Energy @ {T}']
        ax.plot(Steps,Ets)
    fig.savefig(OUTPUT_DIR / "EnergyVsSteps.png")

def calcAvgEandC(T,Ets):
    avg = np.average(Ets[int(0.3*len(Ets)):])
    avgSq = np.average(Ets[int(0.3*len(Ets)):]**2)
    C = (avgSq - (avg**2))/T
    return avg, C

def plotAvgECvsT():
    avgs = np.zeros_like(Ts)
    Cs = np.zeros_like(Ts)
    for i,T in enumerate(Ts):
        avgs[i],Cs[i] = calcAvgEandC(T,df[f'Energy @ {T}'])

    fig, ax = plt.subplots()
    ax.plot(Ts,avgs)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Average Energy')
    fig.savefig(OUTPUT_DIR / "AverageEnergyVsTemperature.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(Ts,Cs)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Specific Heat')
    fig.savefig(OUTPUT_DIR / "SpecificHeatVsTemperature.png")


plotAvgECvsT()
plotEtsVsSteps()
