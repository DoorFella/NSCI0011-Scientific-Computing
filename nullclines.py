from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt



###

C = 20 #micro F/cm^2
V_K = - 84 #mV
g_K = 8 #mS/cm^2
V_Ca = 120 #mV
g_Ca = 4.4 #mS/cm^2
V_L = -60 #mV
g_L = 2 #mS/cm^2
v1 = -1.2 #mV
v2 = 18 #mV
v3 = 2 #mV
v4 = 30 #mV
phi = 0.04 #per ms




def Vdotzero(V, I_app=300):
    
    
    m_inf = 0.5*(1 + np.tanh((V - v1)/v2))
    
    tau = 1/np.cosh((V - v3)/(2*v4))

    
    w= ( g_Ca * m_inf * (V - V_Ca) +g_L *(V - V_L) - I_app)/(- g_K * ( V - V_K))
        
    
    return w


def wdotzero(V):
    w_inf = 0.5*(1 + np.tanh((V - v3)/v4))
    return w_inf


V = np.linspace(-70, 70, 100)

wdotzero = wdotzero(V)

I_app = [300, 150, 60]


fig, ax = plt.subplots()

for i in I_app:
    Vdot= Vdotzero(V, I_app=i)
    ax.plot(V, Vdot, label= f"I_app = {i}pA")



ax.plot(V, wdotzero)
ax.legend()
plt.ylim(top = 1)
plt.xlabel("V (mV)")
plt.ylabel("w")


plt.savefig("nullclines.png")


I_app = np.linspace(0, 300, 100)


