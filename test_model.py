from scipy.integrate import solve_ivp
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




I_app = 300 #pA



def f(t, x):
    V,w=x
    
    m_inf = 0.5*(1 + np.tanh((V - v1)/v2))
    w_inf = 0.5*(1 + np.tanh((V - v3)/v4))
    tau = 1/np.cosh((V - v3)/(2*v4))

    
    dxdt = [
        (- g_Ca * m_inf * (V - V_Ca) - g_K * w * ( V - V_K) - g_L *(V - V_L) + I_app)/C,
        
        phi * (w_inf - w)/ tau
    ]
    
    return dxdt

sol=solve_ivp(f, t_span=[0, 200], y0 =[-20, 0])
#print(sol.y)



fig, ax = plt.subplots()



ax.plot(sol.t, sol.y[0, :], label = f"I_app = 300 pA")

plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.legend()

plt.show()

 
I_app = [300, 150, 60, 0]




