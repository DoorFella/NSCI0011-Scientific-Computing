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




#I_app = 300 #pA



def Morris_Lecar(t, x, I_app=300):
    V,w=x
    
    m_inf = 0.5*(1 + np.tanh((V - v1)/v2))
    w_inf = 0.5*(1 + np.tanh((V - v3)/v4))
    tau = 1/np.cosh((V - v3)/(2*v4))

    
    dxdt = [
        (- g_Ca * m_inf * (V - V_Ca) - g_K * w * ( V - V_K) - g_L *(V - V_L) + I_app)/C,
        
        phi * (w_inf - w)/ tau
    ]
    
    return dxdt



fig, ax = plt.subplots()

I_app = [300, 150, 60, 0]

for i in I_app:
    sol=solve_ivp(f, t_span=[0, 2000], y0 =[40, 0.1], args=(i,), max_step = 0.05)
    ax.plot(sol.y[0,:], sol.y[1, :], label = f"I_app = {i} pA")

plt.xlabel("V (mV)")
plt.ylabel("w")
plt.legend()

plt.savefig("V vs w version7.png")



#making the phase portrait
I_app =150
V = np.arange(-75, 75, 5)
w = np.arange(0, 1, 0.01)

VV, ww = np.meshgrid(V, w)

m_inf = 0.5*(1 + np.tanh((VV - v1)/v2))
w_inf = 0.5*(1 + np.tanh((VV - v3)/v4))
tau = 1/np.cosh((VV - v3)/(2*v4))

dVdt = (- g_Ca * m_inf * (VV - V_Ca) - g_K * ww * ( VV - V_K) - g_L *(VV - V_L) + I_app)/C
dwdt = phi * (w_inf - ww)/ tau

fig, ax = plt.subplots()


ax.streamplot(V, w, dVdt, dwdt)


plt.xlabel("V (mV)")
plt.ylabel("w")
#plt.legend()

plt.savefig("phase portrait.png")


 




