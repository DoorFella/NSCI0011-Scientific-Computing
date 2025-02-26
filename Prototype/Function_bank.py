from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

#Morris-Lecar model

def Morris_Lecar(t, x, I_app=300):
    """Function which defines the Morris-Lecar model.

    Args:
        t: Time dummy variable needed for ivp_solve in scipy
        x (2D numpy array): contains the V and w variables. needed for ivp_solve from scipy
        I_app (int, optional): Applied current. Defaults to 300.

    Returns:
        numpy array: numpy array of the Morris-Lecar model.
    """
    
    V,w=x
    
    m_inf = 0.5*(1 + np.tanh((V - v1)/v2))
    w_inf = 0.5*(1 + np.tanh((V - v3)/v4))
    tau = 1/np.cosh((V - v3)/(2*v4))

    
    dxdt = [
        (- g_Ca * m_inf * (V - V_Ca) - g_K * w * ( V - V_K) - g_L *(V - V_L) + I_app)/C,
        
        phi * (w_inf - w)/ tau
    ]
    
    return dxdt


def ML_fsolve(x, I_app=300):
    """Morris-Lecar model, used with the fsolve function, since it doesn't use time as a parameter

    Args:
        x (2D numpy array): contains the V and w variables. needed for fsolve
        I_app (int, optional): Applied current. Defaults to 300.

    Returns:
        numpy array: numpy array of the Morris-Lecar model. To be used with fsolve
    """
    
    V, w = x

    
    m_inf = 0.5*(1 + np.tanh((V - v1)/v2))
    w_inf = 0.5*(1 + np.tanh((V - v3)/v4))
    tau = 1/np.cosh((V - v3)/(2*v4))

    
    dxdt = [
        (- g_Ca * m_inf * (V - V_Ca) - g_K * w * ( V - V_K) - g_L *(V - V_L) + I_app)/C,
        
        phi * (w_inf - w)/ tau
    ]
    
    return dxdt


def Vdotzero(V, I_app=300):
    """Returns the vertical nullcline in the Morris-Lecar model $\dot{V} = 0$

    Args:
        V (numpy array): array of V values
        I_app (int, optional): Applied current. Defaults to 300.

    Returns:
        numpy array: Returns a numpy array of w when $\dot{V} = 0$
    """
    
    m_inf = 0.5*(1 + np.tanh((V - v1)/v2))
    
    tau = 1/np.cosh((V - v3)/(2*v4))

    
    w= ( g_Ca * m_inf * (V - V_Ca) +g_L *(V - V_L) - I_app)/(- g_K * ( V - V_K))
        
    
    return w


def wdotzero(V):
    """Returns the horizontal nullcline of the Morris-Lecar model, i.e. when $\dot{w} = 0$. This coincides with w_inf

    Args:
        V (numpy array): V values

    Returns:
        Numpy array: wdot=0 values
    """
    w_inf = 0.5*(1 + np.tanh((V - v3)/v4))
    return w_inf

def m_inf(V, v1=-20, v2=24):
    """returns the an array of m_inf given an array of V values

    Args:
        V (array): Potential difference
        v1 (int): 
        v2 (int): 

    Returns:
        array: Array of m_inf values
    """
    return 0.5*(1+ np.tanh((V - v1))/v2)

def w_inf(V, v3 = -16, v4 = 11.2):
    """returns the an array of w_inf given an array of V values

    Args:
        V (array): Potential difference
        v3 (int): 
        v4 (float): 

    Returns:
        array: Array of w_inf values
    """
    return 0.5*(1 + np.tanh((V - v3)/v4))

def tau(V, v3 = -16, v4 = 11.2):
    """returns the an array of tau given an array of V values

    Args:
        V (array): Potential difference
        v3 (int): 
        v4 (float): 

    Returns:
        array: Array of tau values
    """
    
    return 1/np.cosh((V - v3)/(2*v4))

#Chay-Keizer model


def I_KCa(V, Cai, g_KCa = 2000, V_K = -75, K_KCa = 5):
    """Returns an array of K(Ca) current given arrays of V - voltage and Calcium concentrations

    Args:
        V (array): Potential difference
        Cai (array): Calcium concentration
        g_KCa (int): 
        V_K (int): 
        K_KCa (int): 

    Returns:
        array: K(Ca) current
    """
    return g_KCa * Cai  * (V - V_K) / (K_KCa + Cai)


def I_Ca(V, g_Ca=1000, V_Ca=25, v1=-20,v2=24):
    """Returns an array of Ca current given array of V - voltage.
    
    Args:
        V (array): Potential difference
        g_Ca (int, optional): Defaults to 1000.
        V_Ca (int, optional): Defaults to 25.
        v1 (int, optional): Defaults to -20.
        v2 (int, optional): Defaults to 24.

    Returns:
        array: Ca current
    """
    return g_Ca * (V - V_Ca) * 0.5*(1+ np.tanh((V - v1))/v2)

def I_K(V, w, g_K = 2700, V_K = -75, ):
    """Returns an array of K current given array of V - voltage and w - fraction of open ion channels.

    Args:
        V (array): Potential difference
        w (array): Proportion of open ion channels over time
        g_K (int, optional): Defaults to 2700.
        V_K (int, optional): Defaults to -75.

    Returns:
        array: K current
    """
    return g_K * ( V - V_K) *w 

def I_L(V, g_L = 150, V_L = -75):
    """Returns an array of K current given array of V - voltage and w - fraction of open ion channels.

    Args:
        V (array): Potential difference
        g_L (int, optional): Defaults to 150.
        V_L (int, optional): Defaults to -75.

    Returns:
        array: L current
    """
    return g_L *(V - V_L)




def chay_Keizer(t, x, f_i = 0.004, vLPM = 0.18):
    """returns the full Chay-Keizer model.

    Args:
        t (): dummy time variable used for ivp_solve
        x (array): 2D array of V, w, and Cai - Voltage, fraction of open channels, calcium concentration

    Returns:
        array: returns RHS of the ODEs
    """
    Cm = 5300 #fF
    g_Ca = 1000 #pS
    V_Ca = 25 #mV
    g_K = 2700 #pS
    V_K = -75 #mV
    I_app = 0
    v1 = -20 #mV
    v2 = 24
    v3 = -16 #mV
    v4 = 11.2 #mV
    phi = 0.035 #/ms
    g_L = 150 #pS
    V_L = -75 #mV
    g_KCa = 2000 #pS
    K_KCa = 5 #micro M
    f = 0.001 
    alpha = 0.0000045 #micro M /(fA * ms)
        
    
    V, w, Cai = x

    
    dxdt = [
        (- I_Ca(V, g_Ca,V_Ca) - I_K(V, w, g_K, V_K) - I_L(V, g_L, V_L) - I_KCa(V, Cai, g_KCa, V_K, K_KCa)) / Cm,
        phi * (0.5*(1 + np.tanh((V - v3)/v4)) - w)/ tau(V, v3, v4),
        f_i *(-alpha* I_Ca(V, g_Ca,V_Ca) - vLPM * Cai)
        
    ]
    return dxdt