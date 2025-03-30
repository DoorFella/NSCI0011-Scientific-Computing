from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from numpy import linalg as LA



param_dict = {
    "Cm": 5300,
    "g_Ca":1000,
    "V_Ca":25,
    "g_K":2700,
    "V_K":-75,
    "v1":-20,
    "v2":24,
    "v3":-16,
    "v4":11.2,
    "phi":0.035,
    "g_L":150,
    "V_L":-75,
    "g_KCa":2000,
    "K_KCa":5,
    "f":0.001,
    "alpha":0.0000045,
    "f_i":0.004,
    "vLPM":0.18,
    "Cai":0.2,
    "w":0.1
}


def m_inf(V, param_dict):
    """returns the an array of m_inf given an array of V values

    Args:
        V (array): Potential difference
        v1 (int): 
        v2 (int): 

    Returns:
        array: Array of m_inf values
    """
    v1 = param_dict["v1"]
    v2 = param_dict["v2"]
    return 0.5*(1+ np.tanh((V - v1)/v2))

def w_inf(V, param_dict):
    """returns the an array of w_inf given an array of V values

    Args:
        V (array): Potential difference
        v3 (int): 
        v4 (float): 

    Returns:
        array: Array of w_inf values
    """
    v3 = param_dict["v3"]
    v4 = param_dict["v4"]
    return 0.5*(1 + np.tanh((V - v3)/v4))


def tau(V, param_dict):
    """returns the an array of tau given an array of V values

    Args:
        V (array): Potential difference
        v3 (int): 
        v4 (float): 

    Returns:
        array: Array of tau values
    """
    v3 = param_dict["v3"]
    v4 = param_dict["v4"]
    
    return 1/np.cosh((V - v3)/(2*v4))


def I_KCa(V, Cai, param_dict):
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
    g_KCa = param_dict["g_KCa"]
    V_K = param_dict["V_K"]
    K_KCa = param_dict["K_KCa"]
    return g_KCa * Cai  * (V - V_K) / (K_KCa + Cai)


def I_Ca(V, param_dict):
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
    g_Ca = param_dict["g_Ca"]
    V_Ca = param_dict["V_Ca"]
    v1 = param_dict["v1"]
    v2 = param_dict["v2"]
    
    return g_Ca *m_inf(V,param_dict)* (V - V_Ca)

def I_K(V, w, param_dict):
    """Returns an array of K current given array of V - voltage and w - fraction of open ion channels.

    Args:
        V (array): Potential difference
        w (array): Proportion of open ion channels over time
        g_K (int, optional): Defaults to 2700.
        V_K (int, optional): Defaults to -75.

    Returns:
        array: K current
    """
    g_K = param_dict["g_K"]
    V_K = param_dict["V_K"]
    return g_K * ( V - V_K) *w 

def I_L(V, param_dict):
    """Returns an array of K current given array of V - voltage and w - fraction of open ion channels.

    Args:
        V (array): Potential difference
        param_dict (dict): Dictionary of parameters

    Returns:
        array: L current
    """
    g_L = param_dict["g_L"]
    V_L = param_dict["V_L"]
    return g_L *(V - V_L)

def dVdt(V, w, Cai, param_dict):

    """Returns the RHS V equation of the Chay-Keizer model.
    
    Args:
        V (array): Potential difference
        w (array): Proportion of open ion channels over time
        Cai (array): Calcium concentration
        param_dict (dict): Dictionary of parameters 

                
    Returns:
        array: dVdt of the Morris-Lecar model
    """
    Cm = param_dict["Cm"]
    return (- I_Ca(V,param_dict) - I_K(V,w,param_dict) - I_L(V,param_dict) - I_KCa(V, Cai, param_dict)) / Cm

def dwdt(V, w, param_dict):
    """Returns the RHS w equation of the Chay-Keizer model.
    
    Args:
        V (array): Potential difference
        w (array): Proportion of open ion channels over time
        param_dict (dict): Dictionary of parameters
        
    Returns:
        array: dwdt of the Chay-Keizer model
    """
    phi = param_dict["phi"]
    return phi * (w_inf(V,param_dict) - w)/ tau(V,param_dict)

def dCaidt(V, Cai, param_dict):
    """Returns the RHS Cai equation of the Chay-Keizer model.
    
    Args:
        V (array): Potential difference
        Cai (array): Calcium concentration
        param_dict (dict): Dictionary of parameters

    Returns:
        array: dCaidt of the Chay-Keizer model
    """
    f_i = param_dict["f_i"]
    vLPM = param_dict["vLPM"]
    alpha = param_dict["alpha"]
    return f_i *(-alpha* I_Ca(V,param_dict) - vLPM * Cai)

def chay_Keizer(t, x, param_dict):
    """returns the full Chay-Keizer model.

    Args:
        t (): dummy time variable used for ivp_solve
        x (array): 2D array of V, w, and Cai - Voltage, fraction of open channels, calcium concentration

    Returns:
        array: returns RHS of the ODEs
    """
    
    V, w, Cai = x
    
    dxdt = [
        dVdt(V, w, Cai, param_dict),
        dwdt(V, w, param_dict),
        dCaidt(V, Cai, param_dict)
    ]
    return dxdt

def CK_fsolve(x, param_dict):
    """returns the full Chay-Keizer model.

    Args:
        x (array): 2D array of V, w, and Cai - Voltage, fraction of open channels, calcium concentration

    Returns:
        array: returns RHS of the ODEs
    """
    
    V, w, Cai = x
    
    dxdt = [
        dVdt(V, w, Cai, param_dict),
        dwdt(V, w, param_dict),
        dCaidt(V, Cai, param_dict)
    ]
    return dxdt
    

def CK_wV_phase(t, x, param_dict):
    """Returns the RHS of the V-w phase plane of the Chay-Keizer model.
    
    Args:
        V (array): Potential difference
        w (array): Proportion of open ion channels over time
        Cai (array): Calcium concentration
        param_dict (dict): Dictionary of parameters

    Returns:
        array: RHS of the V-w phase plane of the Chay-Keizer model
    """
    V, w = x
    Cai = param_dict["Cai"]
    
    return [dVdt(V, w, Cai, param_dict), dwdt(V, w, param_dict)]

def Vdotzero_CK(V, Cai, param_dict):
    g_K = param_dict["g_K"]
    V_K = param_dict["V_K"]
    
    w = (-I_Ca(V,param_dict) - I_L(V,param_dict) - I_KCa(V, Cai, param_dict)) / ( g_K * ( V - V_K))
    return w

def wdotzero(V, param_dict):
    """Returns the horizontal nullcline of the Morris-Lecar model, i.e. when $\dot{w} = 0$. This coincides with w_inf

    Args:
        V (numpy array): V values

    Returns:
        Numpy array: wdot=0 values
    """
    v3 = param_dict["v3"]
    v4 = param_dict["v4"]
    
    w_inf = 0.5*(1 + np.tanh((V - v3)/v4))
    return w_inf

def Bifurcation_VCai(x, param_dict, V):
    """Returns the RHS of the V-Cai bifurcation of the Chay-Keizer model.
    
    Args:
        x (array): 2D array of V, and Cai - Voltage, calcium concentration
        param_dict (dict): Dictionary of parameters

    Returns:
        array: RHS of the V-Cai bifurcation of the Chay-Keizer model
    """
    w, Cai = x
    return [dVdt(V,w,Cai, param_dict), dwdt(V, w, param_dict)]

def CK_bifurcation(x, param_dict, Cai):
    
    V, w = x
    return [dVdt(V, w, Cai, param_dict), dwdt(V, w, param_dict)]

def CK_fsolve_Cai(x, param_dict, Cai):
    V,w = x
    dxdt = [
        dVdt(V, w, Cai, param_dict),
        dwdt(V, w, param_dict)
    ]
    return dxdt
