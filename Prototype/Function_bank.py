from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from numpy import linalg as LA

#Morris-Lecar model

def m_inf(V, v1, v2):
    """returns the an array of m_inf given an array of V values

    Args:
        V (array): Potential difference
        v1 (int): 
        v2 (int): 

    Returns:
        array: Array of m_inf values
    """
    return 0.5*(1+ np.tanh((V - v1)/v2))

def w_inf(V, v3, v4):
    """returns the an array of w_inf given an array of V values

    Args:
        V (array): Potential difference
        v3 (int): 
        v4 (float): 

    Returns:
        array: Array of w_inf values
    """
    return 0.5*(1 + np.tanh((V - v3)/v4))

def tau(V, v3, v4):
    """returns the an array of tau given an array of V values

    Args:
        V (array): Potential difference
        v3 (int): 
        v4 (float): 

    Returns:
        array: Array of tau values
    """
    
    return 1/np.cosh((V - v3)/(2*v4))



def Morris_Lecar(t, x, I_app=300, C = 20,V_K = - 84,g_K = 8,V_Ca = 120,g_Ca = 4.4,V_L = -60,g_L = 2,v1 = -1.2,v2 = 18,v3 = 2,v4 = 30,phi = 0.04):
    """Function which defines the Morris-Lecar model.

    Args:
        t: Time dummy variable needed for ivp_solve in scipy
        x (2D numpy array): contains the V and w variables. needed for ivp_solve from scipy
        I_app (int, optional): Applied current. Defaults to 300.

    Returns:
        numpy array: numpy array of the Morris-Lecar model.
    """
    
    V,w=x
    

    
    dxdt = [
        (- g_Ca * m_inf(V,v1,v2) * (V - V_Ca) - g_K * w * ( V - V_K) - g_L *(V - V_L) + I_app)/C,
        
        phi * (w_inf(V,v3,v4) - w)/ tau(V,v3,v4)
    ]
    
    return dxdt


def ML_fsolve(x, I_app=300, C = 20,V_K = - 84,g_K = 8,V_Ca = 120,g_Ca = 4.4,V_L = -60,g_L = 2,v1 = -1.2,v2 = 18,v3 = 2,v4 = 30,phi = 0.04):
    """Morris-Lecar model, used with the fsolve function, since it doesn't use time as a parameter

    Args:
        x (2D numpy array): contains the V and w variables. needed for fsolve
        I_app (int, optional): Applied current. Defaults to 300.

    Returns:
        numpy array: numpy array of the Morris-Lecar model. To be used with fsolve
    """
    
    V, w = x

    
    dxdt = [
        (- g_Ca * m_inf(V,v1,v2) * (V - V_Ca) - g_K * w * ( V - V_K) - g_L *(V - V_L) + I_app)/C,
        
        phi * (w_inf(V,v3,v4) - w)/ tau(V,v3,v4)
    ]
    
    return dxdt


def Vdotzero(V, I_app=300, V_K = - 84,g_K = 8,V_Ca = 120,g_Ca = 4.4,V_L = -60,g_L = 2,v1 = -1.2,v2 = 18):
    """Returns the vertical nullcline in the Morris-Lecar model $\dot{V} = 0$

    Args:
        V (numpy array): array of V values
        I_app (int, optional): Applied current. Defaults to 300.

    Returns:
        numpy array: Returns a numpy array of w when $\dot{V} = 0$
    """
    


    
    w= ( g_Ca * m_inf(V, v1, v2) * (V - V_Ca) +g_L *(V - V_L) - I_app)/(- g_K * ( V - V_K))
        
    
    return w


def wdotzero(V, v3=2,v4=30):
    """Returns the horizontal nullcline of the Morris-Lecar model, i.e. when $\dot{w} = 0$. This coincides with w_inf

    Args:
        V (numpy array): V values

    Returns:
        Numpy array: wdot=0 values
    """
    w_inf = 0.5*(1 + np.tanh((V - v3)/v4))
    return w_inf


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
    return g_Ca *m_inf(V,v1,v2)* (V - V_Ca)

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




def chay_Keizer(t, x, f_i = 0.004, vLPM = 0.18, Cm = 5300, g_Ca = 1000, V_Ca = 25, g_K = 2700, V_K = -75, I_app = 0, v1 = -20, v2 = 24, v3 = -16, v4 = 11.2, phi = 0.035, g_L = 150, V_L = -75, g_KCa = 2000, K_KCa = 5, alpha = 0.0000045):
    """returns the full Chay-Keizer model.

    Args:
        t (): dummy time variable used for ivp_solve
        x (array): 2D array of V, w, and Cai - Voltage, fraction of open channels, calcium concentration

    Returns:
        array: returns RHS of the ODEs
    """
    
    V, w, Cai = x
    
    dxdt = [
        (- I_Ca(V,g_Ca,V_Ca,v1,v2) - I_K(V,w,g_K,V_K) - I_L(V,g_L,V_L) - I_KCa(V, Cai, g_KCa, V_K, K_KCa)) / Cm,
        phi * (w_inf(V,v3,v4) - w)/ tau(V,v3,v4),
        f_i *(-alpha* I_Ca(V,g_Ca,V_Ca,v1,v2) - vLPM * Cai)
        
    ]
    return dxdt

#def find_eigvals_2D(callback, length=300):
#    varied = np.linspace(0, 300, length)
#    eigenvalues=np.zeros(length)
#    eigenvectors=np.zeros(length)
#    V = np.zeros(length)
#    w = np.zeros(length)
#
#    for i in range(len(varied)):
#        sol = fsolve(callback, [1,1], args=(varied[i],))
#        V[i], w[i] = sol
#        J=optimize.approx_fprime([V[i], w[i]], callback)
#
#        eigenvalues[i], eigenvectors[i] = LA.eig(J)
#    
#        if np.real(eigenvalues[0])>=0:
#        stability[i] = 1
#        
#        
    