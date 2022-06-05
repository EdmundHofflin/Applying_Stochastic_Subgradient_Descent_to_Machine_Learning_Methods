"""
Edmund Hofflin: u6373930

This file exports a variety of different functions and their derivatives/subdifferentials
"""

# Imports
import numpy as np
from helpers import local_approx

# =========
# Smooth 1D
# =========

def smooth_quadratic(a : float = None, b : float = None, c : float = None):
    """ Generic quadratic function """
    
    # If parameters are None, randomise them
    if not a:
        a = np.random.uniform(0,1)
    if not b:
        b = np.random.uniform(-1,1)
    if not c:
        c = np.random.uniform(-1,1)


    def f(x : np.ndarray):
        return float(a * x[0] ** 2 + b * x[0] + c)
    def df(x : np.ndarray):
        return np.array([float(2 * a * x[0] + b)])
    return f, df

def sixtic(a6 : float = 0, a5 : float = 0, a4 : float = 0, a3 : float = 0, a2 : float = 0, a1 : float = 0, a0 : float = 0):
    """ Generic quadratic function """
    def f(x : np.ndarray):
        return float(a6 * x[0] ** 6 + a5 * x[0] ** 5 + a4 * x[0] ** 4 + a3 * x[0] ** 3 + a2 * x[0] ** 2 + a1 * x[0] + a0)
    def df(x : np.ndarray):
        return np.array([float(6 * a6 * x[0] ** 5 + 5 * a5 * x[0] ** 4 + a4 * x[0] ** 3 + 3 * a3 * x[0] ** 2 + 2 * a2 * x[0] + a1)])
    return f, df

def log_inverse(a : float = 1):
    """ Generic quadratic function """
    def f(x : np.ndarray):
        return float(np.log(x[0]) + a/x[0])
    def df(x : np.ndarray):
        return np.array([(x[0] - a)/(x[0] ** 2)])
    return f, df

# =========
# Smooth nD
# =========

def figure():
    """ Function used for figure """
    def f(x : np.ndarray):
        a = x[0]
        b = x[1]
        return float(0.0025 * (a ** 4 + b ** 2) - 100 * np.exp(- a ** 2 - (b ** 2)))
    def df(x : np.ndarray):
        a = x[0]
        b = x[1]
        return [np.array([float(0.01 * a ** 3 + 200 * a * np.exp(- a ** 2 - (b ** 2))), float(b * (0.005 + 200 * np.exp(- a ** 2 - (b ** 2))))])]
    return f, df

# ============
# NonSmooth 1D
# ============

RABS_CONSTANT = 1e-10

def quadratic(a : float = None, b : float = None, c : float = None):
    """ Generic quadratic function """
    # If parameters are None, randomise them
    if not a:
        a = np.random.uniform(0,1)
    if not b:
        b = np.random.uniform(-10,10)
    if not c:
        c = np.random.uniform(-10,10)
    
    # Parameter dict
    par_dict = dict()
    par_dict["a"] = a
    par_dict["b"] = b
    par_dict["c"] = c
    
    # Function and Clarke Subdifferential
    def f(x : np.ndarray):
        return float(a * (x[0] ** 2) + b * x[0] + c)
    def clarke_f(x : np.ndarray):
        return [np.array([float(a * 2 * x[0] + b)])]
    
    # Output
    return f, clarke_f, par_dict

def relu(rabs : float = RABS_CONSTANT):
    """ Standard ReLU function """

    # Parameter dict
    par_dict = dict()
    par_dict["rabs"] = rabs

    # Function and Clarke Subdifferential
    def f(x : np.ndarray):
        return 0.0 if x[0] < 0 else x[0]
    def clarke_f(x : np.ndarray):
        def df(x : np.ndarray):
            return np.array([0]) if x < 0 else np.array([1.0])
        subgrad = list()
        # If x is near 0, function is not differentiable
        if np.linalg.norm(x, ord=2) < rabs:
            _, _, cvh = local_approx(df, x, rabs)
            subgrad = list(cvh)
        # Otherwise, function is differentiable
        else:
            subgrad = [df(x)]
        return subgrad
    
    # Output
    return f, clarke_f, par_dict

# ============
# NonSmooth nD
# ============

def p_norm(a : float = None, ord : int = 2, rabs : float = RABS_CONSTANT):
    """ Standard p norms """
    # If parameters are None, randomise them
    if not a:
        a = np.random.uniform(0,1)
    
    # Parameter dict
    par_dict = dict()
    par_dict["a"] = a
    par_dict["ord"] = ord
    par_dict["rabs"] = rabs

    # Function and Clarke Subdifferential
    def f(x : np.ndarray):
        return float(a * np.linalg.norm(x, ord=ord))
    if ord == 1:
        def clarke_f(x : np.ndarray):
            def df(x : np.ndarray):
                grad = np.zeros(x.shape[0])
                for i in range(len(grad)):
                    grad[i] = np.sign(x[i])
                return a * grad
            # If x is near 0, function is not differentiable
            if np.linalg.norm(x, ord=2) < rabs:
                _, _, cvh = local_approx(df, x, rabs * 2)
                subgrad = list(cvh)
            # Otherwise, function is differentiable
            else:
                subgrad = [df(x)]
            return subgrad
    elif np.isinf(ord):
        def clarke_f(x : np.ndarray):
            def df(x : np.ndarray):
                grad = np.zeros(x.shape[0])
                idx = 0
                for i in range(len(grad)):
                    if np.abs(x[i]) > np.abs(x[idx]):
                        idx = i
                grad[idx] = np.sign(x[idx]) * 1.0
                return a * grad
            # If another component is close to max(x), then function is not differentiable
            idx_max = 0
            diff_matrix = np.zeros((x.shape[0], x.shape[0]))
            for i in range(x.shape[0]):
                for j in range(i, x.shape[0]):
                    diff_matrix[i,j] = np.abs(x[i] - x[j])
                    diff_matrix[j,i] = np.abs(x[i] - x[j])
                if np.abs(x[i]) > np.abs(x[idx_max]):
                    idx_max = i
            close_components = list()
            for i in range(x.shape[0]):
                if diff_matrix[idx_max, i] < rabs:
                    close_components.append(i)
            if len(close_components) > 1:
                cvh = np.zeros((100 * x.shape[0], x.shape[0]))
                for i in range(100 * x.shape[0]):
                    r = np.random.uniform(0, 1, size=len(close_components))
                    r = r / np.sum(r)
                    for j in range(len(close_components)):
                        cvh[i,close_components[j]] = np.sign(x[close_components[j]]) * r[j]
                subgrad = list(cvh)
                if np.max(x[close_components]) < 2 * rabs:
                    subgrad.append(np.zeros(shape=(x.shape[0])))
            # Otherwise the function is differentiable
            else:
                subgrad = [df(x)]
            return subgrad
    else:
        def clarke_f(x : np.ndarray):
            def df(x : np.ndarray):
                grad = np.zeros(x.shape[0])
                for i in range(len(grad)):
                    grad[i] = x[i] * np.abs(x[i]) ** (ord - 2)
                grad = (a / (np.linalg.norm(x, ord=ord) ** (ord - 1))) * grad
                return grad
            # If x is near 0, function is not differentiable
            if np.linalg.norm(x, ord=2) < rabs:
                _, _, cvh = local_approx(df, x, rabs * 2)
                subgrad = list(cvh)
            # Otherwise, function is differentiable
            else:
                subgrad = [df(x)]
            return subgrad
        
    # Output
    return f, clarke_f, par_dict

def piecewise(rabs=RABS_CONSTANT):
    """ Function used for figure """
    
    # Parameter dict
    par_dict = dict()
    par_dict["rabs"] = rabs

    # Function and Clarke Subdifferential
    def f(x : np.ndarray):
        if np.linalg.norm(x, ord=np.inf) > 1:
            return 1 + 2 * np.sum(np.log(np.max(np.abs(x))))
        else:
            return p_norm(a=1, ord=np.inf, rabs=rabs)[0](x)
    def clarke_f(x : np.ndarray):
        def df(x : np.ndarray):
            if np.linalg.norm(x, ord=np.inf) > 1:
                if np.abs(x[0]) > np.abs(x[1]):
                    return np.array([2/x[0], 0])
                else:
                    return np.array([0, 2/x[1]])
            else:
                grad = np.zeros(x.shape[0])
                idx = 0
                for i in range(len(grad)):
                    if np.abs(x[i]) > np.abs(x[idx]):
                        idx = i
                grad[idx] = np.sign(x[idx]) * 1.0
                return grad
        if np.linalg.norm(x, ord=np.inf) > 1 + rabs:
            return [df(x)]
        elif np.linalg.norm(x, ord=np.inf) > 1 - rabs:
            _, _, cvh = local_approx(df, x, rabs * 2)
            subgrad = list(cvh)
        else:
            return p_norm(a=1, ord=np.inf, rabs=rabs)[1](x)
    
    # Output
    return f, clarke_f, par_dict

def flat_waves(ord : int = 2.0, a : float = None, b : float = None, rabs=RABS_CONSTANT):
    """ Function used for waves """
    # If parameters are None, randomise them
    if not a:
        a = np.random.uniform(1,2)
    if not b:
        b = np.random.uniform(1,2)
    
    # Parameter dict
    par_dict = dict()
    par_dict["ord"] = ord
    par_dict["a"] = a
    par_dict["b"] = b
    par_dict["rabs"] = rabs

    # Function and Clarke Subdifferential
    f_norm, clarke_f_norm, _ = p_norm(ord=ord,rabs=rabs)
    def f(x : np.ndarray):
        return a * np.cos(f_norm(x)/b)
    def clarke_f(x : np.ndarray):
        factor = -(a/b) * np.sin(f_norm(x)/b)
        return list(factor * np.array(clarke_f_norm(x)))
    
    # Output
    return f, clarke_f, par_dict

def waves(ord : int = 1, a : float = None, b : float = None, c : float = None, rabs=RABS_CONSTANT):
    """ Function used for waves """

    # If parameters are None, randomise them
    if not a:
        a = np.random.uniform(1,2)
    if not b:
        b = np.random.uniform(1,2)
    if not c:
        c = np.random.uniform(0,2)
    
    # Parameter dict
    par_dict = dict()
    par_dict["ord"] = ord
    par_dict["a"] = a
    par_dict["b"] = b
    par_dict["c"] = c
    par_dict["rabs"] = rabs

    # p-Norm for reference
    f_norm, clarke_f_norm, _ = p_norm(ord=ord, a=1.0, rabs=rabs)

    # Function and Clarke Subdifferential
    def f(x : np.ndarray):
        return c * f_norm(x) - a * np.cos(f_norm(x)/b)
    def clarke_f(x : np.ndarray):
        factor = (a/b) * np.sin(f_norm(x)/b)
        return list((c + factor) * np.array(clarke_f_norm(x)))
    
    # Output
    return f, clarke_f, par_dict

def spiral(a : float = None, b : float = None, c : float = None, rabs=RABS_CONSTANT):
    """ Spiral! """
    # If parameters are None, randomise them
    if not a:
        a = np.random.uniform(0,1.5)
    if not b:
        b = np.random.uniform(5,7.5)
    if not c:
        c = np.random.uniform(0.5,3)
    
    # Parameter dict
    par_dict = dict()
    par_dict["a"] = a
    par_dict["b"] = b
    par_dict["c"] = c
    par_dict["rabs"] = rabs
    
    # p-Norm
    f_norm, clarke_f_norm, _ = p_norm(ord=1.0, a=1.0, rabs=rabs)

    # Helper functions
    def f_arctan2(x : np.ndarray):
        return np.arctan2(x[1], x[0])
    def clarke_f_arctan2(x : np.ndarray):
        return  (1 / (x[0] ** 2 + x[1] ** 2)) * np.array(-x[1], x[0])

    # Function and Clarke Subdifferential
    def f(x : np.ndarray):
        return c * np.cos(a * f_norm(x) + b * f_arctan2(x)) + f_norm(x)
    def clarke_f(x : np.ndarray):
        if np.linalg.norm(x, ord=2) < rabs / 2:
            return [np.array([0, 0])]
        return list(map(lambda subgrad: subgrad - c * (b * clarke_f_arctan2(x) + a * subgrad) * np.sin(b * f_arctan2(x) + a * f_norm(x)), clarke_f_norm(x)))
    
    # Output
    return f, clarke_f, par_dict
