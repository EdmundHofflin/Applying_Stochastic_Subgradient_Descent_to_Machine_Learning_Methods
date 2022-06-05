"""
Edmund Hofflin: u6373930

This file exports a set of helpful functions used through out the optimisation framework and tests.
"""

# Imports
from typing import Callable
import numpy as np
import scipy as sp
import scipy.spatial


# Function for calculating relative error
def relative_error(true : float, approx : float):
    """ Returns the relative error of an approximation of a true value.
    
    Inputs:
        - true (float): the underlying true value
        - approx (float): the approximation of the true value
    Outputs:
        - relative_error (float): the relative error
    """

    # ========================
    # Validate / Unpack Inputs
    # ========================

    # true
    if not (type(approx) == float or type(approx) == np.float_):
        raise ValueError("Input true = {} has type {} when it should have type float".format(true, type(true)))
    
    # approx
    if not (type(approx) == float or type(approx) == np.float_):
        raise ValueError("Input approx = {} has type {} when it should have type float".format(approx, type(approx)))

    # ==================
    # Function Operation
    # ==================

    # Calculate relative_error
    relative_error = np.abs(true - approx)/np.abs(true)

    # Return relative_error
    return relative_error


# Function for calculating absolute error
def absolute_error(true : float, approx : float):
    """ Returns the absolute error of an approximation of a true value.
    
    Inputs:
        - true (float): the underlying true value
        - approx (float): the approximation of the true value
    Outputs:
        - absolute_error (float): the absolute error
    """

    # ========================
    # Validate / Unpack Inputs
    # ========================

    # true
    if not (type(approx) == float or type(approx) == np.float_):
        raise ValueError("Input true = {} has type {} when it should have type float".format(true, type(true)))
    
    # approx
    if not (type(approx) == float or type(approx) == np.float_):
        raise ValueError("Input approx = {} has type {} when it should have type float".format(approx, type(approx)))

    # ==================
    # Function Operation
    # ==================

    # Calculate relative_error
    absolute_error = np.abs(true - approx)

    # Return relative_error
    return absolute_error


# Function for uniformly randomly selecting points around a hypersphere
def hypersphere(dim : int, radius : float, n : int):
    """ Returns n points that have been uniformly selected from a dim-dimensional unit hyperesphere 
    
    Inputs:
        - dim (int): the dimension of the unit hypersphere.
        - radius (float): radius of the hypersphere
        - n (int): the number of points to generate.
    Outputs:
        - points (list(np.ndarray)): the generated points
    """

    # ========================
    # Validate / Unpack Inputs
    # ========================

    # radius
    if not type(radius) == float:
        raise ValueError("Input radius = {} has type {} when it should have type float".format(radius, type(radius)))
    if not radius > 0:
        raise ValueError("Input radius = {} is non-positive when it should be positive".format(radius))

    # dim
    if not type(dim) == int:
        raise ValueError("Input dim = {} has type {} when it should have type int".format(dim, type(dim)))
    if not dim > 0:
        raise ValueError("Input dim = {} is non-positive when it should be positive".format(dim))
    
    # n
    if not type(n) == int:
        raise ValueError("Input n = {} has type {} when it should have type int".format(n, type(n)))
    if not n > 0:
        raise ValueError("Input n = {} is non-positive when it should be positive".format(n))
    
    # ==================
    # Function Operation
    # ==================

    points = list()
    for _ in range(n):
        # Avoid errors, e.g. divide by 0
        exception = True
        while exception:
            try:
                # Muller Normalised Gaussian Method
                x = np.random.normal(0, 1, [dim])
                unit_x = radius * x / np.sum(x ** 2) ** (1/2)
                points.append(unit_x)
                exception = False
            except Exception:
                print(Exception + ' Will discard point and try again.')
    return np.array(points)



# Function for calculating the convex hull from a set of numpy arrays
def convex_hull(points : np.ndarray, n : int, qhull_options=None):
    """ Returns an approximate convex hull of a list of numpy arrays, populated by a given number of randomly chosen points within the hull.
    
    Inputs:
        - points (np.ndarray): matrix where each row is a point to calculate the convex hull over
        - n (int): the number of points to populate the convex hull with
        - qhull_options (str):
    Outputs:
        - convex_hull (np.ndarray): the 2D numpy array that represents an approximate convex hull of the points
    """

    # ========================
    # Validate / Unpack Inputs
    # ========================

    # points
    if not type(points) == np.ndarray:
        raise ValueError("Input points = {} has type {} when it should have type np.ndarray".format(points, type(points)))
    if not len(points.shape) == 2:
        raise ValueError("Input points = {} has {} axes when it should only have two 2".format(points, len(points.shape)))
    
    # n
    if not type(n) == int:
        raise ValueError("Input n = {} has type {} when it should have type int".format(n, type(n)))
    if not n > 0:
        raise ValueError("Input n = {} is non-positive when it should be positive".format(n))

    # ==================
    # Function Operation
    # ==================

    # Calculate convexhull region and vertices
    if qhull_options:
        cv = sp.spatial.ConvexHull(points)
    else:
        cv = sp.spatial.ConvexHull(points, qhull_options=qhull_options)
    vertices = np.array(points)[cv.vertices]

    # Calculate max and min values for each dimension
    max_vals = np.max(vertices,axis=0)
    min_vals = np.min(vertices,axis=0)

    # Generate points
    output = list()
    for i in range(n):
        # Avoid errors
        exception = True
        while exception:
            try:
                # Generate points
                pt = np.zeros(len(max_vals))
                for i in range(len(max_vals)):
                    pt[i] = np.random.uniform(min_vals[i], max_vals[i])
                # Calculate inclusion
                inclusion = np.all(cv.equations[:,0:-1] @ pt + cv.equations[:,-1] <= 0)
                # Check inclusion
                if inclusion:
                    # Add point
                    output.append(pt)
                    exception = False
            except Exception:
                print(Exception + ' Will discard point and try again.')
    if np.all(cv.equations[:,0:-1] @ np.zeros(len(max_vals)) + cv.equations[:,-1] <= 0):
        output.append(np.zeros(len(max_vals))) 

    return output



def local_approx(f : Callable, x : np.ndarray, radius : float = 1.0, sphere_n : int = None, convex_n : int = None, qhull_options=None):
    """ Higher order function that will produce an approximate set of f(x) by computing the convex hull of f(y) with y in hypersphere of some radius r around x.

    Inputs:
        - f (function): Function to approximate.
        - x (np.ndarray): Input to approximate.
        - radius (float, optional): Radius of hypersphere. Default = 1.0.
        - sphere_n (int, optional): Number of values to sample from the hypersphere. Default = 10 ** dim.
        - convex_n (int, optional): Number of points to sample from the convex hull. Default = 100 ** dim
    Outputs:
        - sphere (): TODO
        - values (): TODO
        - cvh (np.ndarray): the 2D numpy array that represents an approximate convex hull of the points
    """

    # ========================
    # Validate / Unpack Inputs
    # ========================

    # f
    if not callable(f):
        raise ValueError("Input f = {} has type {} when it should be a function from np.ndarray to np.ndarray.".format(f, type(f)))
    
    # x
    if not type(x) == np.ndarray:
        raise ValueError("Input x = {} has type {} when it should have type np.ndarray".format(x, type(x)))
    if not len(x.shape) == 1:
        raise ValueError("Input x = {} has {} axes when it should only have 1".format(x, len(x.shape)))
    
    # f and x compatibility
    try:
        val = f(x)
    except:
        raise ValueError("Input f = {} and input x = {} are not compatible, i.e. f(x) is undefined".format(f, x))
    if not type(val) == np.ndarray:
            raise ValueError("Input f = {} returns values of type {} when it should return a np.ndarray".format(f, type(val)))
    
    # radius
    if not type(radius) == float:
        raise ValueError("Input radius = {} has type {} when it should have type float".format(radius, type(radius)))
    if not radius > 0:
        raise ValueError("Input radius = {} is non-positive when it should be positive".format(radius))

    # sphere_n
    if not sphere_n:
        sphere_n = 10 ** x.shape[0]
    else:
        if not type(sphere_n) == int:
            raise ValueError("Input sphere_n = {} has type {} when it should have type int".format(sphere_n, type(sphere_n)))
        if not sphere_n > 0:
            raise ValueError("Input sphere_n = {} is non-positive when it should be positive".format(sphere_n))
    
    # convex_n
    if not convex_n:
        convex_n = 10 ** x.shape[0]
    else:
        if not type(convex_n) == int:
            raise ValueError("Input convex_n = {} has type {} when it should have type int".format(convex_n, type(convex_n)))
        if not convex_n > 0:
            raise ValueError("Input convex_n = {} is non-positive when it should be positive".format(convex_n))

    # ==================
    # Function Operation
    # ==================

    # Check if f(x) is 1D
    if len(f(x).shape) == 1 and f(x).shape[0] == 1:
        sphere = np.array([np.array([-radius]), np.array([radius])]) + x
        values = np.array([f(sphere[0]), f(sphere[1])])
        cvh    = np.linspace(values[0], values[1], 101)
    # Otherwise use nD method
    else:
        # Generate hypersphere and translate to x
        sphere = hypersphere(dim=x.shape[0], radius=radius, n=sphere_n) + x

        # Apply f to hypershere
        values = np.apply_along_axis(f, 1, sphere)
        
        # Approximate convex hull
        cvh = convex_hull(values, n=convex_n, qhull_options=qhull_options)

    # Return
    return sphere, values, cvh