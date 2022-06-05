"""
Edmund Hofflin: u6373930

This file exports a set of optimisation algorithms: stocastic gradient descent, and stochastic subgradient descent
"""

# Imports
from re import sub
import numpy as np
import typing as tp
from tqdm import trange

# Module imports
from helpers import relative_error, absolute_error



# ===========================
# Stochastic Gradient Descent
# ===========================

def sgd(f : tp.Callable[[np.ndarray], float], df : tp.Callable[[np.ndarray], np.ndarray], grad_check_bound : tp.Optional[tp.Union[int, tp.Tuple[int, float]]], step_bound : tp.Optional[int], initial_point : np.ndarray, step_sequence : tp.Union[list, tp.Callable[[int], float]]):
    """ This function completes stochastic gradient descent.
    
    Inputs:
        - f (function: np.ndarray -> float): The objective function being minimised.
        - df (function: np.ndarray -> np.ndarray): The gradient function of f.
        - grad_check_bound (int or (int, float) None): An integer that specifies how many consecutive steps must have a gradient near 0 for termination of the algorithm. If a float is also passed, then this is the absolute error tolerance allowed for a gradient to be considered 0. If None, then grad_check_bound is set to 1 and the absolute error is set to 1e-10.
        - step_bound (int or None): The maximum number of steps that the algorithm will run for. If None, then the algorithm will run until the error bound is reached.
        - initial_point (np.ndarray): The initial point to start the algorithm.
        - step_sequence ([float] or function: int -> float): Either a list of floats with length step_bound or a function that returns a step size for all integers 0 to step_bound.
    Output:
        - iterate_points (np.ndarray): List of iterates
        - iterate_values (float): List of function values at the iterates.
        - iterate_subgrads (np.ndarray): List of subgradient sets at the iterates.
        - step (int): Number of steps. 
    """

    # ========================
    # Validate / Unpack Inputs
    # ========================
    
    # f
    if not callable(f):
        raise ValueError("Input f = {} has type {} when it should have type tp.Callable[[np.ndarray], float]".format(f, type(f)))

    # df
    if not callable(df):
        raise ValueError("Input df = {} has type {} when it should have type tp.Callable[[np.ndarray], np.ndarray]".format(df, type(df)))

    # grad_check_bound
    if grad_check_bound:
        if type(grad_check_bound) == int:
            grad_check_abs_tol = 1e-10
        elif type(grad_check_bound) == tuple and type(grad_check_bound[0]) == int and type(grad_check_bound[1]) == float:
            grad_check_abs_tol = grad_check_bound[1]
            grad_check_bound   = grad_check_bound[0]
        else:
            raise ValueError("Input grad_check_bound = {} has type {} when it should have type int or (int, float) or None".format(grad_check_bound, type(grad_check_bound)))
        if not grad_check_bound > 0:
            raise ValueError("Input grad_check_bound = {} is non-positive when it should be".formamt(grad_check_bound))
        if not grad_check_abs_tol > 0:
            raise ValueError("Input grad_check_bound = ({}, {}) has a non-positive second argument, when both arguments should be positive".format(grad_check_bound, grad_check_abs_tol))
    else:
        grad_check_bound = 1
        grad_check_abs_tol = 1e-10
    
    # step_bound
    if step_bound:
        if not type(step_bound) == int:
            raise ValueError("Input step_bound = {} has type {} when it should have type int".format(step_bound, type(step_bound)))
    else:
        step_bound = np.inf
    
    # initial_point
    if not type(initial_point) == np.ndarray:
        raise ValueError("Input initial_point = {} has type {} when it should have type np.ndarray".format(initial_point, type(initial_point)))
    
    # step_sequence
    if type(step_sequence) == list and type(step_sequence[0]) == float:
        if not len(step_sequence) == step_bound:
            raise ValueError("Input step_sequence = {} has length {} when it should have length equal to other input step_bound = {}. Alteratively, step_sequence can be defined by a function with type tp.Callable[[int], float]".format(step_sequence, len(step_sequence), step_bound))
        else:
            step_func = lambda x: step_sequence[x]
    elif callable(step_sequence):
        step_func = lambda x: step_sequence(x)
    else:
        raise ValueError("Input step_sequence = {} has type {} when it should have type list[float] or tp.Callable[[int], float]".format(step_sequence, type(step_sequence)))
    
    # f and initial_point compatibility
    try:
        val = f(initial_point)
    except:
        raise ValueError("Input f = {} and input initial_point = {} are not compatible, i.e. f(initial_point) is undefined".format(f, initial_point))
    if not type(val) == float:
            raise ValueError("Input f = {} returns values of type {} when it should return a float".format(f, type(val)))
    
    # df and initial_point compatibility
    try:
        grad = df(initial_point)
    except:
        raise ValueError("Input df = {} and input initial_point = {} are not compatible, i.e. df(initial_point) is undefined".format(df, initial_point))
    if not type(grad) == np.ndarray:
        raise ValueError("Input df = {} returns values of type {} when it should return a np.ndarray".format(df, type(grad)))
    if not initial_point.shape == grad.shape:
        raise ValueError("Input df = {} returns values of shape {} when it should return a value with the same shape as its input = {}".format(df, grad.shape, initial_point.shape))


    # ==================
    # Function Operation
    # ==================
    
    # Initialise loop variables
    iterates_points = [initial_point]
    iterates_values = [f(initial_point)]
    iterates_grads  = [df(initial_point)]
    check_count = 0

    # Check
    print("Initial Point: {}".format(iterates_points))
    print("Initial Value: {}".format(iterates_values))
    print("Initial Gradient: {}".format(iterates_grads))

    # Loop while stopping conditions aren't met
    if step_bound == np.inf:
        step_count = 0
        while check_count <= grad_check_bound:
            # Check step size
            if step_func(step_count) == 0:
                print("Step size reached 0. Algorithm halted.")
                check_count = np.inf
                break

            # Print update if not halting
            if step_count != 0:
                print("Step {}:".format(step_count))
                print("Current Point: {}".format(iterates_points[-1]))
                print("Current Value: {}".format(iterates_values[-1]))
                print("Current Gradient: {}".format(iterates_grads[-1]))
            
            # Track progress in 10000 step increments
            base_count = step_count
            for internal_step_count in trange(10000, desc="Steps [{},{})".format(base_count,base_count+10000), leave=False, position=2):
                # Calculate new point, value, and gradient
                x_k_point = iterates_points[-1] - step_func(step_count+internal_step_count) * iterates_grads[-1]
                x_k_value = f(x_k_point)
                x_k_grad  = df(x_k_point)

                # Update lists
                iterates_points.append(x_k_point)
                iterates_values.append(x_k_value)
                iterates_grads.append(x_k_grad)

                # Update stopping conditions
                if np.linalg.norm(iterates_grads[-1], ord=2) < grad_check_abs_tol:
                    check_count += 1
                else:
                    check_count = 0
                
                # Check stopping conditions
                if check_count > grad_check_bound:
                    break
            
            # Count final step
            step_count += 1 + internal_step_count
    else:
        # Loop while stopping conditions aren't met
        for step_count in trange(step_bound, desc="Steps", leave=False, position=2):
            # Check step size
            if step_func(step_count) == 0:
                print("Step size reached 0. Algorithm halted.")
                check_count = np.inf
                break

            # Calculate new point, value, and gradient
            x_k_point = iterates_points[-1] - step_func(step_count) * iterates_grads[-1]
            x_k_value = f(x_k_point)
            x_k_grad  = df(x_k_point)

            # Update lists
            iterates_points.append(x_k_point)
            iterates_values.append(x_k_value)
            iterates_grads.append(x_k_grad)

            # Update stopping conditions
            if np.linalg.norm(iterates_grads[-1], ord=2) < grad_check_abs_tol:
                check_count += 1
            else:
                check_count = 0
            
            # Check stopping conditions
            if check_count > grad_check_bound:
                break
        
        # Count final step
        step_count += 1

    # Return final results 
    return iterates_points, iterates_values, iterates_grads, step_count



# ==============================
# Stochastic Subgradient Descent
# ==============================

def ssd(f : tp.Callable[[np.ndarray], float], clarke_f, subgrad_check_bound : tp.Optional[tp.Union[int, tp.Tuple[int, float]]], step_bound : tp.Optional[int], initial_point : np.ndarray, step_sequence : tp.Union[list, tp.Callable[[int], float]], heuristic, printing : bool = False):
    """ This function completes stochastic subgradient descent.
    
    Inputs:
        - f (function: np.ndarray -> float): The objective function being minimised.
        - clarke_f (function: np.ndarray -> set(np.ndarray)): The Clarke subgradient function of f.
        - subgrad_check_bound (int or (int, float) None): An integer that specifies how many consecutive steps must have 0 in the subgradient for termination of the algorithm. If a float is also passed, then this is the absolute error tolerance allowed for a subdifferential to be considered 0. If None, then subgrad_check_bound is set to 1 and the absolute error is set to 1e-10.
        - step_bound (int or None): The maximum number of steps that the algorithm will run for. If None, then the algorithm will run until the error bound is reached.
        - initial_point (np.ndarray): The initial point to start the algorithm.
        - step_sequence ([float] or function: int -> float): Either a list of floats with length step_bound or a function that returns a step size for all integers 0 to step_bound.
        - heuristic (None or function : list[np.ndarray] -> np.ndarray): A heuristic that selects a Clarke subdifferential from the Clarke subgradient. If None, random selection is used.
    Output:
        - iterate_points (np.ndarray): List of iterates
        - iterate_values (float): List of function values at the iterates.
        - iterate_subgrads (np.ndarray): List of subgradient sets at the iterates.
        - iterate_subdiffs (list(np.ndarray)): List of Clarke subdifferentials used at each iterate.
        - step (int): Number of steps. 
    """

    # ========================
    # Validate / Unpack Inputs
    # ========================
    
    # f
    if not callable(f):
        raise ValueError("Input f = {} has type {} when it should have type tp.Callable[[np.ndarray], float]".format(f, type(f)))

    # clarke_f
    if not callable(clarke_f):
        raise ValueError("Input clarke_f = {} has type {} when it should have type tp.Callable[[np.ndarray], set(np.ndarray)]".format(clarke_f, type(clarke_f)))

    # subgrad_check_bound
    if subgrad_check_bound:
        if type(subgrad_check_bound) == int:
            subgrad_check_abs_tol = 1e-10
        elif type(subgrad_check_bound) == tuple and type(subgrad_check_bound[0]) == int and type(subgrad_check_bound[1]) == float:
            subgrad_check_abs_tol = subgrad_check_bound[1]
            subgrad_check_bound   = subgrad_check_bound[0]
        else:
            raise ValueError("Input subgrad_check_bound = {} has type {} when it should have type int or (int, float) or None".format(subgrad_check_bound, type(subgrad_check_bound)))
        if not subgrad_check_bound > 0:
            raise ValueError("Input subgrad_check_bound = {} is non-positive when it should be".formamt(subgrad_check_bound))
        if not subgrad_check_abs_tol > 0:
            raise ValueError("Input subgrad_check_bound = ({}, {}) has a non-positive second argument, when both arguments should be positive".format(subgrad_check_bound, subgrad_check_abs_tol))
    else:
        subgrad_check_bound = 1
        subgrad_check_abs_tol = 1e-10

    # step_bound
    if step_bound:
        if not type(step_bound) == int:
            raise ValueError("Input step_bound = {} has type {} when it should have type int".format(step_bound, type(step_bound)))
    else:
        step_bound = np.inf
    
    # initial_point
    if not type(initial_point) == np.ndarray:
        raise ValueError("Input initial_point = {} has type {} when it should have type np.ndarray".format(initial_point, type(initial_point)))
    
    # step_sequence
    if type(step_sequence) == list and type(step_sequence[0]) == float:
        if not len(step_sequence) == step_bound:
            raise ValueError("Input step_sequence = {} has length {} when it should have length equal to other input step_bound = {}. Alteratively, step_sequence can be defined by a function with type tp.Callable[[int], float]".format(step_sequence, len(step_sequence), step_bound))
        else:
            step_func = lambda x: step_sequence[x]
    elif callable(step_sequence):
        step_func = lambda x: step_sequence(x)
    else:
        raise ValueError("Input step_sequence = {} has type {} when it should have type list[float] or tp.Callable[[int], float]".format(step_sequence, type(step_sequence)))
    
    # f and initial_point compatibility
    try:
        val = f(initial_point)
    except:
        raise ValueError("Input f = {} and input initial_point = {} are not compatible, i.e. f(initial_point) is undefined".format(f, initial_point))
    if not (type(val) == float or type(val) == np.float_):
            raise ValueError("Input f = {} returns values of type {} when it should return a float".format(f, type(val)))
    
    # clarke_f and initial_point compatibility
    try:
        subgrad = clarke_f(initial_point)
    except:
        raise ValueError("Input clarke_f = {} and input initial_point = {} are not compatible, i.e. clarke_f(initial_point) is undefined".format(clarke_f, initial_point))
    if not type(subgrad) == list:
        raise ValueError("Input subgrad = {} returns values of type {} when it should return a list(np.ndarray)".format(subgrad, type(subgrad)))
    eg = subgrad[0]
    if not type(eg) == np.ndarray:
        raise ValueError("Input subgrad = {} returns values of type {} when it should return a list(np.ndarray)".format(subgrad, type(subgrad)))
    # FIXME - fails for tf.test?
    # if not initial_point.shape == eg.shape:
    #     raise ValueError("Input clarke_f = {} returns values of shape {} when it should return a value with the same shape as its input = {}".format(clarke_f, eg.shape, initial_point.shape))
    
    # heuristic
    if heuristic:
        try:
            eg = heuristic(clarke_f(initial_point))
        except:
            raise ValueError("Input heuristic = {} and output of clarke_f = {} are not compatible, i.e. heuristic(clarke_f(initial_point)) is undefined".format(heuristic, clarke_f))
        if not type(eg) == np.ndarray:
            raise ValueError("Input heuristic = {} returns values of type {} when it should return a np.ndarray".format(heuristic, type(eg)))
        if not initial_point.shape == eg.shape:
            raise ValueError("Input heuristic = {} returns values of shape {} when it should return a value with the same shape as the initial point = {}".format(heuristic, eg.shape, initial_point.shape))
    else:
        heuristic = lambda x: x[np.random.randint(0,len(x))]

    # ==================
    # Function Operation
    # ==================
    
    # Initialise loop variables
    iterates_points   = [initial_point]
    iterates_values   = [f(initial_point)]
    iterates_subgrads = [clarke_f(initial_point)]
    iterates_subdiffs = [heuristic(iterates_subgrads[-1])]
    check_count = 0

    # Check
    if printing:
        print("Initial Point: {}".format(iterates_points[-1]))
        print("Initial Value: {}".format(iterates_values[-1]))
        print("Initial Subgradients: {}".format(iterates_subgrads[-1]))
        print("Initial Subdifferential: {}".format(iterates_subdiffs[-1]))

    # Loop while stopping conditions aren't met
    if step_bound == np.inf:
        step_count = 0
        while check_count < subgrad_check_bound:
            # Print update if not halting
            if step_count != 0 and printing:
                print("Step {}:".format(step_count))
                print("Current Point: {}".format(iterates_points[-1]))
                print("Current Value: {}".format(iterates_values[-1]))
                print("Current Subgradients: {}".format(iterates_subgrads[-1]))
                print("Current Subdifferential: {}".format(iterates_subdiffs[-1]))
            
            # Track progress in 10000 step increments
            base_count = step_count
            for internal_step_count in trange(10000, desc="Steps [{},{})".format(base_count,base_count+10000), leave=False, position=2):
                # Check step size
                if step_func(step_count) == 0 and printing:
                    print("Step size reached 0. Algorithm halted.")
                    check_count = np.inf
                    break

                # Calculate new point, value, and Clarke subdifferentials
                x_k_point = iterates_points[-1] - step_func(step_count+internal_step_count) * iterates_subdiffs[-1]
                try:
                    x_k_value = f(x_k_point)
                except Exception as e:
                    print("Exception when computing function: {}".format(e))
                    print("Step: {}".format(step_count+internal_step_count))
                    print("Points: {}".format(iterates_points))
                    print("Values: {}".format(iterates_values))
                    print("Subgradients: {}".format(iterates_subgrads))
                    print("Subdifferentials: {}".format(iterates_subdiffs))
                    return False
                try:
                    x_k_subgrad = clarke_f(x_k_point)
                except Exception as e:
                    print("Exception when computing clarke f: {}".format(e))
                    print("Step: {}".format(step_count+internal_step_count))
                    print("Points: {}".format(iterates_points))
                    print("Values: {}".format(iterates_values))
                    print("Subgradients: {}".format(iterates_subgrads))
                    print("Subdifferentials: {}".format(iterates_subdiffs))
                    return False

                # Select Clarke subdifferential
                try:
                    x_k_subdiff = heuristic(x_k_subgrad)
                except Exception as e:
                    print("Exception when selecting clarke f: {}".format(e))
                    print("Step: {}".format(step_count+internal_step_count))
                    print("Points: {}".format(iterates_points))
                    print("Values: {}".format(iterates_values))
                    print("Subgradients: {}".format(iterates_subgrads))
                    print("Subdifferentials: {}".format(iterates_subdiffs))
                    return False

                # Update lists
                iterates_points.append(x_k_point)
                iterates_values.append(x_k_value)
                iterates_subgrads.append(x_k_subgrad)
                iterates_subdiffs.append(x_k_subdiff)

                # Update stopping conditions
                if [subdiff for subdiff in iterates_subgrads[-1] if np.linalg.norm(subdiff, ord=2) < subgrad_check_abs_tol]:
                    check_count += 1
                else:
                    check_count = 0
                
                # Check stopping conditions
                if check_count >= subgrad_check_bound:
                    break
            
            # Count final step
            step_count += 1 + internal_step_count
    else:
        for step_count in trange(step_bound, desc="Steps", leave=False, position=2):
            # Check step size
            if step_func(step_count) == 0 and printing:
                print("Step size reached 0. Algorithm halted.")
                check_count = np.inf
                break
            
            # Calculate new point, value, and Clarke subdifferentials
            x_k_point = iterates_points[-1] - step_func(step_count) * iterates_subdiffs[-1]
            try:
                x_k_value = f(x_k_point)
            except Exception as e:
                print("Exception when computing function: {}".format(e))
                print("Step: {}".format(step_count))
                print("Points: {}".format(iterates_points))
                print("Values: {}".format(iterates_values))
                print("Subgradients: {}".format(iterates_subgrads))
                print("Subdifferentials: {}".format(iterates_subdiffs))
                return False
            try:
                x_k_subgrad = clarke_f(x_k_point)
            except Exception as e:
                print("Exception when computing clarke f: {}".format(e))
                print("Step: {}".format(step_count))
                print("Points: {}".format(iterates_points))
                print("Values: {}".format(iterates_values))
                print("Subgradients: {}".format(iterates_subgrads))
                print("Subdifferentials: {}".format(iterates_subdiffs))
                return False

            # Select Clarke subdifferential
            try:
                x_k_subdiff = heuristic(x_k_subgrad)
            except Exception as e:
                print("Exception when selecting clarke f: {}".format(e))
                print("Step: {}".format(step_count))
                print("Points: {}".format(iterates_points))
                print("Values: {}".format(iterates_values))
                print("Subgradients: {}".format(iterates_subgrads))
                print("Subdifferentials: {}".format(iterates_subdiffs))
                return False

            # Update lists
            iterates_points.append(x_k_point)
            iterates_values.append(x_k_value)
            iterates_subgrads.append(x_k_subgrad)
            iterates_subdiffs.append(x_k_subdiff)

            # Update stopping conditions
            if [subdiff for subdiff in iterates_subgrads[-1] if np.linalg.norm(subdiff, ord=2) < subgrad_check_abs_tol]:
                check_count += 1
            else:
                check_count = 0
            
            # Check stopping conditions
            if check_count > subgrad_check_bound:
                break
    
        # Count final step
        step_count += 1

    # Return final results 
    return iterates_points, iterates_values, iterates_subgrads, iterates_subdiffs, step_count