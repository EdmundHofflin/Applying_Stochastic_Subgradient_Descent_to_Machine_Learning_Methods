"""
Edmund Hofflin: u6373930

This file is a temp file for testing the various modules during development
"""

# =======
# Imports
# =======

# My Imports
import algorithms as alg
import test_functions as tf

# General Packages
import numpy as np

# Graphing Packages
import plotly.graph_objects as go

# Recording results
import os

# Progress bars
from tqdm import trange


# ======
# Set up
# ======

# Algorithm Constants
subgrad_check_bound = (1, 1e-5)
step_bound          = int(1e6) # None
if step_bound:
    step_sequence   = lambda x: 1/(x+1) ** 0.55 # lambda x: 1/2 if x < step_bound // 10000 else 1/(x-(step_bound // 10000)+1) ** 0.85
else:
    step_sequence   = lambda x: 1/(x+1) ** 0.85
heuristic           = None # Default to random uniform selection

# Function Dictionary
function_dict = dict()
function_dict["quadratic"] = lambda: tf.quadratic()
function_dict["relu"] = lambda: tf.relu(rabs=subgrad_check_bound[1])
function_dict["absolute"] = lambda: tf.p_norm(ord=2.0, rabs=subgrad_check_bound[1])

function_dict["piecewise"] = lambda: tf.piecewise(rabs=subgrad_check_bound[1])

function_dict["p-norm_1"] = lambda: tf.p_norm(ord=1.0, rabs=subgrad_check_bound[1])
function_dict["p-norm_2"] = lambda: tf.p_norm(ord=2.0, rabs=subgrad_check_bound[1])
function_dict["p-norm_inf"] = lambda: tf.p_norm(ord=np.inf, rabs=subgrad_check_bound[1])

function_dict["f_waves_1"] = lambda: tf.flat_waves(ord=1.0, rabs=subgrad_check_bound[1])
function_dict["f_waves_2"] = lambda: tf.flat_waves(ord=2.0, rabs=subgrad_check_bound[1])
function_dict["f_waves_inf"] = lambda: tf.flat_waves(ord=np.inf, rabs=subgrad_check_bound[1])

function_dict["waves_1"] = lambda: tf.waves(ord=1.0, rabs=subgrad_check_bound[1])
function_dict["waves_2"] = lambda: tf.waves(ord=2.0, rabs=subgrad_check_bound[1])
function_dict["waves_inf"] = lambda: tf.waves(ord=np.inf, rabs=subgrad_check_bound[1])

function_dict["spiral"] = lambda: tf.spiral(rabs=subgrad_check_bound[1])

# Showing
SHOW = False

# ================
# Testing Function
# ================

def test(function_name : str, dim : int = 2, tests : int = 10):
    """ Tests a given function a certain amount of time, recording and analysing the results """

    # Set up result saving by creating folder for results
    results_path = "../local_results/{}/".format(function_name)
    try:
        os.mkdir(results_path)
    except OSError as e:
        print("Results directory {} could not be created. {}".format(results_path, e))
    
    # Create blank results and error file
    with open(results_path + "results.txt", 'w') as file:
        pass
    with open(results_path + "table.txt", 'w') as file:
        file.write("run & PARAMETERS & initial iterate & steps & final iterate & order & order error & rate & rate error \\\\ \hline \hline\n")
    with open(results_path + "error.txt", 'w') as file:
        pass

    # For each test
    for run_number in trange(tests, desc="Test", leave=False, position=1):
        # Select (and potentially randomised) function
        f, clarke_f, par_dict = function_dict[function_name]()

        # Run SSD
        success = False
        errors = 0
        while (not success) and errors < tests:
            initial_point = np.power(1, np.random.randint(0,2,[dim])) * np.random.uniform(5, 10, [dim])
            try:
                points, values, subgrads, subdiffs, steps = alg.ssd(f, clarke_f, subgrad_check_bound, step_bound, initial_point, step_sequence, heuristic)
                success = True
            except Exception as e:
                print("Something went wrong. {}".format(e))
                # Save error
                with open(results_path + "error.txt", 'a') as file:
                    file.write("Error in run {}:\n".format(run_number))
                    file.write(str(e))
                    file.write("\n\n")
                errors += 1
        if errors == tests:
            print("it all went wrong")
            break
        # Calculate rate of convergence
        point_data = np.array(points)
        value_data = np.array(values)
        error_data = np.linalg.norm(point_data - point_data[-1], ord=2, axis=1)
        log_error = np.log(1 + error_data[:-1])
        p, V = np.polyfit(log_error[:-1], log_error[1:], deg=1, cov=True)
        m, b = p[0], p[1]
        # Save Key Data
        with open(results_path + "results.txt", 'a') as file:
            file.write("========\n")
            file.write("Run {}\n".format(run_number))
            file.write("Parameters: {}\n".format(par_dict))
            file.write("========\n")
            file.write("Initial Point: {}\n".format(points[0]))
            file.write("Optimal Point: {}\n".format(points[-1]))
            file.write("Optimal Value: {}\n".format(values[-1]))
            file.write("Optimal Subgradients: {}\n".format(subgrads[-1]))
            file.write("Last Subdiff: {}\n".format(subdiffs[-2]))
            file.write("Steps: {}\n".format(steps))
            file.write("Order of Convergence: {} +/- {}\n".format(m, np.sqrt(V[0][0])))
            file.write("Rate of Convergence: {} +/- {}\n\n".format(np.exp(b), np.exp(np.sqrt(V[1][1]))))
        # Table for latex plotting
        with open(results_path + "table.txt", 'a') as file:
            par_str = ""
            for key in par_dict:
                if not (key == "ord" or key == "rabs"):
                    par_str += "{0:0.6f} & ".format(par_dict[key])
            par_str = par_str[:-3]

            file.write("{0} & {1} & {2} & {3} & {4} & {5:0.6f} & {6:0.6f} & {7:0.6f} & {8:0.6f}\\\\\n".format(run_number, par_str, np.round_(points[0], 6), steps, np.round_(points[-1], 6), m, np.sqrt(V[0][0]), np.exp(b), np.exp(np.sqrt(V[1][1]))))
        # Create subdirectory for plots
        try:
            os.mkdir(results_path + "data_{}/".format(run_number))
        except OSError as e:
            with open(results_path + "error.txt", 'a') as file:
                    file.write("Error in creating subdirectory  plots\n".format(run_number))
                    file.write(str(e))
                    file.write("\n\n")
        # Data
        DATA_DENSITY = 1000
        plotting_boundary = np.ceil(max(1.5 * np.min(np.abs(point_data)), 1.5 * np.max(np.abs(point_data)), 15))
        if dim == 2:
            x_data = np.linspace(-plotting_boundary, plotting_boundary, DATA_DENSITY)
            y_data = np.linspace(-plotting_boundary, plotting_boundary, DATA_DENSITY)
            z_data = np.zeros(shape=(DATA_DENSITY, DATA_DENSITY))
            for i in range(DATA_DENSITY):
                for j in range(DATA_DENSITY):
                    z_data[i,j] = f(np.array([x_data[i], y_data[j]]))
            # Topological plot
            fig_topo = go.Figure(data=[
                go.Surface(z=z_data, x=x_data, y=y_data,
                    colorscale='Blugrn'),
                go.Scatter3d(x=point_data[:,1],
                    y=point_data[:,0], z=value_data,
                    mode='markers',
                    marker=dict(size=5.0, color=np.linspace(0,20,steps), colorscale='Cividis', opacity=1.0))
                ])
        else:
            x_data = np.linspace(-plotting_boundary, plotting_boundary, DATA_DENSITY)
            z_data = np.zeros(shape=(DATA_DENSITY))
            for i in range(DATA_DENSITY):
                z_data[i] = f(np.array([x_data[i]]))
            # Topological plot
            fig_topo = go.Figure(data=[
                go.Scatter(x=x_data, y=z_data),
                go.Scatter(x=point_data[:,0], y=value_data,
                    mode='markers',
                    marker=dict(size=5.0, color=np.linspace(0,20,steps), colorscale='Cividis', opacity=1.0))
                ])
        fig_topo.update_layout(title='Plot of Iterates on Graph - Run {}'.format(run_number))
        fig_topo.write_html(results_path + "data_{}/topo.html".format(run_number))
        # Error Plot
        fig_error = go.Figure(data=[
            go.Scatter(x=np.array(range(steps)), y=error_data,
            mode='markers', marker=dict(size=5.0, color=np.linspace(0,20,steps), colorscale='Cividis', opacity=1.0))
            ])
        fig_error.update_yaxes(type="log")
        fig_error.update_layout(title='Errors of Iterates - Run {}'.format(run_number))
        fig_error.write_html(results_path + "data_{}/error.html".format(run_number))
        # Convergence
        linear_rate = 2 ** (-np.linspace(0,int(np.ceil(np.log(steps))),2*int(np.ceil(np.log(steps)))+1))
        def n_rate(n:float):
            return linear_rate ** n
        fig_convergence = go.Figure(data=[
            go.Scatter(x=log_error[:-1], y=log_error[1:],
            mode='markers', marker=dict(size=5.0, color=np.linspace(0,20,steps), colorscale='Cividis', opacity=1.0),
            name="Convergence"),
            go.Scatter(x=linear_rate, y=n_rate(m),
            mode='markers', marker=dict(size=5.0, color='Blue', opacity=0.8),
            name="Order {}, Rate {}".format(m, np.exp(b))),
            ])
        fig_convergence.update_xaxes(type="log")
        fig_convergence.update_yaxes(type="log")
        fig_convergence.update_layout(title='Convergence Rates - Run {}'.format(run_number), showlegend=True)
        fig_convergence.write_html(results_path + "data_{}/convergence.html".format(run_number))

        # Save data
        np.savetxt(results_path + "data_{}/points.dat".format(run_number), point_data, delimiter=' ',newline='\n', fmt='%4.6f',)
        np.savetxt(results_path + "data_{}/values.dat".format(run_number), value_data, delimiter=' ',newline='\n', fmt='%4.6f',)
        np.savetxt(results_path + "data_{}/errors.dat".format(run_number), error_data, delimiter=' ',newline='\n', fmt='%4.6f',)
        np.savetxt(results_path + "data_{}/logs.dat".format(run_number), log_error, delimiter=' ',newline='\n', fmt='%4.6f',)

        for incr in [1e1, 1e2, 1e3, 1e4, 1e5]:
            if incr < steps // 10:
                incr = int(incr)
                np.savetxt(results_path + "data_{}/points_{}.dat".format(run_number, incr), point_data[::incr], delimiter=' ',newline='\n', fmt='%4.6f',)
                np.savetxt(results_path + "data_{}/values_{}.dat".format(run_number, incr), value_data[::incr], delimiter=' ',newline='\n', fmt='%4.6f',)
                np.savetxt(results_path + "data_{}/errors_{}.dat".format(run_number, incr), error_data[::incr], delimiter=' ',newline='\n', fmt='%4.6f',)
                np.savetxt(results_path + "data_{}/logs_{}.dat".format(run_number, incr), log_error[::incr], delimiter=' ',newline='\n', fmt='%4.6f',)

        # Show plots
        if SHOW:
            fig_topo.show()
            fig_error.show()
            fig_convergence.show()


# ===========
# # Run Tests
# ===========

inputs = [
    # ("quadratic",  1),
    # ("relu",       1),
    # ("absolute",   1),
    # ("piecewise",  2),
    # ("p-norm_1",   2),
    # ("p-norm_2",   2),
    # ("p-norm_inf", 2),
    # ("waves_1",    2),
    # ("waves_2",    2),
    # ("waves_inf",  2),
    # ("f_waves_1",  2),
    # ("f_waves_2",  2),
    # ("f_waves_inf",2),
    # ("spiral", 2)
]

for input in inputs:
    print("Testing {}".format(input[0]))
    test(input[0], input[1], tests=10)