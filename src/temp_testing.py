"""
Edmund Hofflin: u6373930

This file is a temp file for testing the various modules during development
"""

# Imports
import algorithms as alg
import test_functions as tf

import numpy as np

# Constants
subgrad_check_bound = (2, 1e-5)
step_bound          = None
initial_point       = np.random.uniform(-5, 5, [2])
if step_bound:
    step_sequence = lambda x: 1/2 if x < step_bound // 5 else 1/(x-(step_bound // 5)+1)
else:
    step_sequence = lambda x: 1/2 if x < 10000 else 1/(x-10000+1)
    # step_sequence = lambda x: 1/(1+x)
heuristic = None

# # Function
# f, df = tf.figure()

# # Run
# points, values, grads, steps = alg.sgd(f, df, relative_error_bound, step_bound, initial_point, step_sequence)
# print("Optimal Point: {}".format(points[-1]))
# print("Optimal Value: {}".format(values[-1]))
# print("Optimal Gradient: {}".format(grads[-1]))
# print("Steps: {}".format(steps))

# Function
# f, clarke_f = tf.relu(rabs=subgrad_check_bound[1])
# f, clarke_f = tf.p_norm(a=1.0, ord=1, rabs=subgrad_check_bound[1])
f, clarke_f = tf.piecewise_log_1norm(rabs=subgrad_check_bound[1])

# Run
points, values, subgrads, subdiffs, steps = alg.ssd(f, clarke_f, subgrad_check_bound, step_bound, initial_point, step_sequence, heuristic)
print("Optimal Points: {}".format(points[-(subgrad_check_bound[0]+1):]))
print("Optimal Values: {}".format(values[-(subgrad_check_bound[0]+1):]))
print("Optimal Subgradients: {}".format(subgrads[-1]))
print("Optimal Subdifferential: {}".format(subdiffs[-1]))
print("Steps: {}".format(steps))

# import plotly.express as px
# import pandas as pd

# for i in range(subgrad_check_bound[0]):
#     df = pd.DataFrame(subgrads[-subgrad_check_bound[0]+i], columns=['x', 'y'])
#     fig = px.scatter(df, x='x', y='y')
#     fig.show()

import plotly.graph_objects as go

DATA_DENSITY = 1000

# Generate Plot Data
data = np.array(points)
x_data = np.linspace(data[-1,0] - 1.1 * max(np.abs(data[-1,0] - np.min(data)), np.abs(data[-1,0] - np.max(data))), data[-1,0] + 1.1 * max(np.abs(data[-1,0] - np.min(data)), np.abs(data[-1,0] - np.max(data))), DATA_DENSITY)
y_data = np.linspace(data[-1,1] - 1.1 * max(np.abs(data[-1,1] - np.min(data)), np.abs(data[-1,1] - np.max(data))), data[-1,1] + 1.1 * max(np.abs(data[-1,1] - np.min(data)), np.abs(data[-1,1] - np.max(data))), DATA_DENSITY)
z_data = np.zeros(shape=(DATA_DENSITY, DATA_DENSITY))
for i in range(DATA_DENSITY):
    for j in range(DATA_DENSITY):
        z_data[i,j] = f(np.array([x_data[i], y_data[j]]))
f_data = np.zeros(shape=(len(points)))
for i in range(len(points)):
    f_data[i] = f(points[i])

# Plot
fig = go.Figure(data=[
    go.Surface(z=z_data,
        x=x_data, y=y_data, colorscale='Blugrn'),
    go.Scatter3d(x=data[:,0],
        y=data[:,1], z=f_data,
        mode='markers',
        marker=dict(size=2.5, color=np.linspace(0,20,len(points)), colorscale='Cividis', opacity=1.0))
    ])
fig.update_layout(title='Plot of Convergence') #,
    # autosize=False,
    # width=500, height=500,
    # margin=dict(l=65, r=50, b=65, t=90))
fig.show()