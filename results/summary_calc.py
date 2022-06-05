import os
import numpy as np
import pandas as pd

# Generating Order and Rate CSV
df = pd.DataFrame(columns=['function', 'run', 'steps', 'order', 'order error', 'rate', 'rate error'])
dict_ready = False

for root, dirs, _ in os.walk(".", topdown=True):
    for dir in dirs:
        for _, _, files in os.walk(os.path.join(root, dir)):
            for file in files:
                if str(file) == "results.txt":
                    with open(os.path.join(root, dir, file)) as f:
                        lines = f.readlines()
                    for line in lines:
                        if line[:3] == "Run":
                            new_dict = dict()
                            new_dict['function'] = str(dir)
                            splits = line.split(" ")
                            new_dict['run'] = int(splits[1])
                        if line[:5] == "Steps":
                            splits = line.split(" ")
                            new_dict['steps'] = int(splits[1])
                        if line[:20] == "Order of Convergence":
                            splits = line.split(" ")
                            new_dict['order'] = float(splits[3])
                            new_dict['order error'] = float(splits[5])
                        if line[:19] == "Rate of Convergence":
                            splits = line.split(" ")
                            new_dict['rate'] = float(splits[3])
                            new_dict['rate error'] = float(splits[5])
                            df = df.append(new_dict, ignore_index=True)
df.to_csv("summary.csv")


# Weighted Average Results
steps_array = list()
rate_array = list()
rate_error_array = list()
order_array = list()
order_error_array = list()

for _, _, files in os.walk(".", topdown=True):
    for file in files:
        if str(file) == "results.txt":
            with open(os.path.join(root, file)) as f:
                lines = f.readlines()
            for line in lines:
                if line[:5] == "Steps":
                    splits = line.split(" ")
                    steps_array.append(int(splits[1]))
                if line[:20] == "Order of Convergence":
                    splits = line.split(" ")
                    order_array.append(float(splits[3]))
                    order_error_array.append(float(splits[5]))
                if line[:19] == "Rate of Convergence":
                    splits = line.split(" ")
                    rate_array.append(float(splits[3]))
                    rate_error_array.append(float(splits[5]))
    
print("Weighted Rate: {:0.6f}".format(np.average(np.array(rate_array), weights=np.array(steps_array))))
print("Weighted Order Error: {:0.6f}".format(np.average(rate_error_array), weights=np.array(steps_array)))
print("Weighted Order: {:0.6f}".format(np.average(np.array(order_array), weights=np.array(steps_array))))
print("Weighted Order Error: {:0.6f}".format(np.average(order_error_array), weights=np.array(steps_array)))