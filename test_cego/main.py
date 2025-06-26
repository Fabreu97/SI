import sys
import os
import time
import random

## importa classes
from vs.environment import Env
from explorer import Explorer, victims_found
from rescuer import Rescuer

"""
    Constantes que variam no intervalo de [-1,1]
"""

file = None

def main(data_folder_name, config_ag_folder_name):

    constant_k = []
    constant_u = []

    with open("constant_known_position.txt", "r") as file:
        for line in file:
            constant_k.append(float(line.strip('\n')))

    with open("constant_unknown_position.txt", "r") as file:
        for line in file:
            constant_u.append(float(line.strip('\n')))

    better = constant_u[0]
    constant_u.pop(0)
    constant_k.pop(0)
    better_const_k = constant_k[:]
    better_const_u = constant_u[:]

    while True:
        # Set the path to config files and data files for the environment
        current_folder = os.path.abspath(os.getcwd())
        config_ag_folder = os.path.abspath(os.path.join(current_folder, config_ag_folder_name))
        data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))
        
        # Instantiate the environment
        env = Env(data_folder)
        
        # Instantiate master_rescuer
        # This agent unifies the maps and instantiate other 3 agents
        rescuer_file = os.path.join(config_ag_folder, "rescuer_1_config.txt")
        master_rescuer = Rescuer(env, rescuer_file, 4)   # 4 is the number of explorer agents

        # Explorer needs to know rescuer to send the map 
        # that's why rescuer is instatiated before
        for exp in range(1, 5):
            filename = f"explorer_{exp:1d}_config.txt"
            explorer_file = os.path.join(config_ag_folder, filename)
            Explorer(env, explorer_file, master_rescuer, tuple(constant_k), tuple(constant_u))

        # Run the environment simulator
        env.run()
        if len(victims_found) > better:
            better = len(victims_found)
            better_const_k = constant_k[:]
            better_const_u = constant_u[:]
            try:
                with open('constant_known_position.tmp', 'w') as file:
                    file.write(f"{better}\n")
                    for const in better_const_k:
                        file.write(f"{const}\n")

                os.rename('constant_known_position.tmp', 'constant_known_position.txt')
            except KeyboardInterrupt:
                with open('constant_known_position.tmp', 'w') as file:
                    file.write(f"{better}\n")
                    for const in better_const_k:
                        file.write(f"{const}\n")
            try:
                with open('constant_unknown_position.tmp', 'w') as file:
                    file.write(f"{better}\n")
                    for const in better_const_u:
                        file.write(f"{const}\n")

                os.rename('constant_unknown_position.tmp', 'constant_unknown_position.txt')
            except KeyboardInterrupt:
                with open('constant_unknown_position.tmp', 'w') as file:
                    file.write(f"{better}\n")
                    for const in better_const_u:
                        file.write(f"{const}\n")
        max = 1.0
        for i in range(len(better_const_u)):
            constant_k[i] = better_const_k[i] + random.uniform(-0.1, 0.1)
            constant_u[i] = better_const_u[i] + random.uniform(-0.1, 0.1)
            if constant_k[i] > max:
                max = constant_k[i]
            if constant_u[i] > max:
                max = constant_u[i]
        
        for i in range(len(better_const_u)):
            constant_k[i] /= max
            constant_u[i] /= max

if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""
    
    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
        config_ag_folder_name = sys.argv[2]
    else:
        data_folder_name = os.path.join("datasets", "data_300v_90x90")
        config_ag_folder_name = os.path.join("ex03_mas_random_dfs", "cfg_1")
    
    main(data_folder_name, config_ag_folder_name)
