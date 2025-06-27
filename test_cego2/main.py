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

    constant_u.pop(0)
    constant_k.pop(0)

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
