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
    MA: float
    MB: float
    MC: float
    MD: float
    ME: float
    MF: float
    MG: float
    A: float
    B: float
    C: float
    D: float
    E: float
    F: float
    G: float
    lista = []
    with open("constant.txt", "r") as file:
        for line in file:
            lista.append(float(line.strip('\n')))
    better = lista[0]
    A = lista[1]
    B = lista[2]
    C = lista[3]
    D = lista[4]
    E = lista[5]
    F = lista[6]
    G = lista[7]

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
            Explorer(env, explorer_file, master_rescuer, (A,B,C,D,E,F,G))

        # Run the environment simulator
        env.run()
        if len(victims_found) > better:
            better = len(victims_found)
            MA = A
            MB = B
            MC = C
            MD = D
            ME = E
            MF = F
            MG = G
            #print(f"Vítimas encontradas: {len(victims_found)}")
            #print(f"Best A: {MA}")
            #print(f"Best B: {MB}")
            #print(f"Best C: {MC}")
            #print(f"Best D: {MD}")
            #print(f"Best E: {ME}")
            #print(f"Best F: {MF}")
            try:
                with open('constant.tmp', 'w') as file:
                    file.write(f"{better}\n")
                    file.write(f"{MA}\n")
                    file.write(f"{MB}\n")
                    file.write(f"{MC}\n")
                    file.write(f"{MD}\n")
                    file.write(f"{ME}\n")
                    file.write(f"{MF}\n")
                    file.write(f"{MG}\n")

                os.rename('constant.tmp', 'constant.txt')
            except KeyboardInterrupt:
                with open('constant.tmp', 'w') as file:
                    file.write(f"{better}\n")
                    file.write(f"{MA}\n")
                    file.write(f"{MB}\n")
                    file.write(f"{MC}\n")
                    file.write(f"{MD}\n")
                    file.write(f"{ME}\n")
                    file.write(f"{MF}\n")
                    file.write(f"{MG}\n")
        A = random.uniform(-1, 1)
        B = random.uniform(-1, 1)
        C = random.uniform(-1, 1)
        D = random.uniform(-1, 1)
        E = random.uniform(-1, 1)
        F = random.uniform(-1, 1)
        G = random.uniform(-1, 1)
        
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
