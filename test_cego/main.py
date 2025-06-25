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
    KMA: float
    KMB: float
    KMC: float
    KMD: float
    KME: float
    KMF: float
    KMG: float
    KA: float
    KB: float
    KC: float
    KD: float
    KE: float
    KF: float
    KG: float

    UMA: float
    UMB: float
    UMC: float
    UMD: float
    UME: float
    UMF: float
    UMG: float
    UA: float
    UB: float
    UC: float
    UD: float
    UE: float
    UF: float
    UG: float

    klist = []
    ulist = []
    with open("constant_known_position.txt", "r") as file:
        for line in file:
            klist.append(float(line.strip('\n')))

    with open("constant_unknown_position.txt", "r") as file:
        for line in file:
            ulist.append(float(line.strip('\n')))
    better = ulist[0]
    KMA = KA = klist[1]
    KMB = KB = klist[2]
    KMC = KC = klist[3]
    KMD = KD = klist[4]
    KME = KE = klist[5]
    KMF = KF = klist[6]
    KMG = KG = klist[7]
    UMA = UA = ulist[1]
    UMB = UB = ulist[2]
    UMC = UC = ulist[3]
    UMD = UD = ulist[4]
    UME = UE = ulist[5]
    UMF = UF = ulist[6]
    UMG = UG = ulist[7]

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
            Explorer(env, explorer_file, master_rescuer, (KA,KB,KC,KD,KE,KF,KG), (UA,UB,UC,UD,UE,UF,UG))

        # Run the environment simulator
        env.run()
        if len(victims_found) > better:
            better = len(victims_found)
            UMA = UA
            UMB = UB
            UMC = UC
            UMD = UD
            UME = UE
            UMF = UF
            UMG = UG
            KMA = KA
            KMB = KB
            KMC = KC
            KMD = KD
            KME = KE
            KMF = KF
            KMG = KG
            try:
                with open('constant_known_position.tmp', 'w') as file:
                    file.write(f"{better}\n")
                    file.write(f"{KMA}\n")
                    file.write(f"{KMB}\n")
                    file.write(f"{KMC}\n")
                    file.write(f"{KMD}\n")
                    file.write(f"{KME}\n")
                    file.write(f"{KMF}\n")
                    file.write(f"{KMG}\n")

                os.rename('constant_known_position.tmp', 'constant_known_position.txt')
            except KeyboardInterrupt:
                with open('constant_known_position.tmp', 'w') as file:
                    file.write(f"{better}\n")
                    file.write(f"{KMA}\n")
                    file.write(f"{KMB}\n")
                    file.write(f"{KMC}\n")
                    file.write(f"{KMD}\n")
                    file.write(f"{KME}\n")
                    file.write(f"{KMF}\n")
                    file.write(f"{KMG}\n")
            try:
                with open('constant_unknown_position.tmp', 'w') as file:
                    file.write(f"{better}\n")
                    file.write(f"{UMA}\n")
                    file.write(f"{UMB}\n")
                    file.write(f"{UMC}\n")
                    file.write(f"{UMD}\n")
                    file.write(f"{UME}\n")
                    file.write(f"{UMF}\n")
                    file.write(f"{UMG}\n")

                os.rename('constant_unknown_position.tmp', 'constant_unknown_position.txt')
            except KeyboardInterrupt:
                with open('constant_unknown_position.tmp', 'w') as file:
                    file.write(f"{better}\n")
                    file.write(f"{UMA}\n")
                    file.write(f"{UMB}\n")
                    file.write(f"{UMC}\n")
                    file.write(f"{UMD}\n")
                    file.write(f"{UME}\n")
                    file.write(f"{UMF}\n")
                    file.write(f"{UMG}\n")
        KA = KMA + random.uniform(-0.2, 0.2)
        KB = KMB + random.uniform(-0.2, 0.2)
        KC = KMC + random.uniform(-0.2, 0.2)
        KD = KMD + random.uniform(-0.2, 0.2)
        KE = KME + random.uniform(-0.2, 0.2)
        KF = KMF + random.uniform(-0.2, 0.2)
        KG = KMG + random.uniform(-0.2, 0.2)
        UA = UMA + random.uniform(-0.2, 0.2)
        UB = UMB + random.uniform(-0.2, 0.2)
        UC = UMC + random.uniform(-0.2, 0.2)
        UD = 1.0
        UE = UME + random.uniform(-0.2, 0.2)
        UF = UMF + random.uniform(-0.2, 0.2)
        UG = UMG + random.uniform(-0.2, 0.2)

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
