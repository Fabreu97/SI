##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### This rescuer version implements:
### - clustering of victims by quadrants of the explored region 
### - definition of a sequence of rescue of victims of a cluster
### - assigning one cluster to one rescuer
### - calculating paths between pair of victims using breadth-first search
###
### One of the rescuers is the master in charge of unifying the maps and the information
### about the found victims.

import os
import random
import math
import csv
import sys
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from bfs import BFS
from abc import ABC, abstractmethod

import time
from sklearn.cluster import KMeans # Biblioteca para o cluster utilizando o algoritmo K-means
from a_star import ASTAR

## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file, nb_of_explorers=1,clusters=[]):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file
        @param nb_of_explorers: number of explorer agents to wait for
        @param clusters: list of clusters of victims in the charge of this agent"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.nb_of_explorers = nb_of_explorers       # number of explorer agents to wait for start
        self.received_maps = 0                       # counts the number of explorers' maps
        self.map = Map()                             # explorer will pass the map
        self.victims = {}            # a dictionary of found victims: [vic_id]: ((x,y), [<vs>])
        self.plan = []               # a list of planned actions in increments of x and y
        self.plan_x = 0              # the x position of the rescuer during the planning phase
        self.plan_y = 0              # the y position of the rescuer during the planning phase
        self.plan_visited = set()    # positions already planned to be visited 
        self.plan_rtime = self.TLIM  # the remaing time during the planning phase
        self.plan_walk_time = 0.0    # previewed time to walk during rescue
        self.x = 0                   # the current x position of the rescuer when executing the plan
        self.y = 0                   # the current y position of the rescuer when executing the plan
        self.clusters = clusters     # the clusters of victims this agent should take care of - see the method cluster_victims
        self.sequences = clusters    # the sequence of visit of victims for each cluster 
        
                
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)

    def save_cluster_csv(self, cluster, cluster_id):
        filename = f"./clusters/cluster{cluster_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for vic_id, values in cluster.items():
                x, y = values[0]      # x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([vic_id, x, y, vs[6], vs[7]])

    def save_sequence_csv(self, sequence, sequence_id):
        filename = f"./clusters/seq{sequence_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for id, values in sequence.items():
                x, y = values[0]      # x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([id, x, y, vs[6], vs[7]])

    def cluster_victims(self):
        """ this method does a naive clustering of victims per quadrant: victims in the
            upper left quadrant compose a cluster, victims in the upper right quadrant, another one, and so on.
            
            @returns: a list of clusters where each cluster is a dictionary in the format [vic_id]: ((x,y), [<vs>])
                      such as vic_id is the victim id, (x,y) is the victim's position, and [<vs>] the list of vital signals
                      including the severity value and the corresponding label"""
        victims_positions = []
        for value in self.victims.values():
            victims_positions.append(value[0])

        km = KMeans(n_clusters=4, max_iter=100)
        km.fit(victims_positions)

        # Divide dictionary into quadrants
        cluster_0 = {}
        cluster_1 = {}
        cluster_2 = {}
        cluster_3 = {}

        for (key, victim_data), label in zip(self.victims.items(), km.labels_):
            if label == 0:
                cluster_0[key] = victim_data
            elif label == 1:
                cluster_1[key] = victim_data
            elif label == 2:
                cluster_2[key] = victim_data
            elif label == 3:
                cluster_3[key] = victim_data

        return [cluster_0, cluster_1, cluster_2, cluster_3]

    def predict_severity_and_class(self):
        """ @TODO to be replaced by a classifier and a regressor to calculate the class of severity and the severity values.
            This method should add the vital signals(vs) of the self.victims dictionary with these two values.

            This implementation assigns random values to both, severity value and class"""

        for vic_id, values in self.victims.items():
            severity_value = random.uniform(0.1, 99.9)          # to be replaced by a regressor 
            severity_class = random.randint(1, 4)               # to be replaced by a classifier
            values[1].extend([severity_value, severity_class])  # append to the list of vital signals; values is a pair( (x,y), [<vital signals list>] )


    def sequencing(self):
        """ Currently, this method sort the victims by the x coordinate followed by the y coordinate
            @TODO It must be replaced by a Genetic Algorithm that finds the possibly best visiting order """

        """ We consider an agent may have different sequences of rescue. The idea is the rescuer can execute
            sequence[0], sequence[1], ...
            A sequence is a dictionary with the following structure: [vic_id]: ((x,y), [<vs>]"""

        new_sequences = []

        # Implementar o Têmpera Simulada
        execution_time = 2
        start = time.time()
        end =time.time()

        keys = list(self.sequences[0].keys())
        sequence = list(self.sequences[0].values()) # dict
        position_list = [item[0] for item in sequence]
        index_list = list(range(0, len(sequence)))
        t = 0.0
        while execution_time > t:
            current_value = self._sum_euclian_distance_of_the_sequence(seq = position_list)
            possibles_exchanges = list(range(0, len(sequence)))
            best_value_neighbor = current_value + 10
            i1, i2 = 0,0 
            
            # Obtendo o melhor vizinho
            while(len(possibles_exchanges) > 2):
                index1 = random.choice(possibles_exchanges)
                possibles_exchanges.remove(index1)
                index2 = random.choice(possibles_exchanges)
                possibles_exchanges.remove(index2)
                
                #troca
                position_list[index1], position_list[index2] = position_list[index2], position_list[index1]
                
                neighbor = self._sum_euclian_distance_of_the_sequence(seq = position_list)
                if(best_value_neighbor > neighbor):
                    i1, i2 = index1, index2
                    best_value_neighbor = neighbor
                
                #destroca
                position_list[index1], position_list[index2] = position_list[index2], position_list[index1]
            
            T = self._scheduler(t, execution_time)
            delta_value = best_value_neighbor - current_value
            if delta_value > 0.0:
                position_list[i1], position_list[i2] = position_list[i2], position_list[i1]
                index_list[i1], index_list[i2] = index_list[i2], index_list[i1]
            else:
                prob = (math.e) ** (delta_value / T)
                weights = [prob, 1 - prob]
                if random.choices([True, False], weights=weights, k=1)[0]:
                    position_list[i1], position_list[i2] = position_list[i2], position_list[i1]
                    index_list[i1], index_list[i2] = index_list[i2], index_list[i1]

            end = time.time()
            t = end - start
        seq_dict = {}
        for index in index_list:
            seq_dict[keys[index]] = sequence[index]
        new_sequences.append(seq_dict)
        
        """
        for seq in self.sequences:   # a list of sequences, being each sequence a dictionary
            seq = dict(sorted(seq.items(), key=lambda item: item[0])) # Aluno: está ordenando pelo coordenada y
            new_sequences.append(seq)       
            #print(f"{self.NAME} sequence of visit:\n{seq}\n")
        """
        self.sequences = new_sequences

    def planner(self):
        """ A method that calculates the path between victims: walk actions in a OFF-LINE MANNER (the agent plans, stores the plan, and
            after it executes. Eeach element of the plan is a pair dx, dy that defines the increments for the the x-axis and  y-axis."""


        # let's instantiate the breadth-first search
        bfs = BFS(self.map, self.COST_LINE, self.COST_DIAG)
        astar = ASTAR(self.map, self.COST_LINE, self.COST_DIAG)

        # for each victim of the first sequence of rescue for this agent, we're going go calculate a path
        # starting at the base - always at (0,0) in relative coords
        
        if not self.sequences:   # no sequence assigned to the agent, nothing to do
            return

        # we consider only the first sequence (the simpler case)
        # The victims are sorted by x followed by y positions: [vic_id]: ((x,y), [<vs>]

        sequence = self.sequences[0]
        start = (0,0) # always from starting at the base
        for vic_id in sequence:
            goal = sequence[vic_id][0]
            plan, time = bfs.search(start, goal, self.plan_rtime)
            #plan, time = astar.execute(start, goal, self.plan_rtime)
            self.plan = self.plan + plan
            self.plan_rtime = self.plan_rtime - time
            start = goal

        # Plan to come back to the base
        goal = (0,0)
        plan, time = bfs.search(start, goal, self.plan_rtime)
        #plan, time = astar.execute(start, goal, self.plan_rtime)
        self.plan = self.plan + plan
        self.plan_rtime = self.plan_rtime - time
           

    def sync_explorers(self, explorer_map, victims):
        """ This method should be invoked only to the master agent

        Each explorer sends the map containing the obstacles and
        victims' location. The master rescuer updates its map with the
        received one. It does the same for the victims' vital signals.
        After, it should classify each severity of each victim (critical, ..., stable);
        Following, using some clustering method, it should group the victims and
        and pass one (or more)clusters to each rescuer """

        self.received_maps += 1

        print(f"{self.NAME} Map received from the explorer")
        self.map.update(explorer_map)
        self.victims.update(victims)

        if self.received_maps == self.nb_of_explorers:
            print(f"{self.NAME} all maps received from the explorers")
            #self.map.draw()
            #print(f"{self.NAME} found victims by all explorers:\n{self.victims}")

            #@TODO predict the severity and the class of victims' using a classifier
            self.predict_severity_and_class()

            #@TODO cluster the victims possibly using the severity and other criteria
            # Here, there 4 clusters
            clusters_of_vic = self.cluster_victims()

            for i, cluster in enumerate(clusters_of_vic):
                self.save_cluster_csv(cluster, i+1)    # file names start at 1
  
            # Instantiate the other rescuers
            rescuers = [None] * 4
            rescuers[0] = self                    # the master rescuer is the index 0 agent

            # Assign the cluster the master agent is in charge of 
            self.clusters = [clusters_of_vic[0]]  # the first one

            # Instantiate the other rescuers and assign the clusters to them
            for i in range(1, 4):    
                #print(f"{self.NAME} instantianting rescuer {i+1}, {self.get_env()}")
                filename = f"rescuer_{i+1:1d}_config.txt"
                config_file = os.path.join(self.config_folder, filename)
                # each rescuer receives one cluster of victims
                rescuers[i] = Rescuer(self.get_env(), config_file, 4, [clusters_of_vic[i]]) 
                rescuers[i].map = self.map     # each rescuer have the map

            
            # Calculate the sequence of rescue for each agent
            # In this case, each agent has just one cluster and one sequence
            self.sequences = self.clusters         

            # For each rescuer, we calculate the rescue sequence 
            for i, rescuer in enumerate(rescuers):
                rescuer.sequencing()         # the sequencing will reorder the cluster
                
                for j, sequence in enumerate(rescuer.sequences):
                    if j == 0:
                        self.save_sequence_csv(sequence, i+1)              # primeira sequencia do 1o. cluster 1: seq1 
                    else:
                        self.save_sequence_csv(sequence, (i+1)+ j*10)      # demais sequencias do 1o. cluster: seq11, seq12, seq13, ...

            
                rescuer.planner()            # make the plan for the trajectory
                rescuer.set_state(VS.ACTIVE) # from now, the simulator calls the deliberation method 
         
        
    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           print(f"{self.NAME} has finished the plan [ENTER]")
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy = self.plan.pop(0)
        #print(f"{self.NAME} pop dx: {dx} dy: {dy} ")

        # Walk - just one step per deliberation
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            #print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")

            # check if there is a victim at the current position
            if self.map.in_map((self.x, self.y)):
                vic_id = self.map.get_vic_id((self.x, self.y))
                if vic_id != VS.NO_VICTIM:
                    self.first_aid()
                    #if self.first_aid(): # True when rescued
                        #print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")                    
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
            
        return True
    
    def _eucliadian_distance(self, coord1: tuple, coord2: tuple) -> float:
        return math.sqrt( (coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 )
    
    def _sum_euclian_distance_of_the_sequence(self, seq: list) -> float:
        value = 0.0
        previous_position = (0,0)
        for position in seq:
            value += self._eucliadian_distance(position, previous_position)
            previous_position = position
        value += self._eucliadian_distance(seq[-1], (0,0))
        return value
            
    def _scheduler(self, t: float, t_max: float) -> float:
        probabilidade_final = 0.01      # 0%
        probabilidade_inicial = 100     # 100%

        return probabilidade_final + (probabilidade_inicial - probabilidade_final) * ((t_max - t)/t_max) ** 2