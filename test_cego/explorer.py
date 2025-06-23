# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

import sys
import os
import random
import math
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map

victims_found = {}

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Explorer(AbstAgent):
    """ class attribute """
    MAX_DIFFICULTY = 1             # the maximum degree of difficulty to enter into a cell
    def __init__(self, env, config_file, resc, constant):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.walk_stack = Stack()  # a stack to store the movements
        self.walk_time = 0         # time consumed to walk when exploring (to decide when to come back)
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals
        self.visit_count = {}      # Quantidade de vezes que a posição foi visitada
        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
        self.visit_count[(0,0)] = 0

        # Get a random direction
        if self.NAME[-1] == '1':
            self.init_direction = (1,-1) # cima direita
        elif(self.NAME[-1] == '2'):
            self.init_direction = (1,1) # baixo direita
        elif(self.NAME[-1] == '3'):
            self.init_direction = (-1,1) # baixo esquerda
        elif(self.NAME[-1] == '4'):
            self.init_direction = (-1,-1) # cima esquerda
        
        self.A = constant[0]
        self.B = constant[1]
        self.C = constant[2]
        self.D = constant[3]
        self.E = constant[4]
        self.F = constant[5]
        self.G = constant[6]
        global victims_found
        victims_found.clear()
        self.delta = None
        self.greater_distance = 1.0
        self.max_visit = 1

    def _euclidean_distance(self, coord: tuple) -> float:
        return math.sqrt(coord[0]**2 + coord[1]**2)

    def _correct_direction(self, coord: tuple) -> int:
        if ( (coord[0] * self.init_direction[0] > 0) and (coord[1] * self.init_direction[1]) > 0):
            return 1
        return 0
    
    def _correct_quadrant(self, coord: tuple) -> int:
        if ( (coord[0] * self.init_direction[0] > 0) and (coord[1] * self.init_direction[1]) > 0):
            return 1
        elif(coord[0] == 0 and (coord[1] * self.init_direction[1]) > 0):
            return self.G # vizinho
        elif(coord[1] == 0 and (coord[0] * self.init_direction[0]) > 0):
            return self.G # vizinho
        return 0

    

    def get_next_position(self):

        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()
        # Loop until a CLEAR position is found

        """
            Algoritmo para ir para próxima posição determinado pela função abaixo:
            valor = A * visit_count[coord + direction] + B * euclidian_distance[coord]
                    + C * objective_direction
        """
        value = {}
        for i, delta in Explorer.AC_INCR.items():
            if obstacles[i] == VS.CLEAR:
                dx, dy = delta
                future_position = (self.x + dx, self.y + dy)
                if not self.map.in_map(future_position):
                    visit_count = 0
                    unknown = 1
                else:
                    visit_count = self.visit_count[future_position]
                    unknown = 0
                distance = self._euclidean_distance(future_position)
                if distance > self.greater_distance:
                    self.greater_distance = distance
                distance = distance / self.greater_distance
                objective_direction = self._correct_direction(delta)
                correct_quadrant = self._correct_quadrant(future_position)
                if self.delta is None:
                    back = 0
                else:
                    if (self.delta + 4) % 8 == i:
                        back = 1
                    else:
                        back = 0
                value[i] = self.A * visit_count/self.max_visit + self.B * distance + self.C * objective_direction + self.D * unknown + self.E * back + self.F + correct_quadrant
        direction = max(value, key=value.get)
        self.delta = direction
        position = (Explorer.AC_INCR[direction][0] + self.x, Explorer.AC_INCR[direction][1] + self.y) 
        if position in self.visit_count:
            self.visit_count[position] += 1
            if self.visit_count[position] > self.max_visit:
                self.max_visit = self.visit_count[position]
        else:
            self.visit_count[position] = 1
        return Explorer.AC_INCR[direction]
        
    def explore(self):
        # get an random increment for x and y       
        dx, dy = self.get_next_position()

        # Moves the body to another position  
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()


        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            #print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            self.walk_stack.push((dx, dy))

            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy

            # update the walk time
            self.walk_time = self.walk_time + (rtime_bef - rtime_aft)
            #print(f"{self.NAME} walk time: {self.walk_time}")

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                global victims_found
                if (self.x, self.y) not in victims_found:
                    victims_found[(self.x, self.y)] = 1
                #print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                #print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            
            # Calculates the difficulty of the visited cell 
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            #print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return

    def come_back(self):
        dx, dy = self.walk_stack.pop()
        dx = dx * -1
        dy = dy * -1

        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            #print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
        
    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        # forth and back: go, read the vital signals and come back to the position

        time_tolerance = 0.5 * self.TLIM #3 * self.COST_DIAG * Explorer.MAX_DIFFICULTY + self.COST_READ

        # keeps exploring while there is enough time
        if  self.walk_time < (self.get_rtime() - time_tolerance):
            self.explore()
            return True

        # no more come back walk actions to execute or already at base
        if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
            # time to pass the map and found victims to the master rescuer
            self.resc.sync_explorers(self.map, self.victims)
            # finishes the execution of this agent
            return False
        
        # proceed to the base
        self.come_back()
        return True

