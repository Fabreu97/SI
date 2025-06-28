######################################
# Autor: Fernando Abreu
# Date: 27/06/2025
######################################
# Inspirado no arquivo bfs.py
######################################

import math
import heapq

######################################
# Node
######################################
class Node:
    def __init__(self, position, parent=None):
        self.parent: Node = parent
        self.position = position  # Tupla (x, y)
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.position)

######################################
# ASTAR
######################################
class ASTAR:
    def __init__(self, map, cost_line = 1.0, cost_diag = 1.5):
        self.map = map
        self.cost_line = cost_line
        self.cost_diag = cost_diag
        self.tlim = float('inf')

        self.deltas = [             # the increments for each walk action]
            (0, -1),             #  u: Up
            (1, -1),             # ur: Upper right diagonal
            (1, 0),              #  r: Right
            (1, 1),              # dr: Down right diagonal
            (0, 1),              #  d: Down
            (-1, 1),             # dl: Down left left diagonal
            (-1, 0),             #  l: Left
            (-1, -1)             # ul: Up left diagonal
        ]
    
    def _movements_to_goal(self, current_node: Node):
        movements = []
        current = current_node
        cost = 0.0
        while current.parent is not None:
            pos_current = current.position
            pos_previous = current.parent.position
            
            dx, dy = pos_current[0] - pos_previous[0], pos_current[1] - pos_previous[1]
            movements.append((dx,dy))
            current = current.parent
        if self.tlim > cost:
            return movements[::-1], current_node.g
        return [], -1

    def execute(self, initial_postion, goal_position, tlim = float('inf')):
        if initial_postion == goal_position:
            return [], 0
        
        open_list = []
        closed_list = {}
        self.tlim = tlim

        initial_node    = Node(initial_postion, parent=None)
        initial_node.h  = eucliadian_distance(initial_postion, goal_position)
        initial_node.g  = 0
        initial_node.f  =  initial_node.g + initial_node.h
        heapq.heappush(open_list, initial_node)

        while(len(open_list) > 0):
            current_node: Node = heapq.heappop(open_list)

            if current_node.position in closed_list and closed_list[current_node.position] <= current_node.g:
                continue
            closed_list[current_node.position] = current_node.g

            if current_node.position == goal_position:
                # Verifica o limite de custo antes de reconstruir o caminho
                if current_node.g > self.tlim:
                    return [], -1
                return self._movements_to_goal(current_node) # _movements_to_goal pode ser simplificado

            position = current_node.position
            for index, (dx, dy) in enumerate(self.deltas):
                candidate = (position[0] + dx, position[1] + dy)
                if not self.map.in_map(candidate):
                    continue
                
                if index % 2:
                    cost = self.cost_diag
                else:
                    cost = self.cost_line

                neighbor = Node(candidate, current_node)
                neighbor.parent = current_node
                neighbor.h = eucliadian_distance(candidate, goal_position)
                neighbor.g = current_node.g + cost
                neighbor.f = neighbor.g + neighbor.h

                if candidate in closed_list and closed_list[candidate] <= neighbor.g:
                    continue
                
                heapq.heappush(open_list, neighbor)
        return None, 0 # sem caminho encontrado


######################################
# Algoritmos
######################################

def eucliadian_distance(coord1: tuple, coord2: tuple) -> float:
        return math.sqrt( (coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 )