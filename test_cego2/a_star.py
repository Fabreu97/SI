import math
from vs.constants import VS

######################################
# Classes
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
# Algoritmos
######################################

def eucliadian_distance(self, coord1: tuple, coord2: tuple) -> float:
        return math.sqrt( (coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 )

def movements_to_goal(current_node: Node):
    movements = []
    current = current_node
    while current.parent is not None:
        pos_current = current.position
        pos_previous = current.parent.position
        
        dx, dy = pos_current[0] - pos_previous[0], pos_current[1] - pos_previous[1]
        movements.append((dx,dy))
    return movements[::-1]