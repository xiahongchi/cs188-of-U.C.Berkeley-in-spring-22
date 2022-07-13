# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    actions = []
    start_state = problem.getStartState()

    s_stack = util.Stack()
    p_stack = util.Stack()
    s_stack.push((start_state,0))
    expanded_states = set([])

    while not s_stack.isEmpty():
        curr_state, flag = s_stack.pop()
        expanded_states.add(curr_state)
        if flag == 0:
            if not p_stack.isEmpty():
                actions.append(p_stack.pop())
            if problem.isGoalState(curr_state):
                break
            s_stack.push((curr_state,1))
            successors = problem.getSuccessors(curr_state)
            for state,action,_ in successors:
                if state not in expanded_states:
                    s_stack.push((state,0))
                    p_stack.push(action)
        if flag == 1:
            if len(actions)>0:
                actions.pop()

    return actions


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    actions = []
    start_state = problem.getStartState()

    queue = util.Queue()
    queue.push(start_state)

    path_dict = {}
    expanded_states = set([])
    expanded_states.add(start_state)

    while not queue.isEmpty():
        curr_state = queue.pop()
        if problem.isGoalState(curr_state):
            while curr_state in path_dict.keys():
                curr_state,action = path_dict[curr_state]
                actions = [action] + actions
            break
        successors = problem.getSuccessors(curr_state)
        for state,action,_ in successors:
            if state not in expanded_states:
                path_dict[state] = (curr_state,action)
                expanded_states.add(state)
                queue.push(state)

    return actions

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    actions = []
    start_state = problem.getStartState()

    queue = util.PriorityQueue()
    queue.push(start_state,0)

    path_dict = {}
    path_dict[start_state] = (None, None, 0)
    expanded_states = set([])
    expanded_states.add(start_state)

    while not queue.isEmpty():
        curr_state = queue.pop()
        _, _, curr_cost = path_dict[curr_state]
        
        if problem.isGoalState(curr_state):
            while curr_state != start_state:
                curr_state,action,_ = path_dict[curr_state]
                actions = [action] + actions
            break
        
        successors = problem.getSuccessors(curr_state)
        for state,action,cost in successors:
            if state not in expanded_states:
                expanded_states.add(state)
                queue.push(state,curr_cost+cost)
                path_dict[state] = (curr_state, action, curr_cost+cost)
            elif path_dict[state][2]>curr_cost+cost:
                path_dict[state] = (curr_state, action, curr_cost+cost)
                queue.update(state,curr_cost+cost)

    return actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    actions = []
    start_state = problem.getStartState()

    queue = util.PriorityQueue()
    queue.push(start_state,heuristic(start_state,problem))

    path_dict = {}
    path_dict[start_state] = (None, None, 0)
    expanded_states = set([])
    expanded_states.add(start_state)

    while not queue.isEmpty():
        curr_state = queue.pop()
        _, _, curr_cost = path_dict[curr_state]
        
        if problem.isGoalState(curr_state):
            while curr_state != start_state:
                curr_state,action,_ = path_dict[curr_state]
                actions = [action] + actions
            break
        
        successors = problem.getSuccessors(curr_state)
        for state,action,cost in successors:
            if state not in expanded_states:
                expanded_states.add(state)
                queue.push(state,curr_cost+cost+heuristic(state,problem))
                path_dict[state] = (curr_state, action, curr_cost+cost)
            elif path_dict[state][2]>curr_cost+cost:
                path_dict[state] = (curr_state, action, curr_cost+cost)
                queue.update(state,curr_cost+cost+heuristic(state,problem))

    return actions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
