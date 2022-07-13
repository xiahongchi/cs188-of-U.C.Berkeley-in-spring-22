# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        evalScore = 0
        x, y = newPos
        foodcnt = 0
        for food in newFood.asList():
            foodx, foody = food
            evalScore += 10 / (abs(x-foodx) + abs(y-foody) + 0.0001)
            foodcnt += 1
        
        for ghost in newGhostStates:
            ghostx, ghosty = ghost.getPosition()
            if ghost.scaredTimer > 0:
                evalScore -= 0.005 / (abs(x-ghostx) + abs(y-ghosty) + 0.0001)
            else:
                evalScore -= 15 / (abs(x-ghostx) + abs(y-ghosty) + 0.0001)

        evalScore += 100/(foodcnt+0.001) + 5 * successorGameState.getScore()

        return evalScore

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        self.dict = {}
        
        self.max_value(1, 0, gameState)
        return self.dict[(1,0)][1]

    
    def max_value(self, roundNum, agentIndex, gameState):
        
        evalScore = -999999.0
        selectAction = None
        
        if roundNum == self.depth + 1:
            evalScore = self.evaluationFunction(gameState)
        else:
            actionNum = 0
            for action in gameState.getLegalActions(agentIndex):
                gameStateSuccessor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    self.max_value(roundNum+1, 0, gameStateSuccessor)
                    currScore, _ = self.dict[(roundNum+1,0)]
                else:
                    self.min_value(roundNum, agentIndex+1, gameStateSuccessor)
                    currScore, _ = self.dict[(roundNum,agentIndex+1)]
                if evalScore < currScore:
                    evalScore = currScore
                    selectAction = action
                actionNum += 1
            if actionNum == 0:
                evalScore = self.evaluationFunction(gameState)
        self.dict[(roundNum,agentIndex)] = (evalScore,selectAction)
        

    def min_value(self, roundNum, agentIndex, gameState):
        
        evalScore = 999999.0
        selectAction = None
        if roundNum == self.depth + 1:
            evalScore = self.evaluationFunction(gameState)
        else:
            actionNum = 0
            for action in gameState.getLegalActions(agentIndex):
                gameStateSuccessor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    self.max_value(roundNum+1, 0, gameStateSuccessor)
                    currScore, _ = self.dict[(roundNum+1,0)]
                else:
                    self.min_value(roundNum, agentIndex+1, gameStateSuccessor)
                    currScore, _ = self.dict[(roundNum,agentIndex+1)]
                if evalScore > currScore:
                    evalScore = currScore
                    selectAction = action
                actionNum += 1
            if actionNum == 0:
                evalScore = self.evaluationFunction(gameState)
        self.dict[(roundNum,agentIndex)] = (evalScore,selectAction)




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.dict = {}
        alpha = -999999.0
        beta = 999999.0
        self.max_value_alpha_beta(1, 0, gameState, alpha, beta)
        return self.dict[(1,0)][1]

    def max_value_alpha_beta(self, roundNum, agentIndex, gameState, alpha, beta):
        
        evalScore = -999999.0
        selectAction = None
        
        if roundNum == self.depth + 1:
            evalScore = self.evaluationFunction(gameState)
        else:
            actionNum = 0
            for action in gameState.getLegalActions(agentIndex):
                gameStateSuccessor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    self.max_value_alpha_beta(roundNum+1, 0, gameStateSuccessor, alpha, beta)
                    currScore, _ = self.dict[(roundNum+1,0)]
                else:
                    self.min_value_alpha_beta(roundNum, agentIndex+1, gameStateSuccessor, alpha, beta)
                    currScore, _ = self.dict[(roundNum,agentIndex+1)]
                if evalScore < currScore:
                    evalScore = currScore
                    selectAction = action
                if evalScore > beta:
                    self.dict[(roundNum,agentIndex)] = (evalScore,selectAction)
                    return
                alpha = max(alpha, evalScore)
                actionNum += 1
            if actionNum == 0:
                evalScore = self.evaluationFunction(gameState)
        self.dict[(roundNum,agentIndex)] = (evalScore,selectAction)
        

    def min_value_alpha_beta(self, roundNum, agentIndex, gameState, alpha, beta):
        
        evalScore = 999999.0
        selectAction = None

        if roundNum == self.depth + 1:
            evalScore = self.evaluationFunction(gameState)
        else:
            actionNum = 0
            for action in gameState.getLegalActions(agentIndex):
                gameStateSuccessor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    self.max_value_alpha_beta(roundNum+1, 0, gameStateSuccessor, alpha, beta)
                    currScore, _ = self.dict[(roundNum+1,0)]
                else:
                    self.min_value_alpha_beta(roundNum, agentIndex+1, gameStateSuccessor, alpha, beta)
                    currScore, _ = self.dict[(roundNum,agentIndex+1)]
                if evalScore > currScore:
                    evalScore = currScore
                    selectAction = action
                if evalScore < alpha:
                    self.dict[(roundNum,agentIndex)] = (evalScore,selectAction)
                    return
                beta = min(beta, evalScore)
                actionNum += 1
            if actionNum == 0:
                evalScore = self.evaluationFunction(gameState)
        self.dict[(roundNum,agentIndex)] = (evalScore,selectAction)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.dict = {}
        self.max_decision(1,0,gameState)
        return self.dict[(1,0)][1]

    def max_decision(self, roundNum, agentIndex, gameState):
        
        evalScore = -999999.0
        selectAction = None

        if roundNum == self.depth + 1:
            evalScore = self.evaluationFunction(gameState)
        else:
            actionNum = 0
            for action in gameState.getLegalActions(agentIndex):
                gameStateSuccessor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    self.max_decision(roundNum+1, 0, gameStateSuccessor)
                    currScore, _ = self.dict[(roundNum+1,0)]
                else:
                    self.chance_decision(roundNum, agentIndex+1, gameStateSuccessor)
                    currScore, _ = self.dict[(roundNum,agentIndex+1)]
                if evalScore < currScore:
                    evalScore = currScore
                    selectAction = action
                actionNum += 1
            if actionNum == 0:
                evalScore = self.evaluationFunction(gameState)
        self.dict[(roundNum,agentIndex)] = (evalScore,selectAction)


    def chance_decision(self, roundNum, agentIndex, gameState):
        
        evalScore = 0.

        if roundNum == self.depth + 1:
            evalScore = self.evaluationFunction(gameState)
        else:
            actionNum = 0
            for action in gameState.getLegalActions(agentIndex):
                gameStateSuccessor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    self.max_decision(roundNum+1, 0, gameStateSuccessor)
                    currScore, _ = self.dict[(roundNum+1,0)]
                else:
                    self.chance_decision(roundNum, agentIndex+1, gameStateSuccessor)
                    currScore, _ = self.dict[(roundNum,agentIndex+1)]
                
                evalScore += currScore
                actionNum += 1
            
            if actionNum == 0:
                evalScore = self.evaluationFunction(gameState)
            else:
                evalScore /= actionNum

        self.dict[(roundNum,agentIndex)] = (evalScore,None)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    food = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    x,y = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    score = currentGameState.getScore()

    evalScore = 0
    epsilon = 0.000001

    for ghostState in ghostStates:
        ghostx, ghosty = ghostState.getPosition()
        if ghostState.scaredTimer > 0:
            evalScore += ghostState.scaredTimer
            evalScore += 100/(abs(x-ghostx)+abs(y-ghosty)+epsilon)
        else:
            evalScore -= 1/(abs(x-ghostx)+abs(y-ghosty)+epsilon)

    for capsule in capsules:
        capsulex, capsuley = capsule
        evalScore += 1/(abs(x-capsulex)+abs(y-capsuley)+epsilon)
    
    for foodx, foody in food.asList():
        evalScore += 5/(abs(x-foodx)+abs(y-foody)+epsilon)

    evalScore += score

    return evalScore

# Abbreviation
better = betterEvaluationFunction
