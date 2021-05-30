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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        newScore = successorGameState.getScore()
        "*** YOUR CODE HERE ***"

        value = newScore
        ghostsDistnace = []
        for ghost in newGhostStates:
            position = ghost.getPosition()
            ghostsDistnace.append(manhattanDistance(newPos,position))       #calculate the distances from ghosts

        if ghostsDistnace[0] > 0:
            value -= 17.0 / ghostsDistnace[0]           #if the ghost is near move away

        foodDistance = []
        flag=0
        for food in newFood.asList():
            foodDistance.append(manhattanDistance(newPos, food))
            flag=1

        if flag==1:
            foodDistance.sort()
            min = foodDistance.pop(0)
            value += 11.0 / min

        return value

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        """
        def minimax(agentIndex, depth, gameState):
            if not gameState.getLegalActions(0) or depth == self.depth:  # utility
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                v = []
                for action in gameState.getLegalActions(0):                       #find the best action
                    v.append((minimax(1, depth, gameState.generateSuccessor(0, action))) )
                return max(v)
            else:
                if gameState.getNumAgents() == agentIndex + 1:
                    v = []
                    for action in gameState.getLegalActions(agentIndex):           #find the best action
                        v.append((minimax(0, depth + 1, gameState.generateSuccessor(agentIndex, action)) )) #increase depth
                    return min(v)
                else:
                    v = []
                    for action in gameState.getLegalActions(agentIndex):
                        v.append((minimax(agentIndex + 1, depth, gameState.generateSuccessor(agentIndex, action)) ))
                    return min(v)

        actions = []
        for agentState in gameState.getLegalActions(0):
            actions.append((minimax(1, 0, gameState.generateSuccessor(0, agentState)),agentState))
        return max(actions,key = lambda item:item[0])[1]   #find max based on first item of tuple


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def minimax(agentIndex, depth, gameState, a, b):
            if not gameState.getLegalActions(0) or depth == self.depth:  # utility
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                v = []
                for action in gameState.getLegalActions(0):                       #find the best action
                    temp = minimax(1, depth, gameState.generateSuccessor(0, action), a, b)
                    v.append(temp)
                    if(temp > b):
                        return max(v)           #pruning
                    if(temp > a):
                        a = temp
                return max(v)
            else:
                if gameState.getNumAgents() == agentIndex + 1:
                    v = []
                    for action in gameState.getLegalActions(agentIndex):           #find the best action
                        temp = minimax(0, depth + 1, gameState.generateSuccessor(agentIndex, action), a, b)
                        v.append(temp)
                        if(temp < a):
                            return min(v)       #pruning
                        if(temp < b):
                            b = temp
                    return min(v)

                else:
                    v = []
                    for action in gameState.getLegalActions(agentIndex):
                        temp = minimax(agentIndex + 1, depth, gameState.generateSuccessor(agentIndex, action), a, b)
                        v.append(temp)
                        if(temp < a):
                            return min(v)       #pruning
                        if(temp < b):
                            b = temp
                    return min(v)

        actions = []
        a = -(float("inf"))
        b = + (float("inf"))
        for agentState in gameState.getLegalActions(0):
            score = (minimax(1, 0, gameState.generateSuccessor(0, agentState),a,b))
            if score > b:
                return max(actions,key = lambda item:item[0])[1]   #find max based on first item of tuple #pruning
            if score > a:
                a = score
            actions.append((score,agentState))
        return max(actions,key = lambda item:item[0])[1]   #find max based on first item of tuple

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        def minimax(agentIndex, depth, gameState):
            if not gameState.getLegalActions(0) or depth == self.depth:  # utility
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                v = []
                for action in gameState.getLegalActions(0):                       #find the best action
                    v.append((minimax(1, depth, gameState.generateSuccessor(0, action))) )
                return max(v)
            else:
                if gameState.getNumAgents() == agentIndex + 1:
                    v = []
                    for action in gameState.getLegalActions(agentIndex):           #find the best action
                        v.append((minimax(0, depth + 1, gameState.generateSuccessor(agentIndex, action)) )) #increase depth
                    return sum(v)/len(v)
                else:
                    v = []
                    for action in gameState.getLegalActions(agentIndex):
                        v.append((minimax(agentIndex + 1, depth, gameState.generateSuccessor(agentIndex, action)) ))
                    return sum(v)/len(v)

        actions = []
        for agentState in gameState.getLegalActions(0):
            actions.append((minimax(1, 0, gameState.generateSuccessor(0, agentState)),agentState))
        return max(actions,key = lambda item:item[0])[1]   #find max based on first item of tuple


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    newScore = currentGameState.getScore()

    "*** YOUR CODE HERE ***"

    value = newScore
    ghostsDistnace = []
    for ghost in newGhostStates:
        position = ghost.getPosition()
        ghostsDistnace.append(manhattanDistance(newPos, position))
        if newScaredTimes[newGhostStates.index(ghost)] > manhattanDistance(newPos, position):
            value += 20.0 / manhattanDistance(newPos, position)     #so pacman can reach the ghost

    if ghostsDistnace[0] > 0:
        value -= 17.0 / ghostsDistnace[0]               #if the ghost is near move away

    foodDistance = []
    flag=0
    for food in newFood.asList():
        foodDistance.append(manhattanDistance(newPos, food))
        flag=1

    if flag==1:
        foodDistance.sort()
        min = foodDistance.pop(0)
        value += 11.0 / min

    return value

# Abbreviation
better = betterEvaluationFunction
