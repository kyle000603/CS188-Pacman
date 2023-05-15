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


from hashlib import new
from operator import indexOf
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
        foods = newFood.asList()
        score = 0
        if successorGameState.isWin():
            return 9999999
        for stat in newGhostStates:
            if stat.scaredTimer == 0 and manhattanDistance(stat.getPosition(), newPos) <= 1:
                return -9999999
            if stat.scaredTimer > 0 and manhattanDistance(stat.getPosition(), newPos) <= 1:
                score += 100
        
        if action == 'Stop':
            score -= 50
        food_distance = [manhattanDistance(food, newPos) for food in foods]
        score += float(1/min(food_distance))
        score -= len(foods)

        return score
        return successorGameState.getScore()

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
        agent_num = gameState.getNumAgents()
        func_ev = self.evaluationFunction
        def func(action):return ghost_min(gameState.generateSuccessor(0, action), 1, 1)

        def pac_max(state, depth, agent_idx = 0):
            pac_action = state.getLegalActions(agent_idx)

            if depth == self.depth or not pac_action:
                return func_ev(state)
            
            max_score = max(ghost_min(state.generateSuccessor(agent_idx, action), depth + 1, agent_idx + 1) for action in pac_action)
            return max_score
        def ghost_min(state, depth, agent_idx):
            ghost_action = state.getLegalActions(agent_idx)
            
            if not ghost_action:
                return func_ev(state)
            
            if agent_idx == agent_num - 1:
                return min(pac_max(state.generateSuccessor(agent_idx, action), depth) for action in ghost_action)
            else:
                return min(ghost_min(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1) for action in ghost_action)
        action_queue = util.PriorityQueueWithFunction(func)
        for action in gameState.getLegalActions(0):
            action_queue.push(action)
        while not action_queue.isEmpty():
            minimax_val = action_queue.pop()
        return minimax_val
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agent_num = gameState.getNumAgents()
        func_ev = self.evaluationFunction
        alpha, beta = -999999, 999999

        def func(action):return ghost_min(gameState.generateSuccessor(0, action), 1, 1, alpha, beta)

        def pac_max(state, depth, agent_idx, alpha, beta):
            pac_action = state.getLegalActions(agent_idx)
            max_score = -999999

            if depth == self.depth or not pac_action:
                return func_ev(state)
            
            for action in pac_action:
                max_score = max(max_score,ghost_min(state.generateSuccessor(agent_idx, action), depth + 1, agent_idx + 1, alpha, beta))
                if max_score > beta:
                    return max_score
                else:
                    alpha = max(alpha, max_score)
            return max_score
        def ghost_min(state, depth, agent_idx, alpha, beta):
            ghost_action = state.getLegalActions(agent_idx)
            min_score = 999999

            if not ghost_action:
                return func_ev(state)
            
            if agent_idx == agent_num - 1:
                for action in ghost_action:
                    min_score = min(min_score, pac_max(state.generateSuccessor(agent_idx, action), depth, 0, alpha, beta))
                    if min_score < alpha:
                        return min_score
                    beta = min(beta, min_score)

            else:
                for action in ghost_action:
                    min_score = min(min_score, ghost_min(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1, alpha, beta))
                    if min_score < alpha:
                        return min_score
                    beta = min(beta, min_score)

            return min_score
        
        action_queue = util.PriorityQueue()
        for action in gameState.getLegalActions(0):
            score = func(action)
            action_queue.push(action, score)

            if score > beta:
                return action
            alpha = max(score, alpha)

        while not action_queue.isEmpty():
            minimax_val = action_queue.pop()
        return minimax_val
        util.raiseNotDefined()

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
        agent_num = gameState.getNumAgents()
        func_ev = self.evaluationFunction

        def func(action):return ghost_expect(gameState.generateSuccessor(0, action), 1, 1)

        def pac_max(state, depth, agent_idx = 0):
            pac_action = state.getLegalActions(agent_idx)

            if depth == self.depth or not pac_action:
                return func_ev(state)
            
            max_score = max(ghost_expect(state.generateSuccessor(agent_idx, action), depth + 1, agent_idx + 1) for action in pac_action)
            return max_score

        def ghost_expect(state, depth, agent_idx):
            ghost_action = state.getLegalActions(agent_idx)
            
            if not ghost_action:
                return func_ev(state)
            
            expecti_val = 0
            prob = 1.0/len(ghost_action)

            for action in ghost_action:
                if agent_idx == agent_num - 1:
                    temp = pac_max(state.generateSuccessor(agent_idx, action), depth, 0) * prob
                else:
                    temp = ghost_expect(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1) * prob
                expecti_val += temp
            
            return expecti_val

        
        action_queue = util.PriorityQueueWithFunction(func)
        for action in gameState.getLegalActions(0):
            action_queue.push(action)
        while not action_queue.isEmpty():
            minimax_val = action_queue.pop()
        return minimax_val
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    For this implementation, I considered 4 major factors: Food distance, 
    Food left, Capsule left, and Ghosts. For food distance, I only considered
    the minimum distance of food. So I tried to make a higher score at when closer distance.
    For left food, I considered the number of left food. And I tried to make a lower score
    when more food was left. For Capsules, I think it same as food distance but gave it
    a higher score than food (because it gives a higher score than food). Finally for Ghost,
    I have to consider two situations: when the ghost is scared or not. If it's scared, I
    count them as capsules (which means the same score as capsules and higher than food).
    On the opposite, I only think about the nearby situations (which means it only counts if the ghost
    is a nearby Pacman). If the ghost is nearby Pacman, I returned the lowest score of -999999.    
    """
    "*** YOUR CODE HERE ***"
    foods_inlist = currentGameState.getFood().asList()
    cur_pos = currentGameState.getPacmanPosition()
    capsule_pos = currentGameState.getCapsules()
    total_score = 0
    distance = util.manhattanDistance
    ghost_state_inlist = currentGameState.getGhostStates()

    #Food minimum distance evaluation
    food_list = []
    for food in foods_inlist:
        food_list.append(distance(food, cur_pos))
    if food_list:
        total_score += 1.0/min(food_list)

    #Food left evaluation
    total_score -= len(foods_inlist)

    #Capsule distance evaluation
    capsule_list = []
    if capsule_pos:
        for capsule in capsule_pos:
            capsule_list.append(distance(capsule, cur_pos))
    if capsule_list:
        total_score += 2.0/min(capsule_list)
    
    #Ghost evaluation
    if not ghost_state_inlist:
        return total_score
    ghost_list = []
    for ghost in ghost_state_inlist:
        if ghost.scaredTimer > 0:
            ghost_list.append(distance(ghost.getPosition(), cur_pos))
        else:
            if distance(ghost.getPosition(), cur_pos) <= 1:
                return -99999
    if ghost_list:
        total_score += 2.0/min(ghost_list)
    
    return currentGameState.getScore() + total_score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
