# trainPerceptron.py
# -----------------
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


# This file contains feature extraction methods and harness
# code for data classification

import sys
import util
from pacman import Directions
from game import Directions, Actions
# from perceptronPacman import PerceptronPacman
import samples
import numpy as np
import math
import pandas as pd

TRAINING_SET_SIZE = 25000
TEST_SET_SIZE = 20000
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70

# a bit quick and dirty but we use feature names in alphabetical order to make sure they are always the same order in the numpy feature vector.
# We use this here, in perceptronPacman and in the Q3 agent
FEATURE_NAMES =[ 
    'closestFood', 
    'closestFoodNow',
    'closestGhost',
    'closestGhostNow',
    'closestScaredGhost',
    'closestScaredGhostNow',
    'eatenByGhost',
    'eatsCapsule',
    'eatsFood',
    "foodCount",
    'foodWithinFiveSpaces',
    'foodWithinNineSpaces',
    'foodWithinThreeSpaces',  
    'furthestFood', 
    'numberAvailableActions',
    "ratioCapsuleDistance",
    "ratioFoodDistance",
    "ratioGhostDistance",
    "ratioScaredGhostDistance"
    ]

def walls(state):
    # Returns a list of (x, y) pairs of wall positions
    #
    # This version just returns all the current wall locations
    # extracted from the state data.  In later versions, this will be
    # restricted by distance, and include some uncertainty.

    wallList = []
    wallGrid = state.getWalls()
    width = wallGrid.width
    height = wallGrid.height
    for i in range(width):
        for j in range(height):
            if wallGrid[i][j] == True:
                wallList.append((i, j))
    return wallList


def inFront(object, facing, state):
    # Returns true if the object is along the corridor in the
    # direction of the parameter "facing" before a wall gets in the
    # way.

    pacman = state.getPacmanPosition()
    pacman_x = pacman[0]
    pacman_y = pacman[1]
    wallList = walls(state)

    # If Pacman is facing North
    if facing == Directions.NORTH:
        # Check if the object is anywhere due North of Pacman before a
        # wall intervenes.
        next = (pacman_x, pacman_y + 1)
        while not next in wallList:
            if next == object:
                return True
            else:
                next = (pacman_x, next[1] + 1)
        return False

    # If Pacman is facing South
    if facing == Directions.SOUTH:
        # Check if the object is anywhere due North of Pacman before a
        # wall intervenes.
        next = (pacman_x, pacman_y - 1)
        while not next in wallList:
            if next == object:
                return True
            else:
                next = (pacman_x, next[1] - 1)
        return False

    # If Pacman is facing East
    if facing == Directions.EAST:
        # Check if the object is anywhere due East of Pacman before a
        # wall intervenes.
        next = (pacman_x + 1, pacman_y)
        while not next in wallList:
            if next == object:
                return True
            else:
                next = (next[0] + 1, pacman_y)
        return False

    # If Pacman is facing West
    if facing == Directions.WEST:
        # Check if the object is anywhere due West of Pacman before a
        # wall intervenes.
        next = (pacman_x - 1, pacman_y)
        while not next in wallList:
            if next == object:
                return True
            else:
                next = (next[0] - 1, pacman_y)
        return False



def StringNameToNumber(numberString):

    if numberString == Directions.NORTH:
        return 0
    elif numberString == Directions.EAST:
        return 1
    elif numberString == Directions.SOUTH:
        return 2
    elif numberString == Directions.WEST:
        return 3
    elif numberString == Directions.STOP:
        return 4
    

"Feature extractors for Pacman game states"
class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features


def basicFeatureExtractorPacman(state):
    """
    A basic feature extraction function.
    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions
    ##
    """
    features = util.Counter()
    for action in state.getLegalActions():
        successor = state.generateSuccessor(0, action)
        foodCount = successor.getFood().count()
        featureCounter = util.Counter()
        featureCounter['foodCount'] = foodCount
        features[action] = featureCounter

    return features, state.getLegalActions()


def enhancedFeatureExtractorPacman(state):
    """
    Your feature extraction playground.
    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions
    ##
    """

    features = basicFeatureExtractorPacman(state)[0]
    for action in state.getLegalActions():
        features[action] = util.Counter(features[action], **furtherEnhancedPacmanFeatures(state, action))
    return features, state.getLegalActions()


def furtherEnhancedPacmanFeatures(state, action):
    """
    For each state, this function is called with each legal action.
    It should return a counter with { <feature name> : <feature value>, ... }
    """
    features = util.Counter()
    successor = state.generateSuccessor(0, action)  # Generate successor for current state
    # currentGhostPosition = [ghost.getPosition() for ghost in state.getGhostStates()]

    pacmanPosition = successor.getPacmanPosition()
    pacmanPositonCurrent = state.getPacmanPosition()
    foodList = successor.getFood().asList()  # Make list of food
    capsuleList = successor.getCapsules()
    ghostPosition = [ghost.getPosition() for ghost in successor.getGhostStates() if ghost.scaredTimer == 0]  # Find ghost positions
    scaredGhostPosition = [ghost.getPosition() for ghost in successor.getGhostStates() if ghost.scaredTimer > 0]
    possibleActions = successor.getLegalActions()

    max_dist = successor.data.layout.height * successor.data.layout.width

    # this only happens when we win the game
    if not foodList:
        # set to 0 because closer food is good
        closestFood = 0
        closestFoodNow = 1  # if we won the game then we just ate the last food and so must have been 1 away from it previously
        furthestFood = 0
        foodWithinThreeSpaces = 0
        foodWithinFiveSpaces = 0
        foodWithinNineSpaces = 0
    else:  # Find minimum distance to food
        foodDistances = np.array([util.manhattanDistance(i, pacmanPosition) for i in foodList])
        closestFoodNow = min([util.manhattanDistance(i, pacmanPositonCurrent) for i in foodList])
        
        # if we ate a food then set the distance to 0. Otherwise distance might increase which makes eating a food look bad.
        if state.getNumFood() - successor.getNumFood() == 1:
            closestFood = 0
        else:
            closestFood = min(foodDistances)
        furthestFood = max(foodDistances)
        foodWithinThreeSpaces = len(foodDistances[foodDistances <= 3])
        foodWithinFiveSpaces = len(foodDistances[foodDistances <= 5])
        foodWithinNineSpaces = len(foodDistances[foodDistances <= 9])

    if not ghostPosition:
        closestGhostNextState = max_dist
        closestGhostNow = max_dist
    else:
        closestGhostNextState = min(util.manhattanDistance(i, pacmanPosition) for i in ghostPosition)
        closestGhostNow = min(util.manhattanDistance(i, pacmanPositonCurrent) for i in ghostPosition)

    if not scaredGhostPosition:
        closestScaredGhost = max_dist
        closestScaredGhostNow = max_dist
    else:
        closestScaredGhost = min(util.manhattanDistance(i, pacmanPosition) for i in scaredGhostPosition)
        closestScaredGhostNow = min(util.manhattanDistance(i, pacmanPositonCurrent) for i in scaredGhostPosition)

    if not capsuleList:
        closestCapsule = 0
        closestCapsuleNow = 0 # set to 0 because closer capsule is good
    else:  # Find minimum distance to food
        closestCapsule = min([util.manhattanDistance(i, pacmanPosition) for i in capsuleList])
        closestCapsuleNow = min([util.manhattanDistance(i, pacmanPositonCurrent) for i in capsuleList])
    
    # weird things happen very rarely with the scared ghosts which makes this 0. This is never 0 because pacman is eaten
    # I don't know what causes it and it's so rare that for now it's just safest to set this to 1 if it happens.
    if closestGhostNow != 0:
        features["ratioGhostDistance"] = closestGhostNextState / closestGhostNow
    else:
        features["ratioGhostDistance"] = 1
    if closestScaredGhostNow != 0: 
        features["ratioScaredGhostDistance"] = closestScaredGhost / closestScaredGhostNow
    else:
        features["ratioScaredGhostDistance"] = 1
    
    # these features improve when you add the one both denominator and numerator
    features["ratioCapsuleDistance"] = (1 + closestCapsule) / (1 + closestCapsuleNow)
    features["ratioFoodDistance"] = (1 + closestFood) / (1 + closestFoodNow)

    # All features listed with reasoning
    features["foodCount"] = successor.getFood().count()  # We want to decrease food
    features['closestFood'] = 10 / (1+closestFood)  # Pacman wants to move towards food
    features['closestFoodNow'] = 10 / (1+closestFoodNow)
    features["furthestFood"] = furthestFood  # Pacman wants to move towards food 
    features['closestGhost'] = closestGhostNextState  # Pacman wants to stay away from ghosts
    features['closestGhostNow'] = closestGhostNow
    features["closestScaredGhost"] = closestScaredGhost  # Pacman wants to capture scared ghosts
    features['closestScaredGhostNow'] = closestScaredGhostNow
    features["closestCapsule"] = closestCapsule   # Pacman needs to eat capsule to capture scared ghosts
    features["closestCapsuleNow"] = closestCapsuleNow   # Pacman needs to eat capsule to capture scared ghosts
    features["eatsFood"] = (state.getNumFood() - successor.getNumFood()) + np.random.normal(loc=0, scale=0.25) # Pacman wants to eat food
    features["eatenByGhost"] = successor.isLose()
    features["eatsCapsule"] = len(state.getCapsules()) - len(successor.getCapsules())  # Pacman wants to eat capsules
    features["foodWithinThreeSpaces"] = foodWithinThreeSpaces  # Pacman wants to move towards areas with lots of food
    features["foodWithinFiveSpaces"] = foodWithinFiveSpaces # Pacman wants to move towards areas with lots of food
    features["foodWithinNineSpaces"] = foodWithinNineSpaces # Pacman wants to move towards areas with lots of food
    features["numberAvailableActions"] = len(possibleActions) # Pacman wants to move towards areas with lots of food
    
    return features


def default(str):
    return str + ' [Default: %default]'


USAGE_STRING = """
  USAGE:      python featureExtraction.py <options>
  EXAMPLES:  (1) python featureExtraction.py
                  - would extract the features from all training examples
                  using the furtherEnhancedPacmanFeatures function. 
                  This would be broken into train test and validation sets and the features
                  scaled based on the max and min values to be between 0 and 1.
             (2) python featureExtraction.py -t 1000
                  - would extract the features from the first 1000 training examples
                  using the furtherEnhancedPacmanFeatures function. 
                  This would be broken into train test and validation sets and the features
                  scaled based on the max and min values to be between 0 and 1.
                 """


def readCommand(argv):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-t', '--training', help=default('The size of the training set'), default=TRAINING_SET_SIZE, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print("Extracting Features")
    print("--------------------")

    if options.training <= 0:
        print("Training set size should be a positive integer (you provided: %d)" % options.training)
        print(USAGE_STRING)
        sys.exit(2)

    if options.test <= 0:
        print("Testing set size should be a positive integer (you provided: %d)" % options.test)
        print(USAGE_STRING)
        sys.exit(2)

    args['training_size'] = options.training
    args['testing_size'] = options.test

    print(args)
    print(options)

    return args, options


# Dictionary containing full path to .pkl file that contains the agent's training, validation, and testing data.
MAP_AGENT_TO_PATH_OF_SAVED_GAMES = {
    'FoodAgent': ('pacmandata/food_training.pkl', 'pacmandata/food_validation.pkl', 'pacmandata/food_test.pkl'),
    'StopAgent': ('pacmandata/stop_training.pkl', 'pacmandata/stop_validation.pkl', 'pacmandata/stop_test.pkl'),
    'SuicideAgent': (
    'pacmandata/suicide_training.pkl', 'pacmandata/suicide_validation.pkl', 'pacmandata/suicide_test.pkl'),
    'GoodReflexAgent': (
    'pacmandata/good_reflex_training.pkl', 'pacmandata/good_reflex_validation.pkl', 'pacmandata/good_reflex_test.pkl'),
    'ContestAgent': (
    'pacmandata/contest_training.pkl', 'pacmandata/contest_validation.pkl', 'pacmandata/contest_test.pkl')
}


def normalise_data(data, max_values, min_values):
    for feature_vector in data:
        for feature_name, value in feature_vector.items():
            feature_vector[feature_name] = (feature_vector[feature_name] - min_values[feature_name])/(max_values[feature_name] - min_values[feature_name])

    return data


def find_max_and_min_feature_values(data):
    feature_names = data[0].keys()
    max_values = dict(zip(feature_names, [-1*math.inf]*len(feature_names)))
    min_values = dict(zip(feature_names, [math.inf]*len(feature_names)))

    for feature_vector in data:
        for feature_name, value in feature_vector.items():
            if max_values[feature_name] < value:
                max_values[feature_name] = value
            if min_values[feature_name] > value:
                min_values[feature_name] = value

    return max_values, min_values


def convertToBinary(features, labels):
    binary_labels = []
    for i in range(len(features)):
        # print(labels[i])
        temp = [1 if StringNameToNumber(move) == labels[i] else 0 for move in features[i][1]]
        binary_labels.extend(temp)

    return binary_labels


def to_numpy_binary_data(trainingData):
    binary_features = []
    for i in range(len(trainingData)):
        legal_moves = trainingData[i][1]

        for move in legal_moves:
            features = trainingData[i][0][move]

            binary_features.append([features[feature_name] for feature_name in FEATURE_NAMES])
            # binary_features.append(features)

    return np.array(binary_features)


def runExtraction(args):
    featureFunction = enhancedFeatureExtractorPacman

    # data sizes and number of training iterations
    numTraining = args['training_size']
    numTest = args['testing_size']

    # load the data sets
    print("loading data...")
    trainingData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][0]
    validationData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][1]
    testData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][2]
    rawTrainingData, trainingLabels = samples.loadPacmanData(trainingData, numTraining)
    rawValidationData, validationLabels = samples.loadPacmanData(validationData, numTest)
    rawTestData, testLabels = samples.loadPacmanData(testData, numTest)

    # Extract features
    print("Extracting features...")

    trainingData = list(map(featureFunction, rawTrainingData))[:-1]
    validationData = list(map(featureFunction, rawValidationData))[:-1]
    testData = list(map(featureFunction, rawTestData))[:-1]

    trainingLabels = list(map(StringNameToNumber, trainingLabels[:-1]))
    validationLabels = list(map(StringNameToNumber, validationLabels[:-1]))
    testLabels = list(map(StringNameToNumber, testLabels[:-1]))

    # convert the data to binary labels format as numpy arrays
    trainingLabels = convertToBinary(trainingData, trainingLabels)
    trainingData = to_numpy_binary_data(trainingData)

    validationLabels = convertToBinary(validationData, validationLabels)
    validationData = to_numpy_binary_data(validationData)

    testLabels = convertToBinary(testData, testLabels)
    testData = to_numpy_binary_data(testData)

    # find the minimum and maximum values across all data sets for scaling
    max_train = np.max(trainingData, axis=0)
    max_validation = np.max(validationData, axis=0)
    max_test = np.max(testData, axis=0)
    max_values = np.max(np.vstack([max_train, max_validation, max_test]), axis=0)

    min_train = np.min(trainingData, axis=0)
    min_validation = np.min(validationData, axis=0)
    min_test = np.min(testData, axis=0)
    min_values = np.min(np.vstack([min_train, min_validation, min_test]), axis=0)

    # put a one at the front for bias max and 0 for bias min.
    max_values = np.concatenate([[1], np.max(trainingData, axis=0)])
    min_values = np.concatenate([[0], np.min(trainingData, axis=0)])

    # add bias to each data set
    trainingData = np.c_[np.ones(len(trainingData)), trainingData]
    validationData = np.c_[np.ones(len(validationData)), validationData]
    testData = np.c_[np.ones(len(testData)), testData]

    trainingData = (trainingData - min_values)/(max_values - min_values)
    validationData = (validationData - min_values)/(max_values - min_values)
    testData = (testData - min_values)/(max_values - min_values)

    df_train = pd.DataFrame(data=trainingData, columns=["bias"] + FEATURE_NAMES)
    df_train["label"] = trainingLabels
    df_train.to_csv("./pacmandata/q3_train.csv", index=False)

    df_validation = pd.DataFrame(data=validationData, columns=["bias"] + FEATURE_NAMES)
    df_validation["label"] = validationLabels
    df_validation.to_csv("./pacmandata/q3_validation.csv", index=False)

    df_test = pd.DataFrame(data=testData, columns=["bias"] + FEATURE_NAMES)
    df_test["label"] = testLabels
    df_test.to_csv("./pacmandata/q3_test.csv", index=False)

    np.savetxt("./pacmandata/q3_max_feature_values.txt", max_values)
    np.savetxt("./pacmandata/q3_min_feature_values.txt", min_values)


if __name__ == '__main__':
    # Read input
    args, options = readCommand(sys.argv[1:])
    # Run classifier
    runExtraction(args)