# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 11:57:51 2021
Q-Learning Process
@author: Nir
"""

import RubikCube as rc
import numpy as np
import random
import QValueModel as qvm
import json
from collections import deque

#Global parameters
epsilon = 0.1
gamma = 0.4
stepReward = 1
terminalReward = 0
scrambleNum = 5

MEMORY_SIZE = 1000000 
BATCH_SIZE = 20
memory = deque(maxlen=MEMORY_SIZE)

QVModel = qvm.QValueModel()
QVModel.loadModel('savedModel2')

def ReplayBatch(batchSize, maxSteps):
    experienceBatch = []
    
    #play a batch of game with a max number of steps
    for batch in range(batchSize):
        stepCount = 0
        lastAction = 0
        repeatingCount = 0
        #scramble for new game
        rubikCube = rc.RubikCube(3)
        rubikCube.scramble(random.randint(1, scrambleNum))
        rubikCube.calculateCompletion()
        
        #try solving the cube
        while (stepCount < maxSteps and rubikCube.completionPercentage < 1.00):
            #save current state
            currentState = rubikCube.positioning.reshape(-1,54).copy()
            
            #choose action
            if (random.random() < epsilon):
                actionIndex = random.randint(0, 11)
            else:
                actionIndex = np.argmax(QVModel.model.predict(currentState))
            
            
            #refrain from repeating
            if actionIndex % 6 == lastAction % 6:
                repeatingCount += 1
            else:
                repeatingCount = 0
            if repeatingCount > 3:
                newAction = random.randint(0, 11)
                if actionIndex == newAction and newAction > 0:
                    newAction -= 1
                elif actionIndex == newAction and newAction == 0:
                    newAction += 1
                actionIndex = newAction
                repeatingCount = 0
            lastAction = actionIndex
            
            #interpret action
            playDirection = int(actionIndex > 5)
            playRest = actionIndex % 6
            playAxis = playRest % 3
            playSlice = 2 * (playRest % 2)
            rubikCube.rotate(playAxis, playSlice, playDirection)
            
            #calculate reward
            playDelta = rubikCube.calculateCompletion()
            
            if (rubikCube.completionPercentage > 0.999):
                #cube completed
                reward = stepReward * 10
                print('...reached completion while training')
            elif stepCount == maxSteps - 1:
                reward = 0
                print('...reached max step with completion %', rubikCube.completionPercentage)
            else:
                if (playDelta > 0):
                    reward = stepReward
                else:
                    reward = 0
            
            #save experience 
            experienceBatch.append({'s' : currentState, 
                                    'a' : actionIndex,
                                    'r' : reward,
                                    'sTag': rubikCube.positioning.reshape(-1,54)})
            stepCount += 1      
    return experienceBatch
    

def ValueExperience(experiences):
    refinedExperinces = []
    
    for exp in experiences:
        sStateQ = QVModel.model.predict(exp['s'])
        sTagStateQ = QVModel.model.predict(exp['sTag'])
        
        if exp['r'] == terminalReward:
            sStateQ[0][exp['a']] = terminalReward
        else:
            sStateQ[0][exp['a']] = exp['r'] + gamma * np.argmax(sTagStateQ)
        
        refinedExperinces.append({'state' : np.array(exp['s']).reshape(-1, 54),
                                  'qValues' : np.array(sStateQ).reshape(-1, 12)})
        
        memory.append({'state' : np.array(exp['s']).reshape(-1, 54),
                                  'qValues' : np.array(sStateQ).reshape(-1, 12)})
        
    return refinedExperinces

def TrainModelByBatch(batchSize, maxSteps):
    print('Playing cubes...')
    experiences = ReplayBatch(batchSize, maxSteps)
    print('Refining experiences...')
    refinedExp = ValueExperience(experiences)
    if len(memory) < BATCH_SIZE:
            return
    print('Training Batch...')
    QVModel.fitBatch(random.sample(memory, min(len(memory), 50000)), BATCH_SIZE)
    #QVModel.trainBatch(refinedExp)
    
def TrainModel(batches ,batchSize, maxSteps):
    for i in range(batches):
        print('Training batch number ', i)
        TrainModelByBatch(batchSize, maxSteps)
        rc = PlayWithModel(500, False)
        print('Completion reached after 50 steps: ', rc.completionPercentage)
        QVModel.model.save('savedModel2')
    return QVModel.model
    
def PlayWithModel(maxSteps, visual):
    #scramble for new game
    rubikCube = rc.RubikCube(3)
    rubikCube.scramble(scrambleNum)
    rubikCube.calculateCompletion()
    print('Starting completion % at: ', rubikCube.completionPercentage)
    stepCount = 0
    maxPercentage = 0
    lastAction = 0
    repeatingCount = 0
    maxAt = 0
    
    
    if (visual):
        rubikCube.visualizeCube()
    while (stepCount < maxSteps and rubikCube.completionPercentage < 1.00):
        #save current state
        currentState = rubikCube.positioning.reshape(-1,54)
        
        #choose action
        qValues = QVModel.model.predict(currentState)
        #print(qValues)
        actionIndex = np.argmax(qValues)
        
        #refrain from repeating
        if actionIndex % 6 == lastAction % 6:
            repeatingCount += 1
        else:
                repeatingCount = 0
        if repeatingCount > 3:
            newAction = random.randint(0, 11)
            if actionIndex == newAction and newAction > 0:
                newAction -= 1
            elif actionIndex == newAction and newAction == 0:
                newAction += 1
            actionIndex = newAction
            repeatingCount = 0
        lastAction = actionIndex

        #interpret action
        playDirection = int(actionIndex > 5)
        playRest = actionIndex % 6
        playAxis = playRest % 3
        playSlice = 2 * (playRest % 2)
        rubikCube.rotate(playAxis, playSlice, playDirection)
        

        #calculate reward
        rubikCube.calculateCompletion()
        if rubikCube.completionPercentage > maxPercentage:
            maxPercentage = rubikCube.completionPercentage
            maxAt = stepCount
        if (visual):
            rubikCube.visualizeCube()
        stepCount += 1
    print('Max completion % reached: ', maxPercentage, ' - after turn ', maxAt)
    return rubikCube



#a = TrainModel(100000, 10, 100)
b = PlayWithModel(20, True)