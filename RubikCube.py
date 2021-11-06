# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 14:45:46 2021
Rubik Cube Class
@author: Nir
"""

import numpy as np
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
import random

class RubikCube:
    def __init__(self, size):
        #define cube edge size
        self.size = size
    
        #Defining the positioning of colors (in this case represented with int) around the cube
        #0 - south, 1 - east, 2 - north, 3 - west, 4 - up, 5 - down
        self.positioning = np.array([np.zeros(shape=(size,size)),
                                     np.zeros(shape=(size,size)),
                                     np.zeros(shape=(size,size)),
                                     np.zeros(shape=(size,size)),
                                     np.zeros(shape=(size,size)),
                                     np.zeros(shape=(size,size))])
        self.positioning[0].fill(1)
        self.positioning[1].fill(2)
        self.positioning[2].fill(3)
        self.positioning[3].fill(4)
        self.positioning[4].fill(5)
        self.positioning[5].fill(6)
        
        #define colors
        self.colors = {1 : [0, 0.608, 0.282],
                       2 : [1, 1, 1],
                       3 : [0.717, 0.071, 0.204],
                       4 : [1, 0.835, 0],
                       5 : [0, 0.274, 0.678],
                       6 : [1, 0.345, 0],}
        
        #define percentage to completion 
        self.completionPercentage = 1.00
        
        #define change towards completion 
        self.stepDelta = 0.00
        
    #make a certain rotation
    #at this time, ONLY for 3x3 cube
    def rotate(self, axis, sliceIndex, direction):
        #rotate counter-clockwise 90 degrees
        spins = 1
        if direction > 0:
            spins = 3
            
        for _ in range(spins):
        #around up-down axis
            if axis == 0: 
                if (sliceIndex <= 1):
                    self.rotateFace(4)
                    self.swapSlices([0, 3, 2, 1], 0, axis)
                if (sliceIndex >= 1):
                    self.rotateFace(5)
                    self.swapSlices([0, 3, 2, 1], 2, axis)
                    
            #around west-east axis
            elif axis == 1: 
                if (sliceIndex <= 1):
                    self.rotateFace(3)
                    self.swapSlices([0, 5, 2, 4], 0, axis)
                if (sliceIndex >= 1):
                    self.rotateFace(1)
                    self.swapSlices([0, 5, 2, 4], 2, axis)
    
            #around south-north axis
            elif axis == 2: 
                if (sliceIndex <= 1):
                    self.rotateFace(0)
                    self.swapSlices([1, 4, 3, 5], 0, axis)
                if (sliceIndex >= 1):
                    self.rotateFace(2)
                    self.swapSlices([1, 4, 3, 5], 2, axis)
    
    #rotate a full face 
    def rotateFace(self, faceIndex):
        self.positioning[faceIndex] = np.rot90(self.positioning[faceIndex])
        
    #rotate a slice
    def swapSlices(self, facesInd, sliceInd, axis):
        if axis == 0:
            for j in range(self.size):
                tempFirst = self.positioning[facesInd[(0) % len(facesInd)], sliceInd, j].copy()
                nextInd = j
                for i in range(len(facesInd)):
                    if i % 2 == 0:
                        nextInd = self.size - 1 - nextInd
                       
                    tempNext = self.positioning[facesInd[(i + 1) % len(facesInd)], sliceInd, nextInd].copy()
                    self.positioning[facesInd[(i + 1) % len(facesInd)], sliceInd, nextInd] = tempFirst
                    tempFirst = tempNext
            return

        elif axis == 1:
            for j in range(self.size):
                tempFirst = self.positioning[facesInd[(0) % len(facesInd)], j, sliceInd].copy()
                nextInd = j
                for i in range(len(facesInd)):
                    if i % 2 != 0:
                        nextInd = self.size - 1 - nextInd
                    
                    tempNext = self.positioning[facesInd[(i + 1) % len(facesInd)], nextInd, sliceInd].copy()
                    self.positioning[facesInd[(i + 1) % len(facesInd)], nextInd, sliceInd] = tempFirst
                    tempFirst = tempNext
            return

        elif axis == 2:
            for j in range(self.size):
                tempFirst = self.positioning[facesInd[(0) % len(facesInd)], j, sliceInd].copy()
                nextInd = j
                for i in range(len(facesInd)):
                    if i % 2 != 0:
                        nextInd = self.size - 1 - nextInd
                    
                    if facesInd[(i + 1) % len(facesInd)] > 3:
                        tempNext = self.positioning[facesInd[(i + 1) % len(facesInd)], sliceInd, nextInd].copy()
                        self.positioning[facesInd[(i + 1) % len(facesInd)], sliceInd, nextInd] = tempFirst
                    else:
                        tempNext = self.positioning[facesInd[(i + 1) % len(facesInd)], nextInd, sliceInd].copy()
                        self.positioning[facesInd[(i + 1) % len(facesInd)], nextInd, sliceInd] = tempFirst
                        
                    tempFirst = tempNext
            return
     
    def visualizeCube(self):
        plotFigure = a3.Axes3D(pl.figure())
        dotSize = 0.2
        plotFigure.set_xlim3d(0, dotSize * self.size)
        plotFigure.set_ylim3d(0, dotSize * self.size)
        plotFigure.set_zlim3d(-dotSize * self.size, 0)

        dotSize = 0.2
        
        for i in range(6):
            for j in range(self.size):
                for k in range(self.size):
                    color = self.colors[self.positioning[i,j,k]]
                    if i == 0:  

                        polyPoints = np.array([[k * dotSize, 0, -j * dotSize],
                                               [k * dotSize + dotSize, 0, -j * dotSize],
                                               [k * dotSize + dotSize, 0, -(j * dotSize + dotSize)],
                                               [k * dotSize, 0, -(j * dotSize + dotSize)]])
                    elif i == 1:  
                        polyPoints = np.array([[self.size * dotSize, k * dotSize, -j * dotSize],
                                               [self.size * dotSize, k * dotSize + dotSize, -j * dotSize],
                                               [self.size * dotSize, k * dotSize + dotSize, -(j * dotSize + dotSize)],
                                               [self.size * dotSize, k * dotSize, -(j * dotSize + dotSize)]])
                    elif i == 2:  
                        polyPoints = np.array([[k * dotSize, self.size * dotSize, -j * dotSize],
                                               [k * dotSize + dotSize, self.size * dotSize, -j * dotSize],
                                               [k * dotSize + dotSize, self.size * dotSize, -(j * dotSize + dotSize)],
                                               [k * dotSize, self.size * dotSize, -(j * dotSize + dotSize)]])
                    elif i == 3:  
                        polyPoints = np.array([[0, k * dotSize, -j * dotSize],
                                               [0, k * dotSize + dotSize, -j * dotSize],
                                               [0, k * dotSize + dotSize,  -(j * dotSize + dotSize)],
                                               [0, k * dotSize,  -(j * dotSize + dotSize)]])
                    elif i == 4:  
                        polyPoints = np.array([[k * dotSize, j * dotSize, 0],
                                               [k * dotSize, j * dotSize + dotSize, 0],
                                               [k * dotSize + dotSize, j * dotSize + dotSize, 0],
                                               [k * dotSize + dotSize, j * dotSize, 0]])
                    elif i == 5:  
                        polyPoints = np.array([[k * dotSize, j * dotSize, -dotSize * self.size],
                                               [k * dotSize, j * dotSize + dotSize, -dotSize * self.size],
                                               [k * dotSize + dotSize, j * dotSize + dotSize, -dotSize * self.size],
                                               [k * dotSize + dotSize, j * dotSize, -dotSize * self.size]])
                        
                        
                    polygon = a3.art3d.Poly3DCollection([polyPoints])
                    polygon.set_color(colors.rgb2hex(color))
                    polygon.set_alpha(0.8)
                    polygon.set_edgecolor('k')
                    plotFigure.add_collection3d(polygon)
                        
    def scramble(self, iterations):
        for i in range(iterations):
            direction = random.randint(0,1)
            axis = random.randint(0,2)
            sliceInd = random.randint(0,1) * 2
            self.rotate(axis, sliceInd, direction)
             
    def calculateCompletion(self):
        completion = 0
        for i in range(6):
            completion += np.where(self.positioning[i] == i + 1, 1, 0).sum()
        completion = completion / 54
        self.stepDelta = completion - self.completionPercentage
        self.completionPercentage = completion
        return self.stepDelta

    




