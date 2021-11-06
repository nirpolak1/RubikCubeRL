# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 21:39:50 2021
Q Value Model
@author: Nir
"""


import keras
import tensorflow as tf
import numpy as np

class QValueModel:
    def __init__(self):
        lr = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(lr)

        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(1028, input_shape = (54,)))
        self.model.add(keras.layers.LeakyReLU())
        self.model.add(keras.layers.Dense(512))
        self.model.add(keras.layers.LeakyReLU())
        self.model.add(keras.layers.Dense(12))
        self.model.compile(optimizer = self.optimizer, loss = 'mean_squared_error', metrics=['accuracy'])
    
    def fitBatch(self, experiences, miniBatches):
        xValues = np.array([exp['state'] for exp in experiences])
        yValues =  np.array([exp['qValues'] for exp in experiences])

        self.model.fit(xValues, yValues, batch_size= miniBatches, epochs = 1)
    
    def trainBatch(self, experiences):
        lossValues = []
        for experience in experiences:
            # Open a GradientTape.
            with tf.GradientTape() as tape:
                # Forward pass.
                logits = self.model(experience['state'])

                # Loss value for this batch.
                loss_value = experience['qValues'] - logits
                loss_value = tf.pow(loss_value, 2)
                lossValues.append(loss_value)
        
        # Get gradients of loss wrt the weights.
        gradients = tape.gradient(lossValues, self.model.trainable_weights)
        
        # Update the weights of the model.
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
    
    def loadModel(self, path):
        self.model = keras.models.load_model(path)
    
    def trainBatchOld(self, experiences):
        for experience in experiences:
            # Open a GradientTape.
            with tf.GradientTape() as tape:
                # Forward pass.
                logits = self.model(experience['state'])

                # Loss value for this batch.
                loss_value = experience['qValues'] - logits
                loss_value = tf.pow(loss_value, 2)
        
            # Get gradients of loss wrt the weights.
            gradients = tape.gradient(loss_value, self.model.trainable_weights)
        
            # Update the weights of the model.
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        
"""
a = QValueModel()
state = np.arange(54).reshape(1,54)
qValues = np.arange(6)

experienceT = []
for i in range(1000):
    experienceT.append({})
    experienceT[i]['state'] = state
    experienceT[i]['qValues'] = qValues


a.trainBatch(experiences = experienceT)
b = a.model.predict(state)
"""