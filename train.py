# Import libraries
import tensorflow as tf
import numpy as np


# Input Data for training
# Numpy.random.randint function generates random numbers across 0 to 90
# of 0.1 Million samples
# Flatten will transform the generated matrix to a linear form
train_X = np.random.randint(0,90,(100000,1))
train_Y = train_X.flatten()

# Data for Testing
# Testing data is used to cross check the accuracy of the model we're trainig
test_X = np.random.randint(0,90,(10000,1))
test_Y = test_X.flatten()

# Machine Learning Model

"""
Keras are built on top of Tensorflow
Sequential is for stacking of layers that is defined here
Layers are the topmost level of abstraction or class in Keras,
having exactly one input and one output
Dense is used to get dotproduct of the values.
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1024, activation='sigmoid'),
    tf.keras.layers.Dense(512, activation='sigmoid'),
    tf.keras.layers.Dense(256, activation='sigmoid'),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(91, activation=tf.nn.softmax)])


# Configure the model for training using Accuracy as a metrics
model.compile(optimizer=tf.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# Trains the model with a fixed number of epochs
model.fit(train_X, train_Y, epochs=10)
print(model.evaluate(test_X,test_Y))


while True:
    age_val = input("Enter the age : ")
    prediction = model.predict([int(age_val)])
    print("The predicted age is ", np.argmax(prediction[0]))