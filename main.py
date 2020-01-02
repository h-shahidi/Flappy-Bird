#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import json
from keras import initializers
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.models import clone_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
from copy import deepcopy

import os
import datetime # for logging timestamp

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.2 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
TARGET_UPDATE = 4000
BOOTSTRAP_K = 10 # number of bootstrap heads

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)

def buildmodel(bootstrap_head = None):
    if bootstrap_head:
        print("Now we build the model for bootstrap_head = %d" % (bootstrap_head))
    else:
        print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))
   
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model

def startIteration(model, args):
    log_file_name = datetime.datetime.now().strftime("log_%Y_%m_%d_%H_%M_%S.txt")
    log_file = open(log_file_name, "w")
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)

    # store N_t(a)
    Nt = np.zeros(ACTIONS)
    
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    Nt[0] += 1
    x_t, r_0, terminal, curr_score = game_state.frame_step(do_nothing)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #shape(1,80,80,4) 

    #Create target network
    if args['training_algorithm'] == "doubleDQN":
        target_model = clone_model(model)
        target_model.set_weights(model.get_weights())

    if args['mode'] == 'run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weights")
        if args['training_algorithm'] in ("bootstrappedDQN", "bootstrappedDQN+UCB"):
            for i in range(BOOTSTRAP_K):
                if os.path.isfile("model_%d.h5" % (i)):
                    model[i].load_weights("model_%d.h5" % (i))
                    print ("Weight for head %d load successfully", (i))
        else:
            if os.path.isfile("model.h5"):              
                model.load_weights("model.h5")
                print ("Weight load successfully")
    else:                       #We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

    t = 0
    total_reward = 0
    while (True):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t1 = 0
        a_t = np.zeros([ACTIONS])
        
        if t % FRAME_PER_ACTION == 0:
            if args['training_algorithm'] in ("bootstrappedDQN", "bootstrappedDQN+UCB"):
                chosen = np.random.randint(BOOTSTRAP_K)
                q = model[chosen].predict(s_t)
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[action_index] = 1
            else:
                #choose an action epsilon greedy
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(ACTIONS)
                    a_t[action_index] = 1
                else:
                    q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                    max_Q = np.argmax(q)
                    action_index = max_Q
                    a_t[action_index] = 1
            Nt[action_index] += 1

        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        x_t1_colored, r_t1, terminal, curr_score = game_state.frame_step(a_t)
        terminal_check = terminal

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        #calculate bootstrap mask
        #the authors use Bernoulli(0.5), but that essentially means
        #choose with 0.5 probability on each head
        mask = np.random.choice(2, BOOTSTRAP_K, p=[0.5,]*2)

        # store the transition in D
        if args['training_algorithm'] in ("bootstrappedDQN", "bootstrappedDQN+UCB"):
            D.append((s_t, action_index, r_t1, s_t1, terminal, mask))
        else:
            D.append((s_t, action_index, r_t1, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if t > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #shape(32, 80, 80, 4)
            print (inputs.shape)
            targets = np.zeros((inputs.shape[0], ACTIONS))                         #shape(32, 2)

            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                if args['training_algorithm'] in ("bootstrappedDQN", "bootstrappedDQN+UCB"):
                    mask = minibatch[i][5]
                

                inputs[i:i + 1] = state_t  #I saved down s_t
                #Hitting each buttom probability
                if args['training_algorithm'] in ("bootstrappedDQN", "bootstrappedDQN+UCB"):
                    targets[i] = model[chosen].predict(state_t)
                else:
                    targets[i] = model.predict(state_t) 


                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    if args['training_algorithm'] == "DQN":
                        Q_sa = model.predict(state_t1)
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
                    elif args['training_algorithm'] == "doubleDQN":
                        Q_sa = model.predict(state_t1)
                        Q_target = target_model.predict(state_t1)
                        maxQ_ind = np.argmax(Q_sa,axis = 1)
                        targets[i, action_t] = reward_t + GAMMA * Q_target[0][maxQ_ind]
                    elif args['training_algorithm'] == "DQN+UCB":
                        Q_sa = model.predict(state_t1)
                        modified_Q_sa = Q_sa+np.sqrt(2*np.log(t)/(Nt))
                        targets[i, action_t] = reward_t + GAMMA * np.max(modified_Q_sa)
                    elif args['training_algorithm'] == "bootstrappedDQN":
                        Q_sa = model[chosen].predict(state_t1)
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
                    elif args['training_algorithm'] == "bootstrappedDQN+UCB":
                        Q_sa = model[chosen].predict(state_t1)
                        modified_Q_sa = Q_sa+np.sqrt(2*np.log(t)/(Nt))
                        targets[i, action_t] = reward_t + GAMMA * np.max(modified_Q_sa)

            if args['training_algorithm'] in ("bootstrappedDQN", "bootstrappedDQN+UCB"):
                for idx in range(BOOTSTRAP_K):
                    if mask[idx] == 1:
                        loss += model[idx].train_on_batch(inputs, targets)
            else:
                loss += model.train_on_batch(inputs, targets)

        s_t = s_t1
        t = t + 1

        if args['training_algorithm'] == "doubleDQN" and t % TARGET_UPDATE == 0 :
            print("----------------------------Copy to target model----------------------------")
            target_model.set_weights(model.get_weights())

        # save progress every 10000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            if args['training_algorithm'] in ("bootstrappedDQN", "bootstrappedDQN+UCB"):
                for i in range(BOOTSTRAP_K):
                    model[i].save_weights("model_%d.h5" % (i), overwrite=True)
                    with open("model_%d.json" % (i), "w") as outfile:
                        json.dump(models[i].to_json(), outfile)
            else:              
                model.save_weights("model.h5", overwrite=True)
                with open("model.json", "w") as outfile:
                    json.dump(model.to_json(), outfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        printInfo(t, state, action_index, r_t1, Q_sa, loss)

        score_file = open("scores","a") 
        score_file.write(str(curr_score)+"\n")
        score_file.close()

        if terminal_check:
            print("Total rewards: ", total_reward) 
            out_file = open("total_reward","a") 
            out_file.write(str(total_reward)+"\n")
            out_file.close()
            total_reward = 0
        else:
            total_reward = total_reward + r_t1

    print("Episode finished!")
    print("************************")

def printInfo(t, state, action_index, r_t1, Q_sa, loss):
    print("TIMESTEP", t, "/ STATE", state, \
          "/ ACTION", action_index, "/ REWARD", r_t1, \
          "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

def playGame(args):
    if args['training_algorithm'] in ("bootstrappedDQN", "bootstrappedDQN+UCB"):
        model = [buildmodel(i) for i in range(BOOTSTRAP_K)]
    else:
        model = buildmodel()
    startIteration(model, args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='train / run', required=True)
    parser.add_argument('--training_algorithm', help='What training algorithm to be used for training (e.g., DQN, doubleDQN)', required=True)
    args = vars(parser.parse_args())
    assert args['mode'] in ('run', 'train')
    assert args['training_algorithm'] in ('DQN',
                                          'doubleDQN',
                                          'DQN+UCB',
                                          'bootstrappedDQN', 
                                          'bootstrappedDQN+UCB')
    playGame(args)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()
