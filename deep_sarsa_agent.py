import copy
import pylab
import random
import numpy as np
from environment import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 1000 # number of training rounds


# this is DeepSARSA Agent for the GridWorld
# Utilize Neural Network as q function approximator
class DeepSARSAgent:
    def __init__(self):
        self.load_model = False
        # actions which agent can do
        self.action_space = [0, 1, 2, 3, 4]
        # get size of state and action
        self.action_size = len(self.action_space)
        self.state_size = 15 # Is used in NN creation for setting the input size
        # If
        self.discount_factor = 0.99 # used in calculating reward of a choosen next action
        self.learning_rate = 0.001 # used in setting up the optimiser

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .955 # was .9999, too high in our testing
        self.epsilon_min = 0.01 
        self.model = self.build_model()

        # Has the ability to load weights but it is hardcoded to be false (line 17)
        if self.load_model:
            self.epsilon = 0.05
            self.model.load_weights('./save_model/deep_sarsa_trained.h5')

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu')) # Hidden layer 1 uses a Rectified Linear Unit
        model.add(Dense(30, activation='relu')) # Hidden layer 2 also uses a Rectified Linear Unit
        model.add(Dense(self.action_size, activation='linear')) # Output layer
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate)) # We can loss = 'mse' or Mean Squared Error
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.action_size)
        else:
            # Predict the reward value based on the given state
            state = np.float32(state) #not exactly sure what this does? Maybe makes it digestable for the neural network
            
            #THIS IS WHERE THE NEURAL NETWORK REPLACES THE Q-VALUES
            q_values = self.model.predict(state)
            
            return np.argmax(q_values[0]) # return the highest value choice

    def train_model(self, state, action, reward, next_state, next_action, done):
        # Lower epsilon (in higher than min)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state = np.float32(state) #not exactly sure what this does? Maybe makes it digestable for the neural network
        next_state = np.float32(next_state) #not exactly sure what this does? Maybe makes it digestable for the neural network
        
        target = self.model.predict(state)[0]
        # like Q Learning, get maximum Q value at s'
        # But from target model
        if done:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor *
                              self.model.predict(next_state)[0][next_action])

        target = np.reshape(target, [1, 5])
        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(state, target, epochs=1, verbose=0)


if __name__ == "__main__":
    env = Env() # The environment
    agent = DeepSARSAgent() # The agent in the environment

    global_step = 0
    scores, episodes = [], [] # For keeping track of scores and episodes for later printing

    for e in range(EPISODES):
        done = False #mark episode as not done
        score = 0 #reset score
        state = env.reset() # set environment to default state
        state = np.reshape(state, [1, 15])

        while not done:
            # fresh env
            global_step += 1

            # get action for the current state and go one step in environment
            action = agent.get_action(state) # Gets best action by looking at state
            next_state, reward, done = env.step(action) # Increment state
            next_state = np.reshape(next_state, [1, 15])
            next_action = agent.get_action(next_state) #ask agent which action it wants to do

            # It does training with every step... After determining its next action
            agent.train_model(state, action, reward, next_state, next_action, done) 
           
            state = next_state #set current state to the state with the action that the agent chose
            # every time step we do training
            score += reward

            state = copy.deepcopy(next_state) #not sure what this does

            #When finished with round...
            if done:
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b') 
                pylab.savefig("./save_graph/deep_sarsa_.png")
                print("episode:", e, "  score:", score, "global_step",
                      global_step, "  epsilon:", agent.epsilon) # In terms of feedback, this is currently all we get in the terminal

        #every 100 episodes, save the weights
        if e % 100 == 0:
            agent.model.save_weights("./save_model/deep_sarsa.h5")
