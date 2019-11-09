import gym
import random
import numpy as np

EPISODES = 2001
wins = 0
GAMMA=0.99
ALPHA=0.1
EPSILON_MIN = 0.0001
EPSILON_DECAY = 0.99
EPSILON = 1

env = gym.make('Taxi-v3')
model = np.zeros([env.observation_space.n, env.action_space.n])
for x in range(env.observation_space.n):
        for action in range(env.action_space.n):
                model[x, action] = 0

epi_reward = np.zeros([EPISODES])
epi_reward_average = np.zeros([EPISODES])
env.reset()
for episode in range(EPISODES):
    state = env.reset()
    done = False
    if np.random.rand() <= EPSILON:
        action = random.randrange(4)
    else:
        action = np.argmax(model[state, :])

    while not done:
        if episode == 2000:
            env.render()
        new_state, R, done,_ = env.step(action)
        if np.random.rand() <= EPSILON:
            new_action = random.randrange(4)
        else:
            new_action = np.argmax(model[new_state, :])
        model[state][action]+=ALPHA*(R+GAMMA*np.amax(model[new_state][:])-model[state][action])
        state=new_state
        action=new_action
        epi_reward[episode] += R

    EPSILON*=EPSILON_DECAY
    if EPSILON<EPSILON_MIN:
        EPSILON=EPSILON_MIN
    if episode>0:
        epi_reward_average[episode] = np.mean(epi_reward[max(episode - 20, 0):episode])
    if epi_reward[episode] > 0:
        wins+=1

    if episode%200==0:
        print('Episode: ',episode,' Mean Reward: ',epi_reward_average[episode],' Reward: ',epi_reward[episode], ' Wins: ',wins, 'Randomness: ', EPSILON)
