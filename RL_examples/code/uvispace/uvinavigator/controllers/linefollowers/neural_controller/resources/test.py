from DQNagent import  Agent
import numpy as np
import sys
sys.path.append('F:\\Javier\\Desktop\\TFM\\uvispace-main-controller')
from plot_ugv import PlotUgv
from environment import UgvEnv
import math
from collections import deque
import matplotlib.pyplot as plt
from training import Training

#x_trajectory = np.append(np.linspace(0.2, 0.2, 41),
#                         np.cos(np.linspace(0 * math.pi / 180, 90 * math.pi / 180, 61)) * 0.1 + 0.1)
#y_trajectory = np.append(np.linspace(0.2, 0.4, 41),
#                         np.sin(np.linspace(0 * math.pi / 180, 90 * math.pi / 180, 61)) * 0.1 + 0.4)
#x_trajectory = np.append(x_trajectory,
#                         np.cos(np.linspace(270 * math.pi / 180, 180 * math.pi / 180, 81)) * 0.2 + 0.1)
#y_trajectory = np.append(y_trajectory,
#                         np.sin(np.linspace(270 * math.pi / 180, 180 * math.pi / 180, 81)) * 0.2 + 0.7)
#x_trajectory = np.append(x_trajectory,
#                         np.cos(np.linspace(0 * math.pi / 180, 180 * math.pi / 180, 141)) * 0.3 - 0.4)
#y_trajectory = np.append(y_trajectory,
#                         np.sin(np.linspace(0 * math.pi / 180, 180 * math.pi / 180, 141)) * 0.3 + 0.7)
#x_trajectory = np.append(x_trajectory, np.linspace(-0.7,-0.7, 81))
#y_trajectory = np.append(y_trajectory, np.linspace(0.7, 0.3, 81))
#x_trajectory = np.append(x_trajectory,
#                         np.cos(np.linspace(180 * math.pi / 180, 270 * math.pi / 180, 81)) * 0.3 -0.4)
#y_trajectory = np.append(y_trajectory,
#                         np.sin(np.linspace(180 * math.pi / 180, 270 * math.pi / 180, 81)) * 0.3 + 0.3)
#x_trajectory = np.append(x_trajectory, np.linspace(-0.4, 0, 81))
#y_trajectory = np.append(y_trajectory, np.linspace(0, 0, 81))
#x_trajectory = np.append(x_trajectory,
#                         np.cos(np.linspace(270 * math.pi / 180, 360 * math.pi / 180, 81)) * 0.2 + 0)
#y_trajectory = np.append(y_trajectory,
#                         np.sin(np.linspace(270 * math.pi / 180, 360 * math.pi / 180, 81)) * 0.2 + 0.2)



#x_trajectory = np.append(np.linspace(0.2, 0.2, 41),
#                         np.cos(np.linspace(180 * math.pi / 180, 90 * math.pi / 180, 61)) * 0.1 + 0.3)
#y_trajectory = np.append(np.linspace(0.2, 0.4, 41),
#                         np.sin(np.linspace(180 * math.pi / 180, 90 * math.pi / 180, 61)) * 0.1 + 0.4)
#x_trajectory = np.append(x_trajectory,
#                         np.cos(np.linspace(270 * math.pi / 180, 360 * math.pi / 180, 81)) * 0.2 + 0.3)
#y_trajectory = np.append(y_trajectory,
#                         np.sin(np.linspace(270 * math.pi / 180, 360 * math.pi / 180, 81)) * 0.2 + 0.7)
#x_trajectory = np.append(x_trajectory,
#                         np.cos(np.linspace(180 * math.pi / 180, 0 * math.pi / 180, 141)) * 0.3 + 0.8)
#y_trajectory = np.append(y_trajectory,
#                         np.sin(np.linspace(180 * math.pi / 180, 0 * math.pi / 180, 141)) * 0.3 + 0.7)
#x_trajectory = np.append(x_trajectory, np.linspace(1.1, 1.1, 81))
#y_trajectory = np.append(y_trajectory, np.linspace(0.7, 0.3, 81))
#x_trajectory = np.append(x_trajectory,
#                         np.cos(np.linspace(0 * math.pi / 180, -90 * math.pi / 180, 81)) * 0.3 + 0.8)
#y_trajectory = np.append(y_trajectory,
#                         np.sin(np.linspace(0 * math.pi / 180, -90 * math.pi / 180, 81)) * 0.3 + 0.3)
#x_trajectory = np.append(x_trajectory, np.linspace(0.8, 0.4, 81))
#y_trajectory = np.append(y_trajectory, np.linspace(0, 0, 81))
#x_trajectory = np.append(x_trajectory,
#                         np.cos(np.linspace(270 * math.pi / 180, 180 * math.pi / 180, 81)) * 0.2 + 0.4)
#y_trajectory = np.append(y_trajectory,
#                         np.sin(np.linspace(270 * math.pi / 180, 180 * math.pi / 180, 81)) * 0.2 + 0.2)


#
#

x_trajectory = np.append(np.linspace(0.2, 0.2, 41),
                        np.cos(np.linspace(180 * math.pi / 180, 90 * math.pi / 180, 61)) * 0.1 + 0.3)
y_trajectory = np.append(np.linspace(0.2, 0.4, 41),
                        np.sin(np.linspace(180 * math.pi / 180, 90 * math.pi / 180, 61)) * 0.1 + 0.4)
x_trajectory = np.append(x_trajectory,
                        np.cos(np.linspace(270 * math.pi / 180, 360 * math.pi / 180, 81)) * 0.2 + 0.3)
y_trajectory = np.append(y_trajectory,
                        np.sin(np.linspace(270 * math.pi / 180, 360 * math.pi / 180, 81)) * 0.2 + 0.7)
x_trajectory = np.append(x_trajectory,
                        np.cos(np.linspace(180 * math.pi / 180, -90 * math.pi / 180, 141)) * 0.3 + 0.8)
y_trajectory = np.append(y_trajectory,
                        np.sin(np.linspace(180 * math.pi / 180, -90 * math.pi / 180, 141)) * 0.3 + 0.7)
x_trajectory = np.append(x_trajectory,
                        np.cos(np.linspace(90 * math.pi / 180, 180 * math.pi / 180, 61)) * 0.1 + 0.8)
y_trajectory = np.append(y_trajectory,
                        np.sin(np.linspace(90 * math.pi / 180, 180 * math.pi / 180, 61)) * 0.1 + 0.3)
x_trajectory = np.append(x_trajectory,
                        np.cos(np.linspace(360 * math.pi / 180, 270 * math.pi / 180, 61)) * 0.3 + 0.4)
y_trajectory = np.append(y_trajectory,
                        np.sin(np.linspace(360 * math.pi / 180, 270 * math.pi / 180, 61)) * 0.3 + 0.3)
x_trajectory = np.append(x_trajectory,
                        np.cos(np.linspace(270 * math.pi / 180, 180 * math.pi / 180, 81)) * 0.2 + 0.4)
y_trajectory = np.append(y_trajectory,
                         np.sin(np.linspace(270 * math.pi / 180, 180 * math.pi / 180, 81)) * 0.2 + 0.2)





#x_trajectory = np.append(np.linspace(0.2, 0.2, 121),
#                        np.linspace(0.2001,0.25, 10))
#y_trajectory = np.append(np.linspace(0.2, 0.8, 121),
#                       np.linspace(0.8,0.8, 10))
#x_trajectory = np.append(x_trajectory,
#                        np.linspace(0.25,0.25, 361))
#y_trajectory = np.append(y_trajectory,
#                       np.linspace(0.79999,-1, 361))
#

tr=Training()
tr.testing(load_name='second-training.h5', x_trajectory=x_trajectory,y_trajectory=y_trajectory,closed=False)
#
tr=Training()
tr.trainclosedcircuit(load=False,load_name='first-training.h5',save_name='second-training.h5',reward_need=180)

tr=Training()
tr.testing(load_name='second-training.h5', x_trajectory=x_trajectory,y_trajectory=y_trajectory,closed=False)
#tr.traincurve(save_name='first-training.h5')
###tr.trainline(save_name='test-2.h5')
###tr.traincircle(save_name='2-64.h5')
###tr.trainopendcircuit(load=True, load_name='first-training.h5',save_name='second-training.h5')
##tr.trainopendcircuit(load=True,load_name='2-32.h5',save_name='2-32.h5')

#SPACE_X = 4
#SPACE_Y = 3
#PERIOD= 1/30
#NUM_DIV_ACTION = 5
#INIT_TO_ZERO = True
#EPISODES = 500
#state_size = 2
#action_size = 5 * 5
#x_trajectory = np.append(np.linspace(0.2, 0.2, 41),
#                        np.cos(np.linspace(180 * math.pi / 180, 90 * math.pi / 180, 61)) * 0.1 + 0.3)
#y_trajectory = np.append(np.linspace(0.2, 0.4, 41),
#                         np.sin(np.linspace(180 * math.pi / 180, 90 * math.pi / 180, 61)) * 0.1 + 0.4)
#x_trajectory = np.append(x_trajectory,
#                         np.cos(np.linspace(270 * math.pi / 180, 360 * math.pi / 180, 81)) * 0.2 + 0.3)
#y_trajectory = np.append(y_trajectory,
#                         np.sin(np.linspace(270 * math.pi / 180, 360 * math.pi / 180, 81)) * 0.2 + 0.7)
#x_trajectory = np.append(x_trajectory,
#                         np.cos(np.linspace(180 * math.pi / 180, -90 * math.pi / 180, 141)) * 0.3 + 0.8)
#y_trajectory = np.append(y_trajectory,
#                         np.sin(np.linspace(180 * math.pi / 180, -90 * math.pi / 180, 141)) * 0.3 + 0.7)
#x_trajectory = np.append(x_trajectory,
#                         np.cos(np.linspace(90 * math.pi / 180, 180 * math.pi / 180, 61)) * 0.1 + 0.8)
#y_trajectory = np.append(y_trajectory,
#                         np.sin(np.linspace(90 * math.pi / 180, 180 * math.pi / 180, 61)) * 0.1 + 0.3)
#x_trajectory = np.append(x_trajectory,
#                         np.cos(np.linspace(360 * math.pi / 180, 270 * math.pi / 180, 61)) * 0.3 + 0.4)
#y_trajectory = np.append(y_trajectory,
#                         np.sin(np.linspace(360 * math.pi / 180, 270 * math.pi / 180, 61)) * 0.3 + 0.3)
#x_trajectory = np.append(x_trajectory,
#                         np.cos(np.linspace(270 * math.pi / 180, 180 * math.pi / 180, 81)) * 0.2 + 0.4)
#y_trajectory = np.append(y_trajectory,
#                         np.sin(np.linspace(270 * math.pi / 180, 180 * math.pi / 180, 81)) * 0.2 + 0.2)
#reward_need = (len(x_trajectory) // 10) * 5 + 15
#print(reward_need)
#for ganma in [0.99, 0.995]:
#    for epsilon_dec in[0.92]:
#        for lr in [0.008]:
#            for batch in [32]:
#                for tau in [0.01]:
#
#                    scores = deque(maxlen=5)
#                    agent = Agent(state_size, action_size, gamma=ganma, epsilon=1, epsilon_min=0.01, epsilon_decay=epsilon_dec,
#                                  learning_rate=lr, batch_size=batch, tau=tau)
#                    plot_ugv = PlotUgv(SPACE_X, SPACE_Y, x_trajectory, y_trajectory, PERIOD)
#                    env = UgvEnv(x_trajectory, y_trajectory, PERIOD,
#                                 NUM_DIV_ACTION, closed=False)
#
#                    for e in range(EPISODES):
#                        state, agent_state=env.reset()
#                        agent_state=agent.format_state(agent_state)
#                        done=False
#                        R=0
#
#                        while not done:
#                            action = agent.action(agent_state)
#                            new_state, new_agent_state, reward, done =env.step(action)
#                            new_agent_state = agent.format_state(new_agent_state)
#                            agent.remember(agent_state, action, reward, new_agent_state, done)
#
#                            agent_state=new_agent_state
#                            R+=reward
#                        if len(agent.memory) > agent.batch_size:
#                            for i in range(500):
#                                agent.replay()
#                                agent.soft_update_target_network()
#
#
#                        agent.reduce_random()
#                        scores.append(R)
#                        mean_score = np.mean(scores)
#                        print("episode: {}/{}, score: {}, e: {:.2}, mean_score: {}, final state :({},{})"
#                                  .format(e, EPISODES, R, agent.epsilon, mean_score,env.state[0],env.state[1]))
#
#                        if mean_score > reward_need:
#                            agent.save_model('2-32.h5')
#                            f = open('parameters.txt', 'a')
#                            f.write(
#                                'Episodes: {}, gamma: {}, epsilon_dec: {}, lr: {}, batch: {}, tau:{}\n'.format(e, ganma,
#                                                                                                               epsilon_dec,
#                                                                                                               lr,
#                                                                                                               batch,
#                                                                                                               tau))
#                            f.close()
#                            break
#





#SPACE_X = 4
#SPACE_Y = 3
#PERIOD= 1/30
#NUM_DIV_ACTION = 5
#INIT_TO_ZERO = True
#EPISODES = 500
#state_size = 2
#action_size = 5 * 5
#x_trajectory = np.append(np.linspace(0.2, 0.2, 41),
#                                 np.cos(np.linspace(180 * math.pi / 180, 90 * math.pi / 180, 61)) * 0.1 + 0.3)
#y_trajectory = np.append(np.linspace(0.2, 0.4, 41),
#                         np.sin(np.linspace(180 * math.pi / 180, 90 * math.pi / 180, 61)) * 0.1 + 0.4)
#x_trajectory = np.append(x_trajectory,
#                         np.cos(np.linspace(270 * math.pi / 180, 360 * math.pi / 180, 81)) * 0.2 + 0.3)
#y_trajectory = np.append(y_trajectory,
#                         np.sin(np.linspace(270 * math.pi / 180, 360 * math.pi / 180, 81)) * 0.2 + 0.7)
#x_trajectory = np.append(x_trajectory,
#                         np.cos(np.linspace(180 * math.pi / 180, 0 * math.pi / 180, 141)) * 0.3 + 0.8)
#y_trajectory = np.append(y_trajectory,
#                         np.sin(np.linspace(180 * math.pi / 180, 0 * math.pi / 180, 141)) * 0.3 + 0.7)
#x_trajectory = np.append(x_trajectory, np.linspace(1.1, 1.1, 81))
#y_trajectory = np.append(y_trajectory, np.linspace(0.7, 0.3, 81))
#x_trajectory = np.append(x_trajectory,
#                         np.cos(np.linspace(0 * math.pi / 180, -90 * math.pi / 180, 81)) * 0.3 + 0.8)
#y_trajectory = np.append(y_trajectory,
#                         np.sin(np.linspace(0 * math.pi / 180, -90 * math.pi / 180, 81)) * 0.3 + 0.3)
#x_trajectory = np.append(x_trajectory, np.linspace(0.8, 0.4, 81))
#y_trajectory = np.append(y_trajectory, np.linspace(0, 0, 81))
#x_trajectory = np.append(x_trajectory,
#                         np.cos(np.linspace(270 * math.pi / 180, 180 * math.pi / 180, 81)) * 0.2 + 0.4)
#y_trajectory = np.append(y_trajectory,
#                         np.sin(np.linspace(270 * math.pi / 180, 180 * math.pi / 180, 81)) * 0.2 + 0.2)
#
#
#reward_need = 180
#
#print(reward_need)
#for gama in [0.995, 0.999]:
#    for epsilon_decay in [ 0.995, 0.99]:
#        for lr in [0.005, 0.01]:
#            for batch in [64, 128]:
#                for tau in [0.01, 0.005]:
#                    scores = deque(maxlen=50)
#                    epi_reward_average=[]
#                    #To plot velocity and distance to trayectory
#                    epi_v_average=[]
#                    epi_d_average=[]
#                    v = deque(maxlen=50)
#                    d = deque(maxlen=50)
#                    agent = Agent(state_size, action_size, gamma=gama, epsilon=0.3, epsilon_min=0.01, epsilon_decay=epsilon_decay,
#                                  learning_rate=lr, batch_size=batch, tau=tau)
#                    plot_ugv = PlotUgv(SPACE_X, SPACE_Y, x_trajectory, y_trajectory, PERIOD)
#                    env = UgvEnv(x_trajectory, y_trajectory, PERIOD,
#                                 5, closed=True)
#                    if True:
#                        agent.load_model('first-training.h5')
#
#                    for e in range(EPISODES):
#                        state, agent_state = env.reset()
#                        agent_state = agent.format_state(agent_state)
#                        done = False
#                        R = 0
#                        epi_v=[]
#                        epi_d=[]
#                        #if e%500==0:
#                        #    plot_ugv.reset(state)
#
#                        while not done:
#                            action = agent.action(agent_state)
#                            new_state, new_agent_state, reward, done = env.step(action)
#                            epi_v.append(env.v_linear)
#                            epi_d.append(np.sqrt(new_agent_state[0]**2))
#                            #if e % 500 == 0:
#                            #    plot_ugv.execute(state)
#                            new_agent_state = agent.format_state(new_agent_state)
#                            agent.remember(agent_state, action, reward, new_agent_state, done)
#
#                            agent_state = new_agent_state
#                            R += reward
#
#                        if len(agent.memory) > agent.batch_size:
#                            agent.replay()
#                            agent.soft_update_target_network()
#                        agent.reduce_random()
#                        scores.append(R)
#                        v.append(np.mean(epi_v))
#                        d.append(np.mean(epi_d))
#                        mean_score = np.mean(scores)
#                        epi_reward_average.append(np.mean(scores))
#                        epi_v_average.append(np.mean(v))
#                        epi_d_average.append(np.mean(d))
#
#                        print("episode: {}/{}, score: {}, e: {:.2}, mean_score: {}, final state :({},{})"
#                              .format(e, EPISODES, R, agent.epsilon, mean_score, env.state[0], env.state[1]))
#                        if mean_score > reward_need:
#                            agent.save_model('second-training.h5')
#                            f = open('parameters-curve.txt', 'a')
#                            f.write(
#                                'Episodes: {}, gamma: {}, epsilon_dec: {}, lr: {}, batch: {}, tau:{}\n'.format(e, gama,
#                                                                                                               epsilon_decay,
#                                                                                                               lr,
#                                                                                                               batch,
#                                                                                                               tau))
#                            f.close()
#                            break