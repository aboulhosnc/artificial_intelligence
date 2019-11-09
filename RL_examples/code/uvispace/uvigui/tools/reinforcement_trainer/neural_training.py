"""This module trains a tests the controller in a different thread of the GUI
"""
import sys
from uvispace.uvinavigator.controllers.linefollowers.neural_controller.DQNagent import Agent
import numpy as np
from uvispace.uvirobot.robot_model.environment import UgvEnv
from collections import deque
import threading
import copy
from PyQt5 import QtCore
import configparser


class NeuralTraining(QtCore.QThread):
    def __init__(self):
        QtCore.QThread.__init__(self)

        self.SPACE_X = 4
        self.SPACE_Y = 3
        self.PERIOD = 1/12
        self.NUM_DIV_ACTION = 5
        self.INIT_TO_ZERO = True
        self.EPISODES = 700
        self.state_size = 2
        self.action_size = 5 * 5

        self.lock = threading.Lock()

    def trainclosedcircuitplot(self, load=False, load_name='emodel.h5', save_name='emodel.h5', differential_car=True):
        """This function defines the training variables and start the thread to train
        """

        self.load = load
        self.load_name = load_name
        self.save_name = save_name
        self.differential_car = differential_car

        self.start()

    def run(self):
        """This function runs the training algorithm
        """
        configuration = configparser.ConfigParser()
        conf_file = "parameters.cfg"
        configuration.read(conf_file)

        if self.differential_car:
            # Read csv file
            coordinates = np.loadtxt(
                open("uvispace/uvigui/tools/reinforcement_trainer/resources/training_differential.csv", "r"),
                delimiter=";")
            x_trajectory = []
            y_trajectory = []
            for point in coordinates:
                x_trajectory.append(point[0])
                y_trajectory.append(point[1])

        else:

            # Read csv file
            coordinates = np.loadtxt(
                open("uvispace/uvigui/tools/reinforcement_trainer/resources/training_ackerman.csv", "r"),
                delimiter=";")
            x_trajectory = []
            y_trajectory = []
            for point in coordinates:
                x_trajectory.append(point[0])
                y_trajectory.append(point[1])

        scores = deque(maxlen=50)
        self.epi_reward_average = []
        # To plot velocity and distance to trayectory
        self.epi_v_average = []
        self.epi_d_average = []
        v = deque(maxlen=50)
        d = deque(maxlen=50)


        #read values from the file
        gamma=float(configuration["par"]["gamma"])
        eps=float(configuration["par"]["ini_randomness"])
        eps_min=float(configuration["par"]["min_randomness"])
        eps_dec=float(configuration["par"]["reduce_random"])
        alpha=float(configuration["par"]["alpha"])
        batch=int(configuration["par"]["batch_size"])


        agent = Agent(self.state_size, self.action_size, gamma=gamma, epsilon=eps, epsilon_min=eps_min, epsilon_decay=eps_dec,
                      learning_rate=alpha, batch_size=batch, tau=0.01)
        self.reward_need = (len(x_trajectory) // 50) * 5 + 15
        print(self.reward_need)
        if self.differential_car:
            env = UgvEnv(x_trajectory, y_trajectory, self.PERIOD,
                         self.NUM_DIV_ACTION, closed=False, differential_car=True)

        else:
            env = UgvEnv(x_trajectory, y_trajectory, self.PERIOD,
                         self.NUM_DIV_ACTION, closed=False, differential_car=False)
        if self.load:
            agent.load_model(self.load_name)

        for e in range(self.EPISODES):
            state, agent_state = env.reset()
            agent_state = agent.format_state(agent_state)
            done = False
            R = 0
            epi_v = []
            epi_d = []

            while not done:
                action = agent.action(agent_state)
                new_state, new_agent_state, reward, done = env.step(action)
                epi_v.append(env.v_linear)
                epi_d.append(np.sqrt(new_agent_state[0] ** 2))
                new_agent_state = agent.format_state(new_agent_state)
                agent.remember(agent_state, action, reward, new_agent_state, done)

                agent_state = new_agent_state
                R += reward

            if len(agent.memory) > agent.batch_size:
                agent.replay()
                agent.soft_update_target_network()
            agent.reduce_random()
            scores.append(R)
            v.append(np.mean(epi_v))
            d.append(np.mean(epi_d))
            mean_score = np.mean(scores)

            # thread-safe copy of averages into shared variables with main thread
            self.lock.acquire()
            self.epi_reward_average.append(np.mean(scores))
            self.epi_v_average.append(np.mean(v))
            self.epi_d_average.append(np.mean(d))
            self.lock.release()

            if e % 100 == 0:
                print("episode: {}, score: {}, e: {:.2}, mean_score: {}, final state :({},{})"
                      .format(e, R, agent.epsilon, mean_score, env.state[0], env.state[1]))

            if mean_score > self.reward_need:
                print("episode: {}, score: {}, e: {:.2}, mean_score: {}, final state :({},{})"
                      .format(e, R, agent.epsilon, mean_score, env.state[0], env.state[1]))
                agent.save_model(self.save_name)
                break
        agent.save_model(self.save_name)

    def read_averages(self):
        """This function locks the variables to be read by the GUI
         """
        self.lock.acquire()
        return_reward = copy.deepcopy(self.epi_reward_average)
        return_v = copy.deepcopy(self.epi_v_average)
        return_d = copy.deepcopy(self.epi_d_average)
        self.lock.release()
        return return_reward, return_v, return_d


class NeuralTesting(QtCore.QThread):
    def __init__(self):
        QtCore.QThread.__init__(self)

        self.SPACE_X = 4
        self.SPACE_Y = 3
        self.PERIOD = 1/12
        self.NUM_DIV_ACTION = 5
        self.INIT_TO_ZERO = True
        self.EPISODES = 5000
        self.state_size = 2
        self.action_size = 5 * 5
        self.v = []
        self.d = []

        self.lock = threading.Lock()

    def testing(self, load_name, x_trajectory, y_trajectory, closed=True, differential_car=True):
        """This function defines the testing variables
        """

        self.load_name = load_name
        self.x_trajectory = x_trajectory
        self.y_trajectory = y_trajectory
        self.closed = closed
        self.states = []
        self.differential_car = differential_car

        self.start()

    def run(self):
        """This function runs the testing algorithm
        """
        if not self.closed:
            reward_need = (len(self.x_trajectory) // 50) * 5 + 10
            print("Reward if it finishes: {}".format(reward_need))
        scores = deque(maxlen=3)
        agent = Agent(self.state_size, self.action_size, gamma=0.999, epsilon=1, epsilon_min=0.01, epsilon_decay=0.995,
                      learning_rate=0.01, batch_size=128, tau=0.01)

        if self.differential_car:
            env = UgvEnv(self.x_trajectory, self.y_trajectory, self.PERIOD,
                         self.NUM_DIV_ACTION, closed=self.closed, differential_car=True)

        else:
            env = UgvEnv(self.x_trajectory, self.y_trajectory, self.PERIOD,
                         self.NUM_DIV_ACTION, closed=self.closed, differential_car=False)

        agent.load_model(self.load_name)

        state, agent_state = env.reset()
        agent_state = agent.format_state(agent_state)
        done = False
        R = 0
        self.v = []
        self.d = []
        self.states.append(state)
        while not done:
            action = agent.action(agent_state, training=False)
            new_state, new_agent_state, reward, done = env.step(action)

            self.states.append(new_state)

            self.lock.acquire()
            self.v.append(env.v_linear)
            self.d.append(np.sqrt(env.distance ** 2))
            self.lock.release()

            new_agent_state = agent.format_state(new_agent_state)
            agent_state = new_agent_state
            R += reward
        scores.append(R)
        mean_score = np.mean(scores)
        mean_v = np.mean(self.v)
        mean_d = np.mean(self.d)
        print(
            "score: {}, laps: {:}, mean_score: {}, final state :({},{}), velocidad media: {}, Distancia media: {}"
            .format(R, env.laps, mean_score, env.state[0], env.state[1], mean_v, mean_d))

    def read_values(self):
        """This function locks the variables to be read by the GUI
        """
        self.lock.acquire()
        return_v = copy.deepcopy(self.v)
        return_d = copy.deepcopy(self.d)
        self.lock.release()
        return return_v, return_d
