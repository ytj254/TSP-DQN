import os
import sys
import timeit
import time

# import gc
# import objgraph
# import guppy
# from pympler import tracker, muppy, summary
from memory_profiler import profile
import random
import torch
import numpy as np
import traci
import traci.constants as tc
from sumolib import checkBinary

# tr = tracker.SummaryTracker()
# hp = guppy.hpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cell_length = 7
detection_length = 350
lane_num = 16
grid_size = int(detection_length / cell_length)
training_epochs = 200

edges = {
    'east_edge': (0, '-E2'),
    'south_edge': (4, '-E3'),
    'west_edge': (8, 'E0'),
    'north_edge': (12, 'E1')
}
incoming_edges = ['-E2', '-E3', 'E0', 'E1']
action_state_map = {
    0: 'grrrgrrGGgrrrgrrGG',
    1: 'grrrgrrrrgrrrgGGGG',
    2: 'grrrgGGrrgrrrgGGrr',
    3: 'grrrgGGGGgrrrgrrrr',
    4: 'grrGgrrrrgrrGgrrrr',
    5: 'grrrgrrrrgGGGgrrrr',
    6: 'gGGrgrrrrgGGrgrrrr',
    7: 'gGGGgrrrrgrrrgrrrr'
}


def set_sumo(gui=False, sumocfg_path='data/Eastway-Central.sumocfg', random=True, log_path=None):
    # we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # cmd mode or visual mode
    if gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    # setting the cmd to run sumo
    if random and not log_path:
        sumo_cmd = [sumoBinary, '-c', sumocfg_path, '--random', '--no-warnings', '--no-step-log']
    elif random and log_path:
        sumo_cmd = [sumoBinary, '-c', sumocfg_path, '--random', '--no-warnings', '--no-step-log',
                    '--tripinfo-output', log_path + '_tripinfo.xml']
    elif not random and log_path:
        sumo_cmd = [sumoBinary, '-c', sumocfg_path, '--no-warnings', '--no-step-log',
                    '--tripinfo-output', log_path + '_tripinfo.xml']
    else:
        sumo_cmd = [sumoBinary, '-c', sumocfg_path, '--no-warnings', '--no-step-log']

    return sumo_cmd


class Simulation:
    def __init__(self, agent, sumo_cmd, is_training=True):
        self.agent = agent
        self.sumo_cmd = sumo_cmd
        self.is_training = is_training
        self.step = 0

        # self.max_step = 4000
        self.yellow_time = 3
        self.red_time = 2
        self.min_green_time = 10
        self.training_epochs = training_epochs

        self.channels = 2
        self.height = grid_size
        self.width = lane_num

        # store data from every epoch
        self.total_waiting_time = 0
        self.total_waiting_times = []
        self.total_neg_reward = 0
        self.total_rewards = []

    # Run an episode of simulation, then start a training session
    # @profile
    def run(self, epsilon):
        start_time = timeit.default_timer()

        traci.start(self.sumo_cmd)
        print('Simulating...')
        self.step = 0
        self.total_neg_reward = 0
        self.total_waiting_time = 0

        # Warm up 10 minutes
        while self.step < 600:
            traci.simulationStep()
            self.step += 1

        while self.step == 600:
            traci.simulationStep()
            self.step += 1
            last_state = self.get_state()
            last_waiting_time = self.get_waiting_times()
            last_action = torch.tensor([[random.randint(0, 7)]], dtype=torch.long, device=device)

        # Start to call agent
        while 600 < self.step <= 4400:
            # Get the current state
            current_state = self.get_state()

            # Calculate the total waiting time of current state
            current_waiting_time = self.get_waiting_times()
            reward = last_waiting_time - current_waiting_time
            reward_tensor = torch.tensor([reward], device=device)
            # self.total_waiting_time += current_waiting_time
            # print('-----current reward:', reward)

            # Save the data into memory
            if self.is_training:
                self.agent.store(last_state, last_action, reward_tensor, current_state)

            # Signal control
            action = self.agent.get_action(current_state, epsilon)
            action_int = int(action)
            last_action_int = int(last_action)
            # print(self.step, action)
            if last_action_int != action_int:
                self.set_yellow_red(action_int, last_action_int)
                self.set_green(action_int)
            else:
                traci.simulationStep()
                self.step += 1
                self.total_waiting_time += self.get_waiting_times()

            last_state = current_state
            last_action = action
            last_waiting_time = current_waiting_time

            if reward < 0:
                self.total_neg_reward += reward

        print(f'Total reward: {self.total_neg_reward} --Epsilon: {epsilon:.3f} --Total steps: {self.step}')
        self.save_episode_stats()
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        print(f'------ Simulation time: {simulation_time} ------')

        if self.is_training:
            print('Training...')
            start_time = timeit.default_timer()

            for _ in range(self.training_epochs):
                self.agent.train()

            training_time = round(timeit.default_timer() - start_time, 1)
            print(f'------ Training time: {training_time} ------')

    # Execute the designated simulation step
    def simulate(self, steps_todo):
        while steps_todo > 0:
            traci.simulationStep()
            self.step += 1
            steps_todo -= 1
            self.total_waiting_time += self.get_waiting_times()

    # Get the state
    def get_state(self):
        state = torch.zeros((1, self.channels, self.height, self.width), device=device)
        for veh_id in traci.vehicle.getIDList():
            traci.vehicle.subscribe(veh_id, (tc.VAR_NEXT_TLS, tc.VAR_LANE_ID, tc.VAR_SPEED))
        p = traci.vehicle.getAllSubscriptionResults()
        for x in p:
            if p[x][tc.VAR_NEXT_TLS]:
                ps_tls = p[x][tc.VAR_NEXT_TLS][0][2]  # get the distance to the traffic light
            else:
                ps_tls = -1  # vehicle has crossed the stop line and set to a negative value
            if p[x][tc.VAR_LANE_ID]:
                ln_id, ln_idx = p[x][tc.VAR_LANE_ID].split('_')  # get the lane id and index

            spd = p[x][tc.VAR_SPEED]  # get the speed
            # get the position in state array
            if 0 < ps_tls < detection_length:
                height_index = int(ps_tls / cell_length)
                for edge in edges.values():
                    if edge[1] in ln_id:
                        width_index = int(ln_idx) + edge[0]
                        state[:, :, height_index, width_index] = torch.tensor([1, spd])
        return state

    def set_green(self, action):
        """
        phase-movement mapping
        {0: (WL, EL), 1: (W, WL), 2: (W, E), 3: (E, EL), 4: (SL, NL), 5: (S, SL), 6: (S, N), 7: (N, NL)}
        """
        green_state = action_state_map[action]
        traci.trafficlight.setRedYellowGreenState('J1', green_state)
        self.simulate(self.min_green_time)
        # print('------Set green------')

    # Activate the corresponding yellow and red phase
    def set_yellow_red(self, action, last_action):
        action_state = action_state_map[action]
        old_action_state = action_state_map[last_action]
        yellow_state = []
        red_state = []
        for i in range(18):
            # print(action_state[i], old_action_state[i])
            if old_action_state[i] == 'G' and old_action_state[i] != action_state[i]:
                yellow_state.append('Y')
            else:
                yellow_state.append(old_action_state[i])
        yellow_state = ''.join(yellow_state)
        traci.trafficlight.setRedYellowGreenState('J1', yellow_state)
        self.simulate(self.yellow_time)

        for i in range(18):
            if yellow_state[i] == 'Y':
                red_state.append('r')
            else:
                red_state.append(yellow_state[i])
        red_state = ''.join(red_state)
        traci.trafficlight.setRedYellowGreenState('J1', red_state)
        self.simulate(self.red_time)

    # Get the waiting times from the sumo
    def get_waiting_times(self):
        waiting_times = 0
        for edge in incoming_edges:
            waiting_times += traci.edge.getWaitingTime(edge)
        return waiting_times

    def save_episode_stats(self):
        self.total_rewards.append(self.total_neg_reward)
        self.total_waiting_times.append(self.total_waiting_time)

    def get_stats(self):
        return {
            'Reward': self.total_rewards,
            'Mean Waiting Time (s)': np.divide(self.total_waiting_times, self.step)
        }

    def save_stats(self, save_time):
        np.savetxt(f'result\\training_stats_{save_time}.csv', self.total_rewards, delimiter=',')
