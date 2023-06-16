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
training_epochs = 100
car_occupancy = 1
bus_occupancy = 30
min_left_green_time = 5
min_through_green_time = 12

edges = {
    'east_edge': (0, '-E2'),
    'south_edge': (4, '-E3'),
    'west_edge': (8, 'E0'),
    'north_edge': (12, 'E1')
}
incoming_edges = ['-E2', '-E3', 'E0', 'E1']
action_state_map = {
    0: 'grrrgrrGGgrrrgrrGG',  # WL EL
    1: 'grrrgrrrrgrrrgGGGG',  # WL WT
    2: 'grrrgGGrrgrrrgGGrr',  # WT ET
    3: 'grrrgGGGGgrrrgrrrr',  # EL ET
    4: 'grrGgrrrrgrrGgrrrr',  # SL NL
    5: 'grrrgrrrrgGGGgrrrr',  # SL ST
    6: 'gGGrgrrrrgGGrgrrrr',  # ST NT
    7: 'gGGGgrrrrgrrrgrrrr'   # NL NT
}


def set_sumo(gui=False, sumocfg_path='data/Eastway-Central.sumocfg', random=True, log_path=None, seed=-1):
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
        if seed < 0:
            sumo_cmd = [sumoBinary, '-c', sumocfg_path, '--random', '--no-warnings', '--no-step-log']
        else:
            sumo_cmd = [sumoBinary, '-c', sumocfg_path, "--seed", "%d" % seed, '--no-warnings', '--no-step-log']
    elif random and log_path:
        if seed < 0:
            sumo_cmd = [sumoBinary, '-c', sumocfg_path, '--random', '--no-warnings', '--no-step-log',
                        '--tripinfo-output', log_path + '_tripinfo.xml']
        else:
            sumo_cmd = [sumoBinary, '-c', sumocfg_path, '--seed', '%d' % seed, '--no-warnings', '--no-step-log',
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
        self.min_left_green_time = min_left_green_time
        self.min_through_green_time = min_through_green_time
        self.training_epochs = training_epochs

        self.channels = 2
        self.height = grid_size
        self.width = lane_num

        # store data from every epoch
        self.total_person_delay = 0
        # self.total_person_delays = []
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
        self.total_person_delay = 0

        # Warm up 10 minutes
        while self.step < 600:
            traci.simulationStep()
            self.step += 1

        while self.step == 600:
            traci.simulationStep()
            self.step += 1

            last_state, last_tot_person_delay = self.get_state()
            # last_state, last_tot_person_delay = self.get_state_cv()

            last_action = torch.tensor([[random.randint(0, 7)]], dtype=torch.long, device=device)

        # Start to call agent
        while 600 < self.step <= 4400:
            # Get the current state
            current_state, current_tot_person_delay = self.get_state()
            # current_state, current_tot_person_delay = self.get_state_cv()

            # Calculate the total waiting time of current state
            reward = last_tot_person_delay - current_tot_person_delay
            reward_tensor = torch.tensor([reward], device=device)
            # print('-----current reward:', reward)

            # Save the data into memory
            if self.is_training:
                self.agent.store(last_state, last_action, reward_tensor, current_state)

            # Signal control
            current_action = self.agent.get_action(current_state, epsilon)
            current_action_int = int(current_action)
            last_action_int = int(last_action)
            # print(self.step, action)
            if current_action_int != last_action_int:
                self.set_yellow_red(current_action_int, last_action_int)
                if current_action_int == 2 or current_action_int == 6:
                    self.set_green(current_action_int, self.min_through_green_time)
                else:
                    self.set_green(current_action_int, self.min_left_green_time)
            else:
                self.simulate(1)

            last_state = current_state
            last_action = current_action
            last_tot_person_delay = current_tot_person_delay

            if reward < 0:
                self.total_neg_reward += reward

        print(f'Total reward: {self.total_neg_reward:.2f} --Epsilon: {epsilon:.3f} --Total steps: {self.step}')
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

    # Get the state
    def get_state(self):
        state = torch.zeros((1, self.channels, self.height, self.width), device=device)
        tot_person_delay = 0

        for veh_id in traci.vehicle.getIDList():
            traci.vehicle.subscribe(veh_id, (tc.VAR_NEXT_TLS, tc.VAR_LANE_ID, tc.VAR_SPEED, tc.VAR_TYPE,
                                             tc.VAR_TIMELOSS))
        p = traci.vehicle.getAllSubscriptionResults()
        for x in p:
            if p[x][tc.VAR_NEXT_TLS]:
                ps_tls = p[x][tc.VAR_NEXT_TLS][0][2]  # get the distance to the traffic light
            else:
                ps_tls = -1  # vehicle crossing the stop line is set to a negative value

            if p[x][tc.VAR_LANE_ID]:
                ln_id, ln_idx = p[x][tc.VAR_LANE_ID].split('_')  # get the lane id and index

            spd = p[x][tc.VAR_SPEED]  # get the speed

            if ps_tls > 0:  # vehicle not crossing the stop line
                delay = p[x][tc.VAR_TIMELOSS]
            else:  # vehicle already crossing the stop line
                delay = 0

            # get the vehicle type and assign the occupancy
            if p[x][tc.VAR_TYPE] == 'car':
                v_occupancy = car_occupancy
                person_delay = delay * car_occupancy
            else:
                v_occupancy = bus_occupancy
                person_delay = delay * bus_occupancy
            tot_person_delay += person_delay

            # get the position in state array
            if 0 < ps_tls < detection_length:
                height_index = int(ps_tls / cell_length)
                for edge in edges.values():
                    if edge[1] in ln_id:
                        width_index = int(ln_idx) + edge[0]
                        state[:, :, height_index, width_index] = torch.tensor([v_occupancy, spd])
        return state, tot_person_delay

    # Get the states of CVs
    def get_state_cv(self):
        state = torch.zeros((1, self.channels, self.height, self.width), device=device)
        tot_person_delay = 0

        for veh_id in traci.vehicle.getIDList():
            traci.vehicle.subscribe(veh_id, (tc.VAR_NEXT_TLS, tc.VAR_LANE_ID, tc.VAR_SPEED, tc.VAR_TYPE,
                                             tc.VAR_TIMELOSS))
        p = traci.vehicle.getAllSubscriptionResults()
        for x in p:
            v_type = p[x][tc.VAR_TYPE]
            if v_type == 'cv' or v_type == 'bus':  # Only cvs and buses are sensed
                if p[x][tc.VAR_NEXT_TLS]:
                    ps_tls = p[x][tc.VAR_NEXT_TLS][0][2]  # get the distance to the traffic light
                else:
                    ps_tls = -1  # vehicle crossing the stop line is set to a negative value

                if p[x][tc.VAR_LANE_ID]:
                    ln_id, ln_idx = p[x][tc.VAR_LANE_ID].split('_')  # get the lane id and index

                spd = p[x][tc.VAR_SPEED]  # get the speed

                if ps_tls > 0:  # vehicle not crossing the stop line
                    delay = p[x][tc.VAR_TIMELOSS]
                else:  # vehicle already crossing the stop line
                    delay = 0

                # get the vehicle type and assign the occupancy
                if v_type == 'cv':
                    v_occupancy = car_occupancy
                    person_delay = delay * car_occupancy
                else:
                    v_occupancy = bus_occupancy
                    person_delay = delay * bus_occupancy
                tot_person_delay += person_delay

                # get the position in state array
                if 0 < ps_tls < detection_length:
                    height_index = int(ps_tls / cell_length)
                    for edge in edges.values():
                        if edge[1] in ln_id:
                            width_index = int(ln_idx) + edge[0]
                            state[:, :, height_index, width_index] = torch.tensor([v_occupancy, spd])
        return state, tot_person_delay

    def set_green(self, action, min_green_time):
        """
        phase-movement mapping
        {0: (WL, EL), 1: (W, WL), 2: (W, E), 3: (E, EL), 4: (SL, NL), 5: (S, SL), 6: (S, N), 7: (N, NL)}
        """
        green_state = action_state_map[action]
        traci.trafficlight.setRedYellowGreenState('J1', green_state)
        self.simulate(min_green_time)
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

    def save_episode_stats(self):
        self.total_rewards.append(self.total_neg_reward)

    def get_stats(self):
        return {
            'Reward': self.total_rewards,
            # 'Mean Waiting Time (s)': np.divide(self.total_person_delays, self.step)
        }

    def save_stats(self, save_time):
        np.savetxt(f'result\\training_stats_{save_time}.csv', self.total_rewards, delimiter=',')
