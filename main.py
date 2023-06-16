import datetime
import time
# import objgraph

import numpy as np
import sim_tsp as sim
from agent_PyTorch import DQN
from utils import *
from Analysis import analysis

import torch


total_episode = 1000


def main_training():
    sumo_cmd = sim.set_sumo()
    model = DQN()
    simulation = sim.Simulation(model, sumo_cmd)
    model_path = create_model_folder()

    episode = 0
    timestamp_start = datetime.datetime.now()
    local_time = time.strftime('%Y-%m-%d-%H-%M-%S')

    while episode < total_episode:
        print(f'\n------ Episode {str(episode+1)} of {total_episode} ------')
        epsilon = model.get_epsilon(episode)  # set the epsilon based on the episode step
        simulation.run(epsilon)

        # backup
        if (episode + 1) % 100 == 0 or (episode + 1) == total_episode:
            model.save_model(model_path)

        episode += 1

    for key, value in simulation.get_stats().items():
        plot_data(value, key, 'training')
    simulation.save_stats(local_time)
    print('----- Stats saved ------')
    print('----- Model saved at:', model_path)
    print('\n------ Start time:', timestamp_start)
    print('----- End time:', datetime.datetime.now())


def main_testing():
    timestamp_start = datetime.datetime.now()
    local = time.strftime('%Y-%m-%d-%H-%M-%S')
    result_path = os.path.join('result', local)
    create_result_folder(result_path)
    model_path = 'models\\17'
    model = DQN(model_path=model_path)

    totres = []
    episode = 0
    n = 50

    while episode < n:
        print(f'Testing agent\n------ Episode {str(episode+1)} of {n} ------')
        sumo_cmd = sim.set_sumo(log_path=os.path.join(result_path, str(episode+1)), seed=episode)
        simulation = sim.Simulation(model, sumo_cmd, is_training=False)
        simulation.run(epsilon=-1)
        episode += 1

    for filename in os.listdir(result_path):
        file_path = os.path.join(result_path, filename)
        res = analysis(file_path)
        totres.append(res)

    ares = np.reshape(totres, (n, 18))
    np.savetxt(result_path + 'totalresult.csv', ares, delimiter=',')
    # for key, value in simulation.get_stats().items():
    #     plot_data(value, key, 'testing')
    print('\n------ Start time:', timestamp_start)
    print('----- End time:', datetime.datetime.now())


# this is the main entry point of this script
if __name__ == "__main__":

    op = input('Select option: \n(1) Training (2) Testing: ')
    if op == '1':
        main_training()
    elif op == '2':
        main_testing()


