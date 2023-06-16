from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np
from sim_tsp import car_occupancy, bus_occupancy


def analysis(input_path):
    # open xml
    domtree = xml.dom.minidom.parse(input_path)

    # obtain file element
    collection = domtree.documentElement
    tripinfos = collection.getElementsByTagName('tripinfo')

    stats_keys = ['delay', 'b_delay', 'c_delay',
                  'NW', 'NS', 'NE', 'EN', 'EW', 'ES', 'SE', 'SN', 'SW', 'WS', 'WE', 'WN', 'BNS', 'BSN']
    delay_stats = dict((k, []) for k in stats_keys)

    for tripinfo in tripinfos:
        depart_time = float(tripinfo.getAttribute('depart'))
        if 600 <= depart_time <= 4200:
            vtype = tripinfo.getAttribute('vType')

            rou = tripinfo.getAttribute('id').split('.')[0]

            v_delay = tripinfo.getAttribute('timeLoss')

            delay_stats['delay'].append(v_delay)

            if vtype == 'bus':
                # print(v_delay)
                delay_stats['b_delay'].append(v_delay)
            else:
                delay_stats['c_delay'].append(v_delay)

            if rou in delay_stats.keys():
                delay_stats[rou].append(v_delay)

    # mean delay
    for k in delay_stats:
        delay_stats[k] = [float(x) for x in delay_stats[k]]  # string to float

    delay_means = dict((k, np.mean(v)) for k, v in delay_stats.items())
    mean_list = list(v for v in delay_means.values())
    person_delay = (sum(delay_stats['c_delay']) * car_occupancy + sum(delay_stats['b_delay']) * bus_occupancy) / \
                   (len(delay_stats['c_delay']) * car_occupancy + len(delay_stats['b_delay']) * bus_occupancy)
    mean_list.append(person_delay)

    # print('delay per vehicle:', delay_means['delay'])
    # print('delay per bus:', delay_means['b_delay'])
    # print('delay per car:', delay_means['c_delay'])
    #
    # print('delay per car(north right turn):', delay_means['NW'])
    # print('delay per car(north through):', delay_means['NS'])
    # print('delay per car(north left turn):', delay_means['NE'])
    # print('delay per car(east right turn):', delay_means['EN'])
    # print('delay per car(east through):', delay_means['EW'])
    # print('delay per car(east left turn):', delay_means['ES'])
    # print('delay per car(south right turn):', delay_means['SE'])
    # print('delay per car(south through):', delay_means['SN'])
    # print('delay per car(south left turn):', delay_means['SW'])
    # print('delay per car(west right turn):', delay_means['WS'])
    # print('delay per car(west through):', delay_means['WE'])
    # print('delay per car(west left turn):', delay_means['WN'])
    # print('delay per bus(north through):', delay_means['BNS'])
    # print('delay per bus(south through):', delay_means['BSN'])

    return mean_list


if __name__ == '__main__':
    analysis('D:\\Paper Publish\\CAV\\TSP$DQN\\Python code\\result\\2023-01-02-17-18-26\\1_tripinfo.xml')