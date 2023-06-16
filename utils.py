import os
from matplotlib import pyplot as plt
from shutil import copyfile


def create_model_folder():
    models_folder = 'models'
    models_path = os.path.join(os.getcwd(), models_folder)
    if not os.path.exists(models_path):
        os.mkdir(models_folder)

    dirs = sorted(int(i) for i in os.listdir(models_path))
    if dirs:
        new_dir = dirs[-1] + 1
    else:
        new_dir = 1

    model_path = os.path.join(models_path, str(new_dir))
    os.mkdir(model_path)
    return model_path


def create_result_folder(folder_path):
    result_path = os.path.join(os.getcwd(), folder_path)
    if not os.path.exists(result_path):
        os.makedirs(folder_path)


def plot_data(data, y_label, train_or_test, x_label='Episode'):
    plt.rcParams.update({'font.size': 15})
    plt.plot(data)
    plt.title(x_label + ' vs ' + y_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(min(data), max(data))
    fig_name = f'{train_or_test}_{y_label}.png'
    plt.savefig(fig_name, dpi=100, bbox_inches='tight')
    # plt.show()


def save_data(file_tosave, model_path):
    print('Saving data at: %s\n' % model_path)
    copyfile(file_tosave, os.path.join(model_path, file_tosave))
