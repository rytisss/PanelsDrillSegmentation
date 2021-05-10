import matplotlib.pyplot as plt
import numpy as np


def render_performance_diagram():
    data_desktopRTX2070Super = {
        'UNet': 0.027872265338897705,
        'UNet +\nCoordConv2D': 0.03714177966117859,
        'UNet +\nSE': 0.028610870361328124,
        'UNet +\nCoordConv2D +\nSE': 0.0382606725692749,
        'UNet +\nRes +\nASPP': 0.02999012589454651,
        'UNet +\nRes +\nASPP +\nCoordConv2D': 0.03960700368881226,
        'UNet +\nRes +\nASPP +\nSE': 0.03138042092323303,
        'UNet +\nRes +\nASPP +\nCoordConv2D +\nSE': 0.04077408647537231
    }

    data_laptopGTX1050Ti = {
        'UNet': 0.06192016196250916,
        'UNet +\nCoordConv2D': 0.08674508333206177,
        'UNet +\nSE': 0.06514274668693543,
        'UNet +\nCoordConv2D +\nSE': 0.09009586071968079,
        'UNet +\nRes +\nASPP': 0.06708421111106873,
        'UNet +\nRes +\nASPP +\nCoordConv2D': 0.09568582463264465,
        'UNet +\nRes +\nASPP +\nSE': 0.06979887533187866,
        'UNet +\nRes +\nASPP +\nCoordConv2D +\nSE': 0.10224510979652404
    }

    nn_names = []
    GPU2070S = []
    for key, value in data_desktopRTX2070Super.items():
        GPU2070S.append(value* 1000)
        nn_names.append(key)

    GPU1050Ti = []
    for key, value in data_laptopGTX1050Ti.items():
        GPU1050Ti.append(value * 1000)

    bar_width = 0.4

    indeces = np.arange(len(nn_names))

    # plt.figure(1, figsize=(12,16))
    fig, ax = plt.subplots(figsize=(11, 7))

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    # Plotting
    rects_1 = plt.bar(indeces - bar_width/2, GPU1050Ti, bar_width, label='Mobile GTX1050Ti', color='#59a4f0', edgecolor='#091229')
    rects_2 = plt.bar(indeces + bar_width/2, GPU2070S, bar_width, label='Desktop RTX2070S', color='#ff7700', edgecolor='#091229')

    # Turn on the minor TICKS, which are required for the minor GRID
    ax.minorticks_on()
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.4)
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='red', alpha=0.2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    #ax.set_xticks(xlab)
    #plt.xticks(nn_names + bar_width / 2, nn_names)
    ax.axes.set_xticks(indeces)
    ax.axes.set_xticklabels(nn_names)

    #ax.set_xticklabels(nn_names)
    ax.legend(loc='upper left',fontsize=16)
    plt.ylabel('Miliseconds [ms]', fontsize=16)
    plt.xlabel('CNN architecture', fontsize=16)
    plt.title('Calculation Performance Evaluation', fontsize=18)

    for i in range(len(rects_1)):
        height_1 = rects_1[i].get_height()
        ax.text(rects_1[i].get_x() + rects_1[i].get_width() / 2, height_1 - 14.5, str(format(GPU1050Ti[i], '.2f')),
            ha='center', rotation='vertical', fontsize=16)
        height_2 = rects_2[i].get_height()
        ax.text(rects_2[i].get_x() + rects_2[i].get_width() / 2, height_2 - 14.5, str(format(GPU2070S[i], '.2f')),
                ha='center', rotation='vertical', fontsize=16)
    plt.tight_layout()
    # function to show the plot
    plt.show()
    fig.savefig('performance.png', dpi=400)
    plt.close()


def render_parameters_diagram():
    data_parameters = {
        'UNet': 1664714,
        'UNet +\nCoordConv2D': 1688618,
        'UNet +\nSE': 1848010,
        'UNet +\nCoordConv2D +\nSE': 1871914,
        'UNet +\nRes +\nASPP': 3750090,
        'UNet +\nRes +\nASPP +\nCoordConv2D': 3773994,
        'UNet +\nRes +\nASPP +\nSE': 3933386,
        'UNet +\nRes +\nASPP +\nCoordConv2D +\nSE': 3957290
    }

    nn_names = []
    parameters = []
    for key, value in data_parameters.items():
        parameters.append(value)
        nn_names.append(key)

    indeces = np.arange(len(nn_names))

    # plt.figure(1, figsize=(12,16))
    fig, ax = plt.subplots(figsize=(11, 7))

    # Plotting
    rects_1 = plt.bar(indeces, parameters, 0.7, label='', color='#ff7700',
                      edgecolor='#091229')

    # Turn on the minor TICKS, which are required for the minor GRID
    ax.minorticks_on()
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.4)
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='red', alpha=0.2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_xticks(xlab)
    # plt.xticks(nn_names + bar_width / 2, nn_names)
    ax.axes.set_xticks(indeces)
    ax.axes.set_xticklabels(nn_names)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    # ax.set_xticklabels(nn_names)
    #ax.legend(loc='upper left')
    plt.ylabel('Number of Parameters', fontsize=16)
    plt.xlabel('CNN architecture', fontsize=16)
    plt.title('Number of Parameters in CNN', fontsize=18)

    for i in range(len(rects_1)):
        height_1 = rects_1[i].get_height()
        ax.text(rects_1[i].get_x() + rects_1[i].get_width() / 2, height_1 - 1050000, str(parameters[i]),
                ha='center', rotation='vertical', fontsize=18)

    plt.tight_layout()
    # function to show the plot
    plt.show()
    fig.savefig('parameters.png', dpi=400)
    plt.close()

def main():
    render_performance_diagram()
    render_parameters_diagram()


if __name__ == '__main__':
    main()
