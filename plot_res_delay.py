
import yaml
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from matplotlib.patches import Ellipse

ncpu = 8
ngpu = 8
ratio = 6.0/8
batching_time = 0.5

dds = [[], []]
vigil = [[],[]]
glimpse = [[], []]
AWStream = [[], []]

def get_delay(result, key):
    return batching_time * result[key][0] * 1.0 / result['realtime_bandwith']

def obtain_max(res):
    max_acc = res['dds'][1]
    for mpeg_result in res['mpeg_results']:
        max_acc = max(max_acc, mpeg_result[1])
    return max_acc

def add_data(ax, data, marker, handles, label, color):

    handles.append(mlines.Line2D([], [], label=label, c=color))

    cov = np.cov(data[0], data[1])
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)

    ell = Ellipse(xy=(np.mean(data[0]), np.mean(data[1])),
            width=lambda_[0]*2,
            height=lambda_[1]*2,
            angle=np.rad2deg(np.arccos(v[0,0])))
    ell.set_edgecolor(color)
    ell.set_facecolor(color)
    ax.add_artist(ell)
    plt.scatter([np.mean(data[0])], np.mean([data[1]]), c='#000000', marker='o', s=10, zorder=3)


def plot(color, results, save_filename):
    
    # initialize
    plt.clf()
    # plt.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots()
    
    for result in results['results']:
        name = result['name']
        
        # calculate each time
        object_ratio = 0.9
        fst_time = 0.4/ncpu + 0.4/ncpu + 2/(ngpu) + get_delay(result, 'low_quality')
        snd_time = 0.4/ncpu + 0.4/ncpu + 2/(ngpu) + (get_delay(result, 'dds')-get_delay(result, 'low_quality'))
        dds_time = fst_time * 1 + snd_time * (1-object_ratio)
        vigil_time = 1.4/ncpu + get_delay(result, 'vigil')
        glimpse_time = 1.4/ncpu + get_delay(result, 'glimpse')
        AWStream_time = 0.4/ncpu + 0.4/ngpu + 2/ngpu + get_delay(result, 'mpeg')
        
        # append them to corresponding dicts
        dds[0].append(dds_time)
        dds[1].append(result['dds'][1] / obtain_max(result))
        vigil[0].append(vigil_time)
        vigil[1].append(result['vigil'][1] / obtain_max(result))
        glimpse[0].append(glimpse_time)
        glimpse[1].append(result['glimpse'][1] / obtain_max(result))
        AWStream[0].append(AWStream_time)
        AWStream[1].append(result['mpeg'][1] / obtain_max(result))
        
        ax.scatter([dds_time], )
        
    # add data to the graph
    handles=[]
    add_data(ax, dds, 'x', handles, 'dds', color['dds'])
    add_data(ax, vigil, 'v', handles, 'vigil', color['vigil'])
    add_data(ax, glimpse, 's', handles, 'glimpse', color['glimpse'])
    add_data(ax, AWStream, '+', handles, 'AWStream', color['AWStream'])
    
    # set lims and lables
    plt.ylim(0, 1.1)
    plt.xlim(0, 1)
    plt.xlabel('Response time (sec)')
    plt.ylabel('Normalized accuracy')
    
    return fig
    

