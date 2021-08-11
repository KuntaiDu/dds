import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Ellipse
import math
from pathlib import Path

# just keep it if you just want the center of each ellipse to be black.
dark_ratio = 10

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    
    If amout > 1, this function will darken the color.
    """
    if amount == 10:
        return '#000000'
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def obtain(percentage, result):
    max_accuracy = 0
    result['mpeg_results'].append(result['dds'])
    for mpeg_result in result['mpeg_results']:
        max_accuracy = max(max_accuracy, mpeg_result[1])
    target_accuracy = max_accuracy * percentage
    # print(target_accuracy)
    max_result = [0, 0]
    # print(result['mpeg_results'])
    for mpeg_result in result['mpeg_results']:
        if mpeg_result[1] <= target_accuracy and mpeg_result[1] > max_result[1]:
            # print(f'{mpeg_result[1]} < {target_accuracy}')
            max_result = mpeg_result
    if max_result == [0, 0]:
        return None
    return [max_result[0], max_result[1]/max(max_accuracy, result['dds'][1])]

def get_nframes(result):
    img_path = Path(f"/data2/yuanx/new_dataset/{result['name']}/src")
    nframes = len([i for i in img_path.iterdir() if '.png' in str(i)])
    return nframes

def get_maxbw(result):
    max_bw = 0
    for mpeg_result in result['mpeg_results']:
        max_bw = max(max_bw, mpeg_result[0])
    return max_bw

def obtain_by_acc(acc, result):
    nframes = get_nframes(result)
    max_result = [0,0]
    for mpeg_result in result['mpeg_results']:
        if mpeg_result[1] < acc and mpeg_result[1] > max_result[1]:
            max_result = mpeg_result
            
    return [max_result[0] / get_maxbw(result), max_result[1]]

def reshape(x):
    return [[i[0] for i in x if i != None], [i[1] for i in x if i != None]]


def check(x):
    assert(len(x[0]) == len(x[1]))
    
def add_data(ax, data, marker, handles, label, color):

    if label != None:
        handles.append(mlines.Line2D([], [], label=label, c=color))

    cov = np.cov(data[0], data[1])
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    print(np.mean(data[0]), np.mean(data[1]))

    ell = Ellipse(xy=(np.mean(data[0]), np.mean(data[1])),
            width=lambda_[0]*2,
            height=lambda_[1]*2,
            angle=np.rad2deg(np.arccos(v[0,0])))
    ell.set_edgecolor(color)
    ell.set_facecolor(color)
    ax.add_artist(ell)
    #plt.scatter(data[0], data[1], c = color)
    plt.scatter([np.mean(data[0])], [np.mean(data[1])], marker = 'o', color = lighten_color(color, dark_ratio), s=10,zorder=3)

def link(a, b):
    plt.plot([np.mean(a[0]), np.mean(b[0])], [np.mean(a[1]), np.mean(b[1])], c='#000000')


def plot(color, results, font_size, save_filename):
    
    
    # initialize
    plt.clf()
    plt.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots(figsize = (7, 5))
    
    # Obtain all AWStream results according to bandwith limitation.
    AWStream = reshape([obtain_by_acc(res['dds'][1], res) for res in results['results']])
    
    # if you want to add legend, use this.
    handles = []
    
    # parse all results. X-axis: normalized bandwith, Y-axis: normalized accuracy (normalized by maximum accuracy of both mpegs and dds)
    dds = [[res['dds'][0] / get_maxbw(res) for res in results['results']],[res['dds'][1] for res in results['results']]]
    vigil = [[res['vigil'][0] / get_maxbw(res) for res in results['results'] ],[res['vigil'][1] for res in results['results'] ]]
    glimpse = [[res['glimpse'][0] / get_maxbw(res) for res in results['results']],[res['glimpse'][1] for res in results['results'] ]]
    
    # add these results to plot
    add_data(ax, AWStream, 'D', handles, 'AWStream', color['dds'])
    add_data(ax, glimpse, '', handles, 'glimpse', color['glimpse'])
    add_data(ax, vigil, '', handles, 'vigil', '#6e5773')
    add_data(ax, dds, 'x', handles, 'dds', color['AWStream'])
    
    # set the range of X-axis and Y-axis.
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0.35, 0.99)
    ax.set_xlabel('Norm. bandwidth consumption')
    ax.set_ylabel('Accuracy ')
    ax.grid()
    
    
    fig.tight_layout()
    
    # save the fig. The bbox_inches='tight' is crucial.
    fig.savefig(save_filename, bbox_inches='tight')
    
    return plt
