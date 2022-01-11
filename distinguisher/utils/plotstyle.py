import brewer2mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set up some better defaults for matplotlib
from matplotlib import rcParams
import matplotlib
import matplotlib.ticker as mtick  # for formatting as percentage

import seaborn as sbn

# colorbrewer2 Dark2 qualitative color table
dark2_colors = brewer2mpl.get_map('Dark2', 'Qualitative', 7).mpl_colors

rcParams['figure.figsize'] = (3.3, 2)  # (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.prop_cycle'] = matplotlib.cycler(color=dark2_colors)
rcParams['lines.linewidth'] = 1
# rcParams['axes.facecolor'] = 'white'
# Give plot a gray background like ggplot.
rcParams['axes.facecolor'] = '#EBEBEB'
rcParams['axes.titlesize'] = 9
rcParams['font.size'] = 10
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['legend.loc'] = 'best'
rcParams['legend.edgecolor'] = 'white'
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'StixGeneral'
rcParams['mathtext.fontset'] = 'cm'
# rcParams['font.family']='serif'

rcParams['grid.linewidth'] = 0.75
rcParams['grid.linestyle'] = '-'
rcParams['grid.color'] = 'white'
rcParams['axes.grid'] = True
rcParams['text.usetex'] = False


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks

    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)

    # turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    # now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()