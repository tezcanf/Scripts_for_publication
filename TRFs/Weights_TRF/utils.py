#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 16:25:01 2021

@author: filtsem
"""

from scipy.stats import ttest_rel, sem
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Configure the matplotlib figure style
FONT = "Times New Roman"
RC = {
    'figure.dpi': 100,
    'savefig.dpi': 100,
    'savefig.transparent': True,
    # Font
    'font.family': FONT,
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'figure.figsize': (4,3)}


plt.rcParams.update(RC) 


    
def line_graph1(x1,x2, x3, x4,legend, title, entropy_effect, language_effect, interaction):


    plot_lines1 = []
    plot_lines2 = []
    fig, ax = plt.subplots()

    
    x1_mean = x1.mean('case').x[2:-2]
    time = x1.time.times[2:-2]
    x1_error = [sem(x1.x[:,t]) for t in range(len(time)) ]
    
    l1, = ax.plot(time, x1_mean,  'yellowgreen', linewidth=0.5)
    ax.fill_between(time, x1_mean-x1_error, x1_mean+x1_error,  alpha=0.7, color='yellowgreen',edgecolor='yellowgreen', linewidth=0.0)

    
    x2_mean = x2.mean('case').x[2:-2]
    time = x2.time.times[2:-2]
    x2_error = [sem(x2.x[:,t]) for t in range(len(time)) ]
    
    l2, = ax.plot(time, x2_mean, 'olivedrab', linewidth=0.5)
    ax.fill_between(time, x2_mean-x2_error, x2_mean+x2_error,  alpha=0.7, color = 'olivedrab',edgecolor='yellowgreen', linewidth=0.0)
    
    x3_mean = x3.mean('case').x[2:-2]
    time = x3.time.times[2:-2]
    x3_error = [sem(x3.x[:,t]) for t in range(len(time)) ]
    
    l3, = ax.plot(time, x3_mean, 'coral', linewidth=0.5)
    ax.fill_between(time, x3_mean-x3_error, x3_mean+x3_error,  alpha=0.7, color = 'coral', edgecolor='yellowgreen', linewidth=0.0)
    
    x4_mean = x4.mean('case').x[2:-2]
    time = x4.time.times[2:-2]
    x4_error = [sem(x4.x[:,t]) for t in range(len(time)) ]
    
    l4, = ax.plot(time, x4_mean,'orangered', linewidth=0.5)
    ax.fill_between(time, x4_mean-x4_error, x4_mean+x4_error,  alpha=0.7, color = 'orangered',edgecolor='yellowgreen', linewidth=0.0)
      

    
    l5, = ax.plot(time, np.multiply(entropy_effect,  0.000012),'black', linewidth=2)
    l6, = ax.plot(time, np.multiply(language_effect, 0.000007),'grey', linewidth=2)
    l7,  = ax.plot(time, np.multiply(interaction, 0.000002),'red' ,linewidth=2)


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.axes.xaxis.set_ticklabels([])
    # ax.axes.yaxis.set_ticklabels([])

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])


    plot_lines2.append([l5, l6, l7])
    #legend2 = plt.legend(plot_lines2[0], ['Word Entropy Main effect','Language Main effect', 'Language - Entropy Interaction'],loc='upper center', bbox_to_anchor=(0.5, -0.05))
       
    # plt.gca().add_artist(legend1)
    # plt.gca().add_artist(legend2)
    plt.title(title)
    plt.xlabel("Time (sec)")
    plt.ylabel('Power of Weights $\mathregular{\sqrt{w^{2}}}$')
    plt.ylim(0.00000,0.00015)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    # plt.ylim(0.00000,0.00025)
    plt.tight_layout()
    ax.yaxis.set_major_locator(MaxNLocator(4))
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    #fig.set_size_inches(4, 2)
    
    plt.show()

    
def line_graph2(x1,x2, x3, x4,legend, title, entropy_effect, language_effect, interaction):


    plot_lines1 = []
    plot_lines2 = []
    fig, ax = plt.subplots()

    
    x1_mean = x1.mean('case').x[2:-2]
    time = x1.time.times[2:-2]
    x1_error = [sem(x1.x[:,t]) for t in range(len(time)) ]
    
    l1, = ax.plot(time, x1_mean,  'yellowgreen', linewidth=0.5)
    ax.fill_between(time, x1_mean-x1_error, x1_mean+x1_error,  alpha=0.7, color='yellowgreen',edgecolor='yellowgreen', linewidth=0.0)

    
    x2_mean = x2.mean('case').x[2:-2]
    time = x2.time.times[2:-2]
    x2_error = [sem(x2.x[:,t]) for t in range(len(time)) ]
    
    l2, = ax.plot(time, x2_mean, 'olivedrab', linewidth=0.5)
    ax.fill_between(time, x2_mean-x2_error, x2_mean+x2_error,  alpha=0.7, color = 'olivedrab',edgecolor='yellowgreen', linewidth=0.0)
    
    x3_mean = x3.mean('case').x[2:-2]
    time = x3.time.times[2:-2]
    x3_error = [sem(x3.x[:,t]) for t in range(len(time)) ]
    
    l3, = ax.plot(time, x3_mean, 'coral', linewidth=0.5)
    ax.fill_between(time, x3_mean-x3_error, x3_mean+x3_error,  alpha=0.7, color = 'coral', edgecolor='yellowgreen', linewidth=0.0)
    
    x4_mean = x4.mean('case').x[2:-2]
    time = x4.time.times[2:-2]
    x4_error = [sem(x4.x[:,t]) for t in range(len(time)) ]
    
    l4, = ax.plot(time, x4_mean,'orangered', linewidth=0.5)
    ax.fill_between(time, x4_mean-x4_error, x4_mean+x4_error,  alpha=0.7, color = 'orangered',edgecolor='yellowgreen', linewidth=0.0)
      

    
    l5, = ax.plot(time, np.multiply(entropy_effect,  0.000012),'black', linewidth=2)
    l6, = ax.plot(time, np.multiply(language_effect, 0.000007),'grey', linewidth=2)
    l7,  = ax.plot(time, np.multiply(interaction, 0.000002),'red' ,linewidth=2)


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    plot_lines2.append([l5, l6, l7])
    #legend2 = plt.legend(plot_lines2[0], ['Word Entropy Main effect','Language Main effect', 'Language - Entropy Interaction'],loc='upper center', bbox_to_anchor=(0.5, -0.05))
       
    # plt.gca().add_artist(legend1)
    # plt.gca().add_artist(legend2)
    # plt.title(title)
    #plt.xlabel("Time (sec)")
    #plt.ylabel('Power of Weights $\mathregular{\sqrt{w^{2}}}$')
    # plt.ylim(0.00000,0.00015)
    plt.ylim(0.00000,0.00025)
    plt.tight_layout()
    ax.yaxis.set_major_locator(MaxNLocator(4))
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    #fig.set_size_inches(4, 2)
    
    plt.show()
    
def line_graph_time_interval(x1,x2, x3, x4,legend, title, entropy_effect, language_effect, interaction):


    plot_lines1 = []
    plot_lines2 = []
    fig, ax = plt.subplots()

    
    x1_mean = x1.mean(0)
    time = [ -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6 ,0.65, 0.7, 0.75]
    x1_error = [sem(x1[:,t]) for t in range(len(time)) ]
    
    l1, = ax.plot(time, x1_mean,  'yellowgreen', linewidth=0.5)
    ax.fill_between(time, x1_mean-x1_error, x1_mean+x1_error,  alpha=0.7, color='yellowgreen',edgecolor='yellowgreen', linewidth=0.0)

    
    x2_mean = x2.mean(0)
    x2_error = [sem(x2[:,t]) for t in range(len(time)) ]
    
    l2, = ax.plot(time, x2_mean, 'olivedrab', linewidth=0.5)
    ax.fill_between(time, x2_mean-x2_error, x2_mean+x2_error,  alpha=0.7, color = 'olivedrab',edgecolor='yellowgreen', linewidth=0.0)
    
    x3_mean = x3.mean(0)
    x3_error = [sem(x3[:,t]) for t in range(len(time)) ]
    
    l3, = ax.plot(time, x3_mean, 'coral', linewidth=0.5)
    ax.fill_between(time, x3_mean-x3_error, x3_mean+x3_error,  alpha=0.7, color = 'coral', edgecolor='yellowgreen', linewidth=0.0)
    
    x4_mean = x4.mean(0)
    x4_error = [sem(x4[:,t]) for t in range(len(time)) ]
    
    l4, = ax.plot(time, x4_mean,'orangered', linewidth=0.5)
    ax.fill_between(time, x4_mean-x4_error, x4_mean+x4_error,  alpha=0.7, color = 'orangered',edgecolor='yellowgreen', linewidth=0.0)
      

    
    # l5, = ax.plot(time, np.multiply(entropy_effect,  -0.000006),'black', linewidth=2)
    # l6, = ax.plot(time, np.multiply(language_effect, -0.000004),'grey', linewidth=2)
    # l7,  = ax.plot(time, np.multiply(interaction, -0.000003),'red' ,linewidth=2)
    l7,  = ax.plot(time, np.multiply(interaction, -0.000006),'red' ,linewidth=2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.axes.xaxis.set_ticklabels([])
    # ax.axes.yaxis.set_ticklabels([])

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])


    # plot_lines2.append([l5, l6, l7])
    plot_lines2.append([l7])
    #legend2 = plt.legend(plot_lines2[0], ['Word Entropy Main effect','Language Main effect', 'Language - Entropy Interaction'],loc='upper center', bbox_to_anchor=(0.5, -0.05))
       
    # plt.gca().add_artist(legend1)
    # plt.gca().add_artist(legend2)
    plt.title(title)
    plt.xlabel("Center of Time Window (sec)")
    plt.ylabel('Accuracy Improvement $\mathregular{R^{2}}$')
    # plt.ylim(-0.000005,0.000015)
    plt.ylim(-0.000008,0.000035)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.tight_layout()
    ax.yaxis.set_major_locator(MaxNLocator(4))
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    #fig.set_size_inches(4, 2)
    
    plt.show()


def stacked_graph_time_interval(x2, x3, x4,legend, title ):


    fig, ax = plt.subplots()

    time = [ -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6 ,0.65, 0.7, 0.75]
    
    plt.stackplot(time,x2.mean(axis=0), x3.mean(axis=0), x4.mean(axis=0), labels=[legend[0], legend[1], legend[2]], colors  = [ "coral", "turquoise", "orchid"])
    

   
    plt.title(title)
    plt.xlabel("Center of Time Window (sec)")
    plt.ylabel('Accuracy Improvement $\mathregular{R^{2}}$')
    # plt.ylim(-0.000005,0.000015)
    plt.ylim(0,0.000035)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()
    

 
