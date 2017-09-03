import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd
import networkx as nx
import math as mt
from scipy import stats
from cycler import cycler
import pickle
import gs_data as gsd
import operator
import os
from sklearn.decomposition import PCA

global data_directory
data_directory = os.getcwd() + "\\data\\"


# pull the summary data into the simulation structure
def full_analysis(simulation) :
  per_network_cooperation_rate(simulation)
  cooperation_to_degree(simulation)

def coop_by_degree_list(simulation, b_list) :
  my_dpi = 96
  l = len(b_list)
  fig = plt.figure(figsize=(1100/my_dpi, l*300/my_dpi), dpi=my_dpi)
  for i in range(l) :
    fig.add_subplot(l, 1, i+1)
    coop_by_degree(simulation, b_list[i])
    plt.ylabel('b = '+str(b_list[i]), fontsize=17, weight='bold')
    plt.xlabel('')
  plt.xlabel('Degree', fontsize=17, weight='bold')
  plt.tight_layout()
  
def coop_by_degree(simulation, b) :
  my_dpi = 300
  fig = plt.figure(figsize=(6, 2), dpi=my_dpi)
  n_steps = len(simulation['coop_degree'])
  b_index = int(round((b-1)*n_steps))
  df = simulation['coop_degree'][b_index]
  
  degree = df['degree'].to_dict()
  coop = df.loc[df['coop_rate'] > 0.5]['degree'].to_dict()
  defect = df.loc[df['coop_rate'] <= 0.5]['degree'].to_dict()
  
  max_degree = max(degree.values())
  min_degree = min(degree.values())
  index = range(min_degree, max_degree+1)
  
  totals = [degree.values().count(x) for x in index]
  coop_counts = [coop.values().count(x) for x in index]
  defect_counts = [defect.values().count(x) for x in index]
  
  c_ratio = map(lambda x, y : x/float(y) if y>0 else 0, coop_counts, totals)
  d_ratio = map(lambda x, y : x/float(y) if y>0 else 0, defect_counts, totals)
  
  #fig = plt.figure(figsize=(1100/my_dpi, 400/my_dpi), dpi=my_dpi)
  
  p2 = plt.bar(index, d_ratio, bottom = c_ratio, color = '#adadad', label = 'Defectors')
  p1 = plt.bar(index, c_ratio, color = '#494949', label = 'Co-operators')
  #plt.title('Co-operation by Degree for '+simulation['graph_name']+': b=' + str(b), fontsize=20)
  plt.legend()
  plt.tick_params(labelsize=9)
  plt.xlim(min_degree-1, max_degree+1)
  plt.xlabel('Degree', fontsize=10, weight='bold')
  plt.ylabel('Co-op. Ratio', fontsize=10, weight='bold')
  plt.tight_layout()
    
# pyplot the degree distribution over all generated networks
def degree_distribution_plot(simulation) :
  hist = simulation['coop_degree'][0].degree.value_counts().sort_index()
  plt.plot(hist)
  plt.xlabel('degree')
  plt.ylabel('frequency')
  plt.title('degree distribution of '+simulation['graph_name'])
  #plt.show()

# Total ratio of cooperation
def overall_coop_measure(simulation) :
  n_steps = len(simulation['networks'][0]['average_cooperation_per_node'])
  n_networks = len(simulation['networks'])
  average_rate = [0. for x in range(n_steps)]
  for g in range(n_networks) :
    average_rate = map(lambda x,y : x+y, average_rate, simulation['networks'][g]['overall_average_cooperation'])
  average_rate = map(lambda x : x/float(n_networks), average_rate)
  overall = sum(average_rate)/(len(average_rate))
  return overall
  
# plot average cooperation ratio by b-values
def cooperation_by_b_plot(simulation, name=None) :
  lab = simulation['graph_name'] if name==None else name
  n_steps = len(simulation['networks'][0]['average_cooperation_per_node'])
  n_networks = len(simulation['networks'])
  average_rate = [0. for x in range(n_steps)]
  for g in range(n_networks) :
    average_rate = map(lambda x,y : x+y, average_rate, simulation['networks'][g]['overall_average_cooperation'])
  average_rate = map(lambda x : x/float(n_networks), average_rate)
  b = [1 + i/float(n_steps) for i in range(n_steps)]
  plt.plot(b, average_rate, linewidth=3.0, label=lab)
  plt.xlim(1.0, 2.0)
  plt.ylim(-0.05, 1.05)
  #plt.title('Co-operation rate over b for '+simulation['graph_name'], fontsize=20)
  plt.xlabel('b', fontsize=14, style='italic', weight='bold')
  plt.ylabel('Co-operation Ratio', fontsize=12, weight='bold')
  
# uses data for each node of each network to construct a dataframe 
# which compares degree to cooperation rate
def cooperation_to_degree(simulation) :
  n_steps = len(simulation['networks'][0]['average_cooperation_per_node'])
  n_networks = len(simulation['networks'])
  n_nodes = len(simulation['networks'][0]['network'])
  simulation['coop_degree'] = [None for x in range(n_steps)]
  for b in range(n_steps) :
    dict = {}
    degree = dict['degree'] = [None for x in range(n_nodes*n_networks)]
    coop_rate = dict['coop_rate'] = [None for x in range(n_nodes*n_networks)]
    for n in range(n_networks) :
      network = simulation['networks'][n]
      for v in range(n_nodes) :
        index = n*n_nodes + v
        degree[index] = network['network'].degree(v)
        coop_rate[index] = network['average_cooperation_per_node'][b][v]
    simulation['coop_degree'][b] = pd.DataFrame(dict)
    
# adds the per network data averages to the overall data structure, giving 
# per node data and overall data. per node will be useful for mapping against
# degree, whilst overall will be useful for overall graph behaviour.
def per_network_cooperation_rate(simulation) :
  networks = simulation['networks']
  n_networks = len(networks)
  for n in range(n_networks) :
    network = networks[n]
    network['average_cooperation_per_node'] = b_sim_coop_average(network['sims'])
    network['overall_average_cooperation'] = b_sim_mean_coop_average(network['average_cooperation_per_node'])

# takes output of "b_sim_coop_average" and averages over node to give
# b -> average_coop
def b_sim_mean_coop_average(averages) :
  return map(lambda z: sum(z)/len(z), averages)

# ADD THIS TO DATA STRUCTURE
# pass ['sims'] data to get average coop rate per node over all simulations
# as a relation on b (b -> sim_avg per node)
def b_sim_coop_average(sims) :
  n_steps = len(sims[0])
  n_sims = len(sims)
  averages = b_data_coop_average(sims[0])
  for s in range(1, n_sims) :
    avg = b_data_coop_average(sims[s])
    for b in range(n_steps) :
      averages[b] = map(operator.add, averages[b], avg[b])
  for b in range(n_steps) :
    averages[b] = map(lambda z: z/float(n_sims), averages[b])
  return averages
  
# pass a single simulation data ("model['networks'][N]['sims'][s]") structure, get b -> average per node
def b_data_coop_average(b_data) :
  n_steps = len(b_data)
  averages = [None for x in range(n_steps)]
  for b in range(n_steps) :
    averages[b] = coop_rate_per_node(b_data[b])
  return averages

# pass data for a single b iteration of a simulation. return cooperation rate
# as a decimal 0 < cr < 1 per node in graph
def coop_rate_per_node(data) :
  network_order = len(data[0])
  samples = len(data)
  averages = [0 for x in range(network_order)]
  for i in range(samples) :
    averages = map(operator.add, averages, map(int, data[i]))
  averages = map(lambda z: z/float(samples), averages)
  return averages
  
# draw cooperation by node spatially (only pass spatial networks)
def draw_spatial_cooperation(sim, network_index, b) :
  G = sim['networks'][network_index]['network']
  avg_coop_per_node = sim['networks'][network_index]['average_cooperation_per_node']
  coord = nx.get_node_attributes(G, 'pos')
  n_steps = len(avg_coop_per_node)
  b_index = int(round((b-1)*n_steps))
  coops = avg_coop_per_node[b_index]
  nx.draw_networkx_edges(G, pos = coord, alpha = 0.7)
  nx.draw_networkx_nodes(G, pos = coord, node_color = coops, cmap=plt.get_cmap('plasma'), node_size = 30)
  plt.xlim(-0.02, 1.02)
  plt.ylim(-0.02, 1.02)
  plt.show()
  
# Plot multiple cooperation by b graphs on the same axis
def mult_coop_by_b_plot(sim_dict) :
  n = len(sim_dict)
  my_dpi=300
  fig = plt.figure(figsize=(6, 4), dpi=my_dpi)
  ax = fig.add_subplot('111')
  ax.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) + cycler('linestyle', ['-', '--', '-.', ':']))
  filename = 'coop_by_b'
  for name, sim in sim_dict.iteritems() :
    filename = filename + '_' + name
    cooperation_by_b_plot(sim, name)
  #plt.title('Co-operation Ratios for Random Graphs', fontsize=40)
  ax.tick_params(labelsize=11)
  plt.legend(loc=(0.71, 0.735), prop={'size': 9})
  plt.tight_layout()
  plt.savefig(data_directory+filename+'_k=10_print.png')
  
# PCA on basic propeties
def graph_prop_pca(csv) :
  props = pd.read_csv(data_directory+csv, index_col = 0)
  props = props.T
  del [props['energy'], props['synergy'], props['reach']]
  props = props.T
  props_norm = norm_cols(props)
  pca = PCA()
  pca.fit(props_norm)
  pca_df = pd.DataFrame(pca.components_.T, columns = ['component_'+str(k) for k in range(len(pca.components_))], index = props_norm.columns)
  pca_df = pca_df.T
  pca_df['explained_var_ratio'] = pca.explained_variance_ratio_
  return pca_df.T


#
def mean(l) :
  return sum(l)/float(len(l))

def var(l) :
  m = mean(l)
  s = sum(map(lambda x: (x-m)**2, l))
  return s/float(len(l)-1)

def sd(l) :
  return mt.sqrt(var(l))
  
def norm_cols(df) :
  normed = props.apply(lambda col : map(lambda x : (x-mean(col))/sd(col), col))
  return normed
  
  
# Analysis of small-world properties vs ER and BA
def sw_plot(net_df) :
  props = {'assortativity': 'Assortativity', 'char_path_length': 'Char. Path', 'clustering': 'Clustering', 'heterogeneity': 'Heterogeneity', 'overall_coop': 'Overall Co-op.'}
  ER = {'assortativity': -0.056, 'char_path_length': 3.269, 'clustering': 0.009, 'heterogeneity': 0.014, 'overall_coop': 0.255}
  BA = {'assortativity': -0.055, 'char_path_length': 2.980, 'clustering': 0.040, 'heterogeneity': 0.138, 'overall_coop': 0.810}
  keys = sorted(props.keys())
  my_dpi = 300
  fig = plt.figure(figsize=(7, 5), dpi = my_dpi)
  cnt = 0
  for k in keys :
    cnt = cnt+1
    fig.add_subplot(2, 3, cnt)
    series = net_df.loc[k]
    plt.plot(series.index, series.values, 'k-', linewidth=2.5, label='REDS')
    plt.plot(series.index, [ER[k] for x in series.index], 'c--', linewidth=2, label = 'Erdos-Renyi')
    plt.plot(series.index, [BA[k] for x in series.index], 'm--', linewidth=2, label = 'Barabasi-Albert')
    plt.xlim(min(series.index), max(series.index))
    plt.ylabel(props[k], fontsize=12, fontweight='bold')
  plt.legend(loc=(1.3, 0.45))
  plt.tight_layout()
  plt.savefig(data_directory+'sw_plot.png', dpi= my_dpi)
    
  