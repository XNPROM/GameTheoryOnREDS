import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd
import networkx as nx
import math as mt
from scipy import stats
import pickle
import gs_data as gsd
import operator

# pull the summary data into the simulation structure
def full_analysis(simulation) :
  per_network_cooperation_rate(simulation)
  cooperation_to_degree(simulation)

def coop_ratio_by_degree(simulation, b) :
  n_steps = len(simulation['coop_degree'])
  b_index = int(round((b-1)*n_steps))
  df = simulation['coop_degree'][b_index]
  degrees = df.degree.value_counts().sort_index().index
  totals = df.degree.value_counts().sort_index()
  coops = df.loc[df['coop_rate'] > 0.5].degree.value_counts().sort_index()
  ratio = coops.divide(totals)
  plt.plot(degrees, ratio)
  plt.title('coop ratio by degree for '+simulation['graph_name'] + ' for b = ' + str(b))
  plt.xlabel('degree')
  plt.ylabel('ratio of cooperators')

# pyplot the degree distribution over all generated networks
def degree_distribution_plot(simulation) :
  hist = simulation['coop_degree'][0].degree.value_counts().sort_index()
  plt.plot(hist)
  plt.xlabel('degree')
  plt.ylabel('frequency')
  plt.title('degree distribution of '+simulation['graph_name'])
  #plt.show()

# plot average cooperation ratio by b-values
def cooperation_by_b_plot(simulation) :
  n_steps = len(simulation['networks'][0]['average_cooperation_per_node'])
  n_networks = len(simulation['networks'])
  average_rate = [0. for x in range(n_steps)]
  for g in range(n_networks) :
    average_rate = map(lambda x,y : x+y, average_rate, simulation['networks'][g]['overall_average_cooperation'])
  average_rate = map(lambda x : x/float(n_networks), average_rate)
  b = [1 + i/float(n_steps) for i in range(n_steps)]
  plt.plot(b, average_rate, 'b-')
  plt.xlim(0.975, 2.025)
  plt.title('cooperation rate over b for '+simulation['graph_name'])
  plt.xlabel('b')
  plt.ylabel('cooperation ratio')
  
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