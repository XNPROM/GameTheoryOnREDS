# This module uses the game_sim module to run full simulation models 
# and return the data in a structured format. These data structures can
# then be saved with names generated from the graph type and simulation
# parameters.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import networkx as nx
import math as mt
from scipy import stats
import pickle
import game_sim as gs
import os
import time

# progress messages
global pm
pm = True

def p_message(message) :
  global pm
  if pm == True :
    print(message)


# directory for loading and saving data
global data_directory
data_directory = os.getcwd() + "\\data\\"

# set the data directory
def set_data_directory(directory) :
  global data_directory
  data_directory = directory

# print the current data directory
def print_data_directory() :
  global data_directory
  print(data_directory)
  
def sec_to_string(seconds) :
  hour = int(seconds)/3600
  secs = seconds - hour * 3600
  min = int(secs)/60
  secs = secs - min * 60
  return str(hour) + ':' + str(min) + ':' + str(secs)

# run a full simulation for with specified (networkx) graph constructor
# and specified parameters
def full_sim_for_family(graph_name, graph_constructor, params, n_networks=5, sims_per_network=5, burn_in=1e4, n_samples=1e3) :
  simulation = {}
  simulation['graph_name'] = graph_name
  simulation['params'] = params
  simulation['n_networks'] = n_networks
  simulation['sims_per_network'] = sims_per_network
  simulation['burn_in'] = burn_in
  simulation['n_samples'] = n_samples
  simulation['networks'] = [{} for x in range(n_networks)]
  start = int(round(time.time()))
  for i in range(n_networks) :
    p_message('network '+str(i+1)+' of '+str(n_networks)+': begin')
    network_data = simulation['networks'][i]
    network_data['network'] = graph_constructor(*params)
    network_data['sims'] = [None for y in range(sims_per_network)]
    for j in range(sims_per_network) :
      p = (i*sims_per_network + j)/float(n_networks*sims_per_network)
      elapsed = int(round(time.time())) - start
      if (p != 0) :
        remaining = int(elapsed*((1-p)/float(p)))
      else :
        remaining = 0
      p_message('elapsed time: ' + sec_to_string(elapsed) + ' |   remaining: ' + sec_to_string(remaining))
      p_message('network '+str(i+1)+' of '+str(n_networks)+': sim '+str(j+1)+' of '+str(sims_per_network)+': running')
      network_data['sims'][j] = gs.get_simulation_data(network_data['network'], burn_in, n_samples)
  elapsed = int(round(time.time())) - start
  p_message('total time taken: ' + sec_to_string(elapsed))
  return simulation

# save the simulation data with auto-generated name
def save_sim_data(simulation) :
  global data_directory
  s = simulation
  filename = s['graph_name'] + "_params=" + '-'.join(map(str, s['params'])) + "_nets=" + str(s['n_networks']) + "_sims=" + str(s['sims_per_network']) + ".gsdata"
  f = open(data_directory+filename, 'w')
  pickle.dump(simulation, f)
  f.close()

# load the simulation data with specified name
def load_sim_data(filename) :
  global data_directory
  f = open(data_directory+filename, 'r')
  sim = pickle.load(f)
  f.close()
  return sim

#erdos = gsd.full_sim_for_family("erdos_renyi", nx.watts_strogatz_graph, [1000, 4, 1], 1, 1, 10000, 1000)

