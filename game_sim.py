#   Runs cooperation simulations to the standard set out in Santos and Pacheco's 2006 
# paper ``A new route to the evolution of co-operation'' on general networkx graphs.

#   This involves attaching a payoff and strategy to each graph node, and iteratively
# letting each node play a prisoner's dilemma type game against each of its neighbours.
# Strategies are updated according to the performance of neighbouring nodes, following
# a "finite population analogue of replicator dynamics".

#   This code does not include any graph construction, as it is intended to be used as a
# module independent of graph type. A graph should be constructed with integer indexing
# and should be passed as 'network' argument to init() and sim().

#   The init() function adds payoff and strategy fields to each node, and distributes
# strategies evenly and randomly amongst the nodes. The payoffs are initially set to zero.

#   The sim() function runs te game on the specified network with specified defection
# temptation B and default mixing length of 1e4 iterations, with data recorded over the 
# subsequent 1e3 iterations.


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import networkx as nx
import math as mt
from scipy import stats
from bitarray import bitarray
import pickle

global b, N, payoff

rd.seed(3)

# get data for specified network over full range of b values
def get_simulation_data(network, n_mix = 1e4, n_data = 1e3, n_steps = 20) :
  data = [None for x in range(n_steps)]
  for i in range(n_steps) :
    print(str(i/float(n_steps)*100)+'%')
    data[i] = full_sim(network, 1+i/float(n_steps), n_mix, n_data)
  return data

def full_sim(network, B, n_mix=int(1e4), n_data = int(1e3)) :
  init(network)
  data = sim(network, B, int(n_mix), int(n_data))
  return data
  
# initialise payoff and strategy fields
def init(network) :
  global N 
  N = len(network)
  cnt = 0
  network.graph['degrees'] = [network.degree(node) for node in range(len(network))]
  for i in range(N) :
    network.node[i]['payoff'] = 0
    network.node[i]['strategy'] = False
  while cnt < N/2 :
    s = mt.floor(rd.random()*N)
    if not network.node[s]['strategy'] :
      network.node[s]['strategy'] = True
      cnt = cnt + 1

# run simulation and return data from final 1e3 iterations
def sim(network, B, n_mix = int(1e4), n_data = int(1e3)) :
  global b
  global payoff
  b = B
  payoff = [[0, int(b*100.0)],[0, 100]]
  data = [bitarray([False for x in range(len(network))]) for y in range(int(n_data))]
  for i in range(n_mix) :
    run_network_game(network)
    strategy_update(network)  
  for j in range(n_data) :
    run_network_game(network)
    strategy_update(network)
    record_data(network, data[j])
  return data

# take single iteration snapshot of node strategies
def record_data(network, data_array) :
  for n in network.nodes() :
    data_array[n] = network.node[n]['strategy']


# update strategy of each node
def strategy_update(network) :
  new_strategies = [None for x in range(len(network))]
  for u in range(len(network)) :
    new_strategies[u] = new_strategy(network, u)
  for q in range(len(new_strategies)) :
    network.node[q]['strategy'] = new_strategies[q]

# return new strategy for a single node
def new_strategy(network, node) :
  global b
  v = rd.choice(network.neighbors(node))
  if network.node[node]['payoff'] < network.node[v]['payoff'] :
    p = (network.node[v]['payoff'] - network.node[node]['payoff'])/(max(network.graph['degrees'][node], network.graph['degrees'][v])*b)
    s = rd.random()
    if (s < p) :
      return network.node[v]['strategy']
  return network.node[node]['strategy']


# update node payoffs
def run_network_game(network) :
  for v in network.nodes() :
    network.node[v]['payoff'] = 0
  for e in network.edges() :
    play_game(network, e[0], e[1])


# single game between two nodes
def play_game(network, vertex1, vertex2) :
  global payoff
  v1 = network.node[vertex1]
  v2 = network.node[vertex2]
  s1 = v1['strategy']
  s2 = v2['strategy']
  v1['payoff'] += payoff[s1][s2]
  v2['payoff'] += payoff[s2][s1]
  
  
#  if network.node[vertex1]['strategy'] and network.node[vertex2]['strategy'] :
#    network.node[vertex1]['payoff'] += 1.0
#    network.node[vertex2]['payoff'] += 1.0
#  elif network.node[vertex1]['strategy'] and not network.node[vertex2]['strategy'] :
#    network.node[vertex1]['payoff'] += 0.0
#    network.node[vertex2]['payoff'] += b
#  elif not network.node[vertex1]['strategy'] and network.node[vertex2]['strategy'] :
#    network.node[vertex1]['payoff'] += b
#    network.node[vertex2]['payoff'] += 0.0
#  else :
#    network.node[vertex1]['payoff'] += 0.0
#    network.node[vertex2]['payoff'] += 0.0

    
    
# analysis

# averages over cooperation ratios from the sample data
def coop_ratio_average(data) :
  ratios = [None for x in range(len(data))]
  for i in range(len(data)) :
    ratios[i] = np.sum(data[i])/float(len(data[i]))
  return sum(ratios)/float(len(ratios))

 # gets average strategy over sample data for each node in network
def average_strategies(data) :
  n_data = len(data)
  n_nodes = len(data[0])
  strategies = [None for x in range(n_nodes)]
  for v in range(n_nodes) :
    sum = 0
    for s in range(n_data) :
      if data[s][v] :
        sum = sum + 1
    if sum > n_data/2 :
      strategies[v] = True
    else :
      strategies[v] = False
  return strategies

# cooperation ratio of average strategies
def average_coop_ratio(data) :
  strategies = average_strategies(data)
  return np.sum(strategies)/float(len(strategies))
  
# networkx functions

# plots the degree distribution
def plot_degree_distribution(network) :
  degrees = network.degree()
  values = sorted(set(degrees.values()))
  hist = [degrees.values().count(x) for x in values]
  plt.figure()
  plt.plot(values, hist, 'bo')
  plt.plot(values, hist, 'k-')
  plt.xlabel('degree')
  plt.ylabel('frequency')
  plt.title('degree distribution')
  plt.show()

# returns the average degree distribution  
def average_degree(network) :
  degrees = network.degree()
  return sum(degrees.values())/float(len(degrees))

# analysis

# cooperation ratio for network
def cooperation_ratio(network) :
  data = [False for x in range(len(network))]
  for i in network.nodes() :
    strategy = network.node[i]['strategy']
    data[i] = strategy
  coops = np.sum(data)
  ratio = coops/float(len(data))
  return ratio

# graph of cooperation ratio over b
def coop_ratio_graph(network, n_steps = 20) :
  global b
  graph = [[None for x in range(n_steps)] for y in range(2)]
  for i in range(n_steps) :
    data = None
    init(network)
    data = sim(network, 1+i/float(n_steps))
    graph[1][i] = average_coop_ratio(data)
    graph[0][i] = b
  return graph

# plot above graph
def plot_graph(graph, style) :
  plt.plot(graph[0], graph[1], style)
 


