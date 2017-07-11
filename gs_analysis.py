import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import networkx as nx
import math as mt
from scipy import stats
import pickle
import gs_data as gsd

# Find the average cooperation coefficient for each node in the graph-data
# structure inside ['networks'][i] 
def mean_node_coop_rate(net_data) :
  s = len(net_data['sims'])
  n = len(net_data['network'])
  rates = [[None for x in range(n)] for y in range(s)]
  for i in range(s) :
    rates[i] = node_coop_rate()
  mean_rates = [None for z in range(n)]
  for k in range(n) :
    total = 0
    for l in range(s) :
      total += rates[l][k]

# pass ['sims'] data to get average coop rate per node over all simulations
# as a relation on b (b -> sim_avg per node)
def b_sim_coop_average(sims) :
  n_steps = len(sims[0][0])
  averages = [[0 for x in range(n_steps)] for y in range(2)]
  for y in range(len(sims)) :
    b_data_avg = b_data_coop_average(sims[y])
    for b in range(n_steps) :
      averages[1][b] = map(operator.add, averages[1][b], b_data_avg[1][b])
  for b2 in range(n_steps) :
    averages[1][b2] = map(lambda z: z/float(len(sims)), averages[1][b2])
    averages[0][b2] = 1 + b2/float(len(sims[0][0]))

# pass a single simulation data ("model['networks'][N]['sims'][s]") structure, get b -> average per node
def b_data_coop_average(b_data) :
  averages = [[None for x in range(len(b_data[0]))] for y in range(2)]
  for b in range(len(b_data[0])) :
    averages[0][b] = b_data[0][b]
    averages[1][b] = coop_rate_per_node(b_data[1][b])
  return averages

# pass data for a single b iteration of a simulation. return cooperation rate
# as a decimal 0 < cr < 1 per node in graph
def coop_rate_per_node(data) :
  network_order = len(data[0])
  samples = len(data)
  averages = [None for x in range(network_order)]
  for j in range(network_order) :
    sum = 0
    for i in range(samples)
      sum += data[i][j]
    mean = sum/float(samples)
    averages[j] = mean
  return averages
