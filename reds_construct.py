import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd
import networkx as nx
import math as mt
from scipy import stats
import pickle
import operator
import os
import time
import community
import statistics as stat

rd.seed(4)

# directory for loading and saving data
global data_directory
data_directory = os.getcwd() + "\\data\\"


# return a networkx REDS graph with specified order, reach, energy and synergy
def reds_graph(n, r, e, s) :
  start = int(round(time.time()))
  G = nx.Graph()
  G.graph['reach'] = r
  G.graph['energy'] = e
  G.graph['synergy'] = s
  G.add_nodes_from(range(n))
  
  for i in range(len(G)) :
    G.node[i]['pos'] = [rd.random() for k in range(2)]
    G.node[i]['in_range'] = {}
    G.node[i]['available'] = set()
  for i in range(n-1) :
    u = G.node[i]
    for j in range(i, n) :
      v = G.node[j]
      d = distance(u, v)
      if d < r :
        u['in_range'][str(j)] = d
        v['in_range'][str(i)] = d
  for i in range(n) :
    update_theoreticals(G, i)
    
  running = True
  while running :
    print('number of edges = '+str(G.size()))
    running = update(G)
  
  end = int(round(time.time()))
  print('total elapsed time = '+sec_to_string(end-start))
  print('graph complete')
  return G

# one update step
def update(G) :
  free_nodes = [k for k in range(len(G)) if len(G.node[k]['available']) > 0]
  if len(free_nodes) == 0:
    return False
  else :
    print('                           free nodes = '+str(len(free_nodes)))
    i = rd.choice(free_nodes)
    j = rd.sample(G.node[i]['available'], 1)[0]
    G.add_edge(i, j)
    update_theoreticals(G, i)
    update_theoreticals(G, j)
    return True
  
# cost of an edge between i and j in graph G
def cost(G, i, j) :
  cost = in_range_distance(G, i, j)/float(1 + G.graph['synergy']*len(list(set(G.neighbors(i)) & set(G.neighbors(j)))))
  return cost

# current energy expended on maintaining edges by i
def current_cost(G, i) :
  total = 0
  for j in G.neighbors(i) :
    total += cost(G, i, j)
  return total
 
 # theoretical total costs of edge between i and j. if edge exists
 # cost is greater than total energy, as multi-graph not allowed
def theoretical_costs(G, i, j) :
  if G.has_edge(i, j) :
    cost = [G.graph['energy'] + 1, G.graph['energy'] + 1]
  else :
    G.add_edge(i, j)
    cost = [current_cost(G, i), current_cost(G, j)]
    G.remove_edge(i, j)
  return cost

# update set of possible edges for edge i
def update_theoreticals(G, i) :
  u = G.node[i]
  for key, value in G.node[i]['in_range'].iteritems() :
    j = int(key)
    v = G.node[j]
    tc = theoretical_costs(G, i, j)
    if tc[0] < G.graph['energy'] and tc[1] < G.graph['energy'] :
      u['available'].add(j)
      v['available'].add(i)
    else :
      if j in u['available'] : u['available'].remove(j)
      if i in v['available'] : v['available'].remove(i)
 
# distance between i and j. returns none if out of range
def in_range_distance(G, i, j) :
  f = G.node[i]['in_range'].get(str(j))
  return f

# check if node with index j is in range of node_1
def in_range(G, i, j) :
  f = in_range_distance(G, i, j)
  if f == None :
    return False
  else :
    return True

# generate a random node position
def random_node_pos() :
  pos = {}
  pos['x'] = rd.random()
  pos['y'] = rd.random()
  return pos

# distance between two nodes computational
def distance(node_1, node_2) :
  x, y = map(lambda p, q: p-q, node_1['pos'], node_2['pos'])
  dist = mt.sqrt(x**2 + y**2)
  return dist

# distance between two nodes on a torus computational
def torus_distance(node_1, node_2) :
  x, y = map(lambda p, q: divmod(p-q, 0.5), node_1['pos'], node_2['pos'])
  dist = mt.sqrt(x**2 + y**2)
  return dist
  
def sec_to_string(seconds) :
  hour = int(seconds)/3600
  secs = seconds - hour * 3600
  min = int(secs)/60
  secs = secs - min * 60
  return str(hour) + ':' + str(min) + ':' + str(secs)

# add graph analysis
def social_properties(G) :
  G.graph['clustering'] = nx.average_clustering(G)
  G.graph['transitivity'] = nx.transitivity(G)
  G.graph['assortativity'] = nx.degree_assortativity_coefficient(G)
  G.graph['char_path_length'] = nx.average_shortest_path_length(G)
  degrees = G.degree().values()
  G.graph['max_degree'] = max(degrees)
  G.graph['mean_degree'] = sum(degrees)/float(len(degrees))
  G.graph['mode_degree'] = max(set(degrees), key = degrees.count)
  G.graph['median_degree'] = stat.median(degrees)
  
  
# drawing the graph
def draw_graph(G) :
  coord = nx.get_node_attributes(G, 'pos')
  degrees = nx.degree_centrality(G)
  costs = [cost(G, i, j) for i,j in G.edges()]
  nx.draw_networkx_edges(G, pos = coord, edge_color = costs, alpha = 0.7)
  nx.draw_networkx_nodes(G, pos = coord, node_color = list(degrees.values()), node_size = 30)
  plt.xlim(-0.02, 1.02)
  plt.ylim(-0.02, 1.02)
  plt.show()

# draw communities
def draw_communities(G) :
  coord = nx.get_node_attributes(G, 'pos')
  partition = community.best_partition(G)
  size = float(len(set(partition.values())))
  count = 0
  for com in set(partition.values()) :
    count = count + 1
    list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos = coord, nodelist = list_nodes, node_size = 30, node_color = [count/float(size) for x in list_nodes], cmap = plt.get_cmap('nipy_spectral'), vmin = 0, vmax = 1)
  nx.draw_networkx_edges(G, coord, alpha = 0.5)
  plt.xlim(-0.02, 1.02)
  plt.ylim(-0.02, 1.02)
  plt.show()
 
# draw communities non-spatially
def draw_communities_spring(G) :
  coord = nx.spring_layout(G)
  partition = community.best_partition(G)
  size = float(len(set(partition.values())))
  count = 0
  for com in set(partition.values()) :
    count = count + 1
    list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos = coord, nodelist = list_nodes, node_size = 30, node_color = [count/float(size) for x in list_nodes], cmap = plt.get_cmap('nipy_spectral'), vmin = 0, vmax = 1)
  nx.draw_networkx_edges(G, coord, alpha = 0.5)
  plt.xlim(-0.02, 1.02)
  plt.ylim(-0.02, 1.02)
  plt.show()
 
 
# save graph
def save_graph(G) :
  global data_directory
  filename = "REDS_" + "n=" + str(len(G)) + "_r=" + str(G.graph['reach']) + "_e=" + str(G.graph['energy']) + "_s=" + str(G.graph['synergy']) + ".graph"
  f = open(data_directory+filename, 'w')
  pickle.dump(G, f)
  f.close()

# load graph
def load_graph(filename) :
  global data_directory
  f = open(data_directory+filename, 'r')
  G = pickle.load(f)
  f.close()
  return G
  

#REDS = reds_graph(3000, 0.05, 0.09, 1)