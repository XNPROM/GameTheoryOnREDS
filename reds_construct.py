import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd
import networkx as nx
import math as mt
import scipy
from scipy import stats
from sklearn.preprocessing import normalize
import pickle
import operator
import os
import os.path
import time
import community
import statistics as stat
import graph_randomiser as gr
import seaborn as sns

rd.seed(4)

# directory for loading and saving data
global data_directory
data_directory = os.getcwd() + "\\data\\"

# construct and save several REDS graphs
def multiple_reds(n, r, e, s, n_graphs) :
  networks = [None for i in range(n_graphs)]
  for i in range(n_graphs) :
    networks[i] = reds_graph(n, r, e, s)
  filename = 'REDS_n='+str(n)+'_r='+str(r)+'_e='+str(e)+'_s='+str(s)+'_n_graphs='+str(n_graphs)+'.redsgraph'
  f = open(data_directory+'REDS\\'+filename, 'w')
  pickle.dump(networks, f)
  f.close()

# return a small-world REDS graph
def small_world_reds_graph(n, r, e, s, p, torus = True, suppress=True) :
  G = reds_graph(n, r, e, s, torus, suppress)
  G.graph['small-world'] = True
  gr.randomise_graph(G, p)
  return G
  
      
# return a networkx REDS graph with specified order, reach, energy and synergy
def reds_graph(n, r, e, s, torus=True, suppress=True, existing_nodes=None) :
  start = int(round(time.time()))
  G = nx.Graph()
  G.graph['reach'] = r
  G.graph['energy'] = e
  G.graph['synergy'] = s
  if existing_nodes == None :
    G.add_nodes_from(range(n))
    for i in range(len(G)) :
      G.node[i]['pos'] = [rd.random() for k in range(2)]
  else :
    G.add_nodes_from(existing_nodes.nodes(data=True))
    n = len(G)
  
  for i in range(len(G)) :
    G.node[i]['in_range'] = {}
    G.node[i]['available'] = set()
  for i in range(n-1) :
    u = G.node[i]
    for j in range(i, n) :
      v = G.node[j]
      if torus == True :
        d = torus_distance(u, v)
      else :
        d = distance(u, v)
      if d < r and u != v:
        u['in_range'][str(j)] = d
        v['in_range'][str(i)] = d
  for i in range(n) :
    update_theoreticals(G, i)
    
  running = True
  while running :
    if suppress == False :
      print('number of edges = '+str(G.size()))
    running = update(G, suppress)
  
  end = int(round(time.time()))
  print('total elapsed time = '+sec_to_string(end-start))
  print('graph complete')
  return G

# one update step
def update(G, suppress = False) :
  free_nodes = [k for k in range(len(G)) if len(G.node[k]['available']) > 0]
  if len(free_nodes) == 0:
    return False
  else :
    if suppress == False :
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
  total = 0.0
  for j in G.neighbors(i) :
    total = total + cost(G, i, j)
  return total
 
 # theoretical total costs of edge between i and j. if edge exists
 # cost is greater than total energy, as multi-graph not allowed
def theoretical_costs(G, i, j) :
  if G.has_edge(i, j) :
    cst = [G.graph['energy'] + 1, G.graph['energy'] + 1]
  else :
    #G.add_edge(i, j)
    cst = [current_cost(G, i) + cost(G, i, j), current_cost(G, j) + cost(G, i, j)]#[current_cost(G, i), current_cost(G, j)]
    #G.remove_edge(i, j)
  return cst

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
  x, y = map(lambda p, q: min(abs(p-q), 1-abs(p-q)), node_1['pos'], node_2['pos'])
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
  try : 
    G.graph['assortativity'] = nx.degree_assortativity_coefficient(G)
  except ValueError :
    G.graph['assortativity'] = 0
  if nx.is_connected(G) : 
    G.graph['char_path_length'] = nx.average_shortest_path_length(G)
  else :
    G.graph['char_path_length'] = 0
  degrees = G.degree().values()
  G.graph['max_degree'] = max(degrees)
  G.graph['mean_degree'] = sum(degrees)/float(len(degrees))
  G.graph['mode_degree'] = max(set(degrees), key = degrees.count)
  G.graph['median_degree'] = stat.median(degrees)
  G.graph['spectral_gap'] = pr_spectral_gap(G)
  G.graph['heterogeneity'] = estrada_heterogeneity(G)

# compute the spectral gap of the graph  
def pr_spectral_gap(G) :
  A = nx.adjacency_matrix(G)
  n = len(G)
  P = normalize(A, norm='l1', axis=1)
  eig = scipy.sparse.linalg.eigs(P, k=2, return_eigenvectors=False, which='LR')
  #print(eig)
  return np.real(max(eig)-min(eig)).item()

# compute estrada heterogeneity index of network
def estrada_heterogeneity(G) :
  n = len(G)
  k = [1/mt.sqrt(G.degree(v)) if G.degree(v) > 0 else 0 for v in range(n)]
  S = [(k[e[0]]-k[e[1]])**2 for e in G.edges()]
  rho = sum(S)/float(n - 2*mt.sqrt(n-1))
  return rho
  
# create RGG from REDS node position. reach defaults to original
def RGG_from_REDS(REDS, mean_deg=None, torus=True) :
  n = len(REDS)
  G = nx.Graph()
  G.add_nodes_from(REDS.nodes(data=True))
  dist = torus_distance if torus == True else distance
  if mean_deg == None :
    reach = REDS.graph['reach']
    for i in range(n-1) :
      u = G.node[i]
      for j in range(i, n) :
        v = G.node[j]
        if torus == True :
          d = torus_distance(u, v)
        else :
          d = distance(u, v)
        if d < reach :
          if i != j :
            G.add_edge(i, j)
  else :
    required_size = n*mean_deg/2
    edge_dists = {}
    for i in range(n-1) :
      u = G.node[i]
      for j in range(i+1, n) :
        v = G.node[j]
        edge_dists[dist(u, v)] = (i, j)
    cnt = 0
    for d in sorted(edge_dists) :
      G.add_edge(*edge_dists[d])
      cnt = cnt + 1
      if cnt >= required_size :
        G.graph['reach'] = d
        return G
  return G

def RGG_md_graph(n, mean_deg, torus=True) :
  dist = torus_distance if torus == True else distance
  G = nx.Graph()
  G.add_nodes_from(range(n))
  for i in range(len(G)) :
    G.node[i]['pos'] = [rd.random() for k in range(2)]
  required_size = n*mean_deg/2
  edge_dists = {}
  for i in range(n-1) :
    u = G.node[i]
    for j in range(i+1, n) :
      v = G.node[j]
      edge_dists[dist(u, v)] = (i, j)
  cnt = 0
  for d in sorted(edge_dists) :
    G.add_edge(*edge_dists[d])
    cnt = cnt + 1
    if cnt >= required_size :
      G.graph['reach'] = d
      return G
  G.graph['reach'] = mt.sqrt(2)
  return G
    

# drawing the graph
def draw_graph(G, fade_boundary_edges=True) :
  coord = nx.get_node_attributes(G, 'pos')
  degrees = nx.degree_centrality(G)
  if fade_boundary_edges == True :
    edge_dists={e: distance(G.node[e[0]], G.node[e[1]]) for e in G.edges()}
    unit_edges = []
    bound_edges = []
    for e, d in edge_dists.iteritems() :
      if d <= G.graph['reach'] :
        unit_edges.append(e)
      else :
        bound_edges.append(e)
    unit_costs = [cost(G, i, j) for i,j in unit_edges]
    bound_costs = [cost(G, i, j) for i,j in bound_edges]
    nx.draw_networkx_edges(G, pos = coord, edge_color = unit_costs, cmap=plt.get_cmap('gnuplot'), edgelist=unit_edges, alpha = 0.6)
    nx.draw_networkx_edges(G, pos = coord, edge_color = bound_costs, cmap=plt.get_cmap('gnuplot'), edgelist=bound_edges, alpha = 0.1)
  else :
    costs = [cost(G, i, j) for i,j in G.edges()]
    nx.draw_networkx_edges(G, pos = coord, edge_color = costs, cmap=plt.get_cmap('gnuplot'), alpha = 0.6)
  nx.draw_networkx_nodes(G, pos = coord, node_color = list(degrees.values()), cmap=plt.get_cmap('gnuplot'), node_size = 30)
  plt.xlim(-0.02, 1.02)
  plt.ylim(-0.02, 1.02)
  plt.xticks([])
  plt.yticks([])

# draw communities
def draw_communities(G, fade_boundary_edges=True) :
  coord = nx.get_node_attributes(G, 'pos')
  partition = community.best_partition(G)
  size = float(len(set(partition.values()))) 
  if fade_boundary_edges == True :
    edge_dists={e: distance(G.node[e[0]], G.node[e[1]]) for e in G.edges()}
    unit_edges = []
    bound_edges = []
    for e, d in edge_dists.iteritems() :
      if d <= G.graph['reach'] :
        unit_edges.append(e)
      else :
        bound_edges.append(e) 
  count = 0
  for com in set(partition.values()) :
    count = count + 1
    list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos = coord, nodelist = list_nodes, node_size = 20, node_color = [count/float(size) for x in list_nodes], cmap = plt.get_cmap('nipy_spectral'), vmin = 0, vmax = 1)
  if fade_boundary_edges == True :
    nx.draw_networkx_edges(G, coord, edgelist=unit_edges, alpha = 0.5)
    nx.draw_networkx_edges(G, coord, edgelist=bound_edges, alpha = 0.05)
  else :
    nx.draw_networkx_edges(G, coord, alpha = 0.3)
  plt.axis('scaled')
  plt.xlim(-0.02, 1.02)
  plt.ylim(-0.02, 1.02)
  plt.xticks([])
  plt.yticks([])
 
# draw communities non-spatially
def draw_communities_spring(G) :
  coord = nx.spring_layout(G)
  partition = community.best_partition(G)
  size = float(len(set(partition.values())))
  count = 0
  for com in set(partition.values()) :
    count = count + 1
    list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos = coord, nodelist = list_nodes, node_size = 20, node_color = [count/float(size) for x in list_nodes], cmap = plt.get_cmap('nipy_spectral'), vmin = 0, vmax = 1)
  nx.draw_networkx_edges(G, coord, alpha = 0.5)
  plt.xlim(-0.02, 1.02)
  plt.ylim(-0.02, 1.02)
  plt.show()

def draw_community_comparison(G, H) :
  my_dpi = 400
  fig = plt.figure(figsize=(8.425, 9.45), dpi = my_dpi)
  fig.add_subplot('221')
  plt.axis('scaled')
  draw_communities(G)
  plt.ylabel('High-Synergy', fontsize=21, weight='bold')
  fig.add_subplot('222')
  plt.axis('scaled')
  draw_graph(G)
  fig.add_subplot('223')
  plt.axis('scaled')
  draw_communities(H)
  plt.ylabel('Low-Synergy', fontsize=21, weight='bold')
  plt.xlabel('Communities', fontsize=21, weight='bold')
  fig.add_subplot('224')
  plt.axis('scaled')
  draw_graph(H)
  plt.xlabel('Degree', fontsize=21, weight='bold')
  plt.tight_layout()
  plt.savefig(data_directory+'community_comp_large.png', dpi = my_dpi)
 
def draw_strategy(G, sim_data, step, fade_boundary_edges=True) :
  coord = nx.get_node_attributes(G, 'pos')
  strats = {k: sim_data[step][k] for k in range(len(G))}
  coops = []
  defects = []
  for k, v in strats.iteritems() :
    if v is True :
      coops.append(k)
    else :
      defects.append(k)
  nx.draw_networkx_nodes(G, pos = coord, node_color = 'b', nodelist = coops, node_size = 4, label = 'C')
  nx.draw_networkx_nodes(G, pos = coord, node_color = 'r', nodelist = defects, node_size = 4, label = 'D')
  #plt.legend(prop={'size': 12})
 
  if fade_boundary_edges == True :
    edge_dists={e: distance(G.node[e[0]], G.node[e[1]]) 
                  for e in G.edges()}
    unit_edges = []
    bound_edges = []
    for e, d in edge_dists.iteritems() :
      if d <= G.graph['reach'] :
        unit_edges.append(e)
      else :
        bound_edges.append(e)
    nx.draw_networkx_edges(G, pos = coord, edgelist=unit_edges, alpha = 0.6)
    nx.draw_networkx_edges(G, pos = coord, edgelist=bound_edges, alpha = 0.1)
  else :
    nx.draw_networkx_edges(G, pos = coord, alpha = 0.6)
      
  plt.xlim(-0.02, 1.02)
  plt.ylim(-0.02, 1.02)
  plt.xticks([])
  plt.yticks([])

def draw_strategies(G, sim_data, time_steps) :
  my_dpi = 300
  l = len(time_steps)
  fig = plt.figure(figsize=(6, 4), dpi=my_dpi)
  for i in range(l) :
    fig.add_subplot(2, (l+1)/2, i+1)
    plt.axis('scaled')
    draw_strategy(G, sim_data, time_steps[i])
    plt.xlabel('t = '+str(time_steps[i]), fontsize=10, weight='bold')
  plt.tight_layout()
  plt.savefig(data_directory+'comm_invade.png', dpi = my_dpi)

# draw the graphs communities and the strategies at a particular 
# timestep in recorded simulation data
def draw_community_strategy(G, sim_data, step) :
  my_dpi = 96
  fig = plt.figure(figsize=(1100/my_dpi, 600/my_dpi), dpi=my_dpi)
  fig.add_subplot('121')
  draw_communities(G)
  plt.xlabel('Communities', fontsize=21, weight='bold')
  fig.add_subplot('122')
  draw_strategy(G, sim_data, step)
  plt.xlabel('Strategies after '+str(step)+' steps', fontsize=21, weight='bold')

  
# plot the degree distribution of a network  
def plot_degree_distribution(G) :
  degrees = G.degree()
  values = sorted(set(degrees.values()))
  hist = [degrees.values().count(x) for x in values]
  plt.plot(values, hist, 'g-', linewidth=3.0)
  plt.xlabel('Degree', fontsize = 15)
  plt.ylabel('Freq', fontsize = 15)
  #plt.xlim(0, 200)
  #plt.ylim(0, 500)
 
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
  
# load a graph of each type from folder
def load_graph_range(directory) :
  graphs = []
  k = 0
  for filename in os.listdir(data_directory + directory) :
    if os.path.splitext(filename)[1] == '.redsgraph' :
      f = open(data_directory+directory+'\\'+filename, 'r')
      nets = pickle.load(f)
      f.close()
      graphs.append(nets[0])
      k = k+1
  return graphs

# plot degree distributions of a range of reds networks
# inputs are a pandas dataframe with energy values as columns
# and synergy as rows, and the name of the directory in which to
# output.  
def plot_multiple_degree_dist(graph_df, dir) :
  df = graph_df.copy().T.iloc[::-1].T
  if '0.0' in df.columns :
    df.pop('0.0')
  plt.figure(figsize = (1600/my_dpi, 1200/my_dpi), dpi = my_dpi)
  fig = plt.gcf()
  ax = fig.add_subplot(111)
  ax.spines['top'].set_color('none')
  ax.spines['bottom'].set_color('none')
  ax.spines['left'].set_color('none')
  ax.spines['right'].set_color('none')
  ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
  k=0
  max_deg = max(df.applymap(lambda g : g.graph['max_degree']).apply(lambda col : max(col)))
  max_freq = max(df.applymap(lambda g : max([g.degree().values().count(x) for x in set(g.degree().values())])).apply(lambda col : max(col)))
  for e, v in df.iteritems() :
    for s, g in v.iteritems() :
      k=k+1
      fig.add_subplot(len(df.iloc[0]), len(df), k)
      plot_degree_distribution(g)
      plt.xlim(0, max_deg)
      plt.ylim(0, max_freq)
      plt.xticks([])
      plt.yticks([])
      if (k-1)/len(df) >= (len(df.iloc[0])-1) :
        plt.xlabel(s, fontsize=14)
      if k % len(df) == 1 : 
        plt.ylabel(e, fontsize=14)
  fig.suptitle('Degree distributions of R=0.1 REDS graphs', fontsize=20)
  ax.set_xlabel('Synergy', fontsize=20)
  ax.set_ylabel('Energy', fontsize=20)
  plt.savefig(data_directory+dir+'\\degree_dist.png', dpi = my_dpi)
  
  
def print_mult_avg_degree(graphs) :
  sort = sorted(graphs, key=lambda g: (g.graph['energy'], g.graph['synergy']))
  for k in range(len(graphs)) :
    g = sort[k].graph
    print('energy='+str(g['energy'])+'\t synergy='+str(g['synergy'])+'\t mean_degree='+str(g['mean_degree']))
  
def arrange(k) :
  return ((3-(k)/5)*5 + (k)%5 +1)
  
# range of REDS graphs over energy for specified synergy  
def double_peak_search(order, synergy, steps) :
  graphs = [None for k in range(steps)]
  for s in range(steps) :
    graphs[s] = reds_graph(order, 0.15, s/float(5*steps), synergy)
  return graphs

def mean_deg_search() :
  energy_index = [(x+1)/float(30) for x in range(9)]
  synergy_index = [y/float(10) for y in range(11)]
  graphs = {}
  for e in energy_index :
    graphs[str('%.3f' % e)] = {}
    for s in synergy_index :
      graphs[str(e)][str(s)] = reds_graph(1000, 0.15, e, s)
  return graphs

def reds_range() :
  energy_index = [x/float(30) for x in range(10)]
  synergy_index = [y/float(10) for y in range(11)]
  graphs = {}
  for e in energy_index :
    graphs[str(e)] = {}
    print('e='+str(e))
    for s in synergy_index :
      graphs[str(e)][str(s)] = reds_graph(1000, 0.1, e, s, suppress = True)
      print('s='+str(s))
  return graphs

#for k, v in md_search.iteritems() :
#  for l, g in v.iteritems() :
#    social_properties(g)
#md_prop = {k: {l: g.graph for l, g in v.iteritems()} for k, v in graphs.iteritems()}

def heatmaps(graph_df, dir) :
  tags = {'mean_degree': 'Mean Degree', 'clustering': 'Clustering Coefficient', 'assortativity': 'Assortativity', 'char_path_length': 'Characteristic Path Length'}
  my_dpi = 96
  for k, v in tags.iteritems() :
    print(k)
    plt.figure(figsize = (1200/my_dpi, 1200/my_dpi), dpi = my_dpi)
    df = graph_df.applymap(lambda p : p.graph[k]).T.iloc[::-1]
    df.index = map(lambda s : s[:4]+'..' if len(s) > 4 else s, df.index)
    sns.heatmap(df, annot=True, cmap = plt.get_cmap('gnuplot'))
    plt.title(v, fontsize=20)
    plt.tick_params(labelsize=14)
    plt.xlabel('Synergy', fontsize=20)
    plt.ylabel('Energy', fontsize=20)
    plt.savefig(data_directory+dir+'\\'+k+'.png', dpi = my_dpi)

def heatmaps_fig(graph_df, dir) :
  tags = {'mean_degree': 'Mean Degree', 'clustering': 'Clustering Coefficient', 'assortativity': 'Assortativity'}
  my_dpi = 96
  fig = plt.figure(figsize = (1100/my_dpi, 380/my_dpi), dpi = my_dpi)
  i = 0
  for k, v in tags.iteritems() :
    print(k)
    i = i+1
    ax=fig.add_subplot(1, 3, i)
    df = graph_df.applymap(lambda p : p.graph[k]).T.iloc[::-1]
    df.index = map(lambda s : s[:4]+'..' if len(s) > 4 else s, df.index)
    x_ticks=np.arange(0.0, 1.25, 0.25)
    sns.heatmap(df, cmap = plt.get_cmap('gnuplot'), xticklabels=x_ticks)
    plt.title(v, fontsize=16, weight='bold')
    plt.tick_params(labelsize=12)
    ax.set_xticks(x_ticks*ax.get_xlim()[1])
    plt.xlabel('Synergy', fontsize=12, weight='bold')
    plt.ylabel('Energy', fontsize=12, weight='bold')
    plt.yticks(rotation=0)
  plt.tight_layout()
  plt.savefig(data_directory+dir+'\\heatmaps_small.png', dpi = my_dpi)


def mean_deg_heatmap(md_prop) :
  md = {k: {l: u['mean_degree'] for l, u in v.iteritems()} for k, v in md_prop.iteritems()}
  md_df = pd.DataFrame(md).T.iloc[::-1]
  trunc_col = map(lambda s : s[:4]+'..' if len(s) > 4 else s, md_df.columns)
  md_df.columns = trunc_col
  sns.heatmap(md_df, annot=True, cmap = plt.get_cmap('gnuplot')) #color
  plt.title('Heatmap of mean degrees of REDS graphs over S and E', fontsize=20)
  plt.ylabel('Energy', fontsize=20)
  plt.xlabel('Synergy', fontsize=20)
  plt.show()

# Searches through R and E parameter space for n=1000, s = 1 
# networks to find connected mean degree 4 networks. Avoids the
# saturated region for performance.
def param_search(e_step, r_step) :
  n = 1000
  s = 1.0
  df = {}
  r = r_step
  e = e_step
  df[str(r)] = {}
  cnt = 0
  match = []
  while cnt < 10000 :
    print('r = '+str(r)+'      e = '+str(e))
    cnt = cnt+1
    G = reds_graph(n, r, e, s, suppress=True)
    social_properties(G)
    df[str(r)][str(e)] = G.graph
    if abs(float(G.graph['mean_degree'])-4) < 1 and nx.is_connected(G) :
      match.append(G.graph)
      print('MATCH!!!!!!!!!!!!!!!!!')
    if G.graph['mean_degree'] > 5 or e > 0.5:
      e = e_step
      r = r + r_step
      df[str(r)] = {}
    else :
      e = e + e_step
    if r > 0.5 :
      return [df, match]
  return [df, match]

  
def converge_to_RGG_plot(graph_df) :
  REDS = graph_df.T['1.0']
  RGG = REDS.map(lambda g : RGG_from_REDS(g))
  REDS_size = REDS.map(lambda g : g.size())
  RGG_size = RGG.map(lambda g : g.size())
  energy = map(float, REDS_size.index)
  my_dpi = 96
  fig = plt.figure(figsize=(1200/my_dpi, 1200/my_dpi), dpi=my_dpi)
  plt.plot(energy, REDS_size.values, 'g-', label='REDS')
  plt.plot(energy, RGG_size.values, 'b-', label='RGG')
  plt.show()


def setup_graph_dict() :
  graph_dict = {'regular': [], 'erdos-renyi' : [], 'reds' : [], 'rgg' : [], 'sw_reds' : [], 'barabasi-albert' : []}
  for i in range(10) :
    graph_dict['regular'].append(nx.watts_strogatz_graph(1000, 10, 0))
    graph_dict['erdos-renyi'].append(nx.watts_strogatz_graph(1000, 10, 1))
    graph_dict['barabasi-albert'].append(nx.barabasi_albert_graph(1000, 5))
    
    reds = reds_graph(1000, 0.1, 0.146, 1.0)
    while not nx.is_connected(reds) :
      reds = reds_graph(1000, 0.1, 0.146, 1.0)
      
    graph_dict['reds'].append(reds)
    graph_dict['rgg'].append(RGG_from_REDS(reds, mean_deg=10))
    
    sw_reds = small_world_reds_graph(1000, 0.1, 0.146, 1.0, 0.2)
    while not nx.is_connected(sw_reds) :
      sw_reds = small_world_reds_graph(1000, 0.1, 0.146, 1.0, 0.2)
      
    graph_dict['sw_reds'].append(sw_reds)
  
  for k, l in graph_dict.iteritems() :
    for g in l :
      social_properties(g)
  return graph_dict
  
def mean_properties(graph_dict) :
  n = len(graph_dict)
  props = {k: map(lambda g : g.graph, l) for k, l in graph_dict.iteritems()}
  means = {k: pd.DataFrame(l).mean() for k, l in props.iteritems()}
  return pd.DataFrame(means)
  
def compare_degree_distribution_plot(graph_dict) :
  n = len(graph_dict)    
  my_dpi = 300
  fig = plt.figure(figsize = (6, 2), dpi = my_dpi)
  i = 0
  filename = 'compare_degree'
  for k, v in graph_dict.iteritems() :
    i = i+1
    filename = filename+'_'+k
    ax = fig.add_subplot(1, n, i)
    average_degree_distribution(v)
    plt.title(k, fontsize=10, weight='bold')
    ax.tick_params(labelsize=9)
  plt.tight_layout()
  plt.savefig(data_directory+filename+'.png', dpi = my_dpi)
  
def average_degree_distribution(graph_list) :
  values_list = [set(g.degree().values()) for g in graph_list]
  values_set = reduce(lambda x,y : x.union(y), values_list)
  values = sorted(values_set)
  hist_list = [[g.degree().values().count(x) for x in values] for g in graph_list]
  hist = reduce(lambda x, y : map(lambda x_i, y_i : x_i + y_i, x, y), hist_list)
  norm_hist = map(lambda f : f/float(len(graph_list)), hist)
  plt.plot(values, norm_hist, 'g-', linewidth = 3.0)
  plt.xlabel('Degree', fontsize = 9, weight='bold')
  plt.ylabel('Freq.', fontsize = 9, weight='bold')

  
