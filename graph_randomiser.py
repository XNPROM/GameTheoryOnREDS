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

# randomly and repeatedly switch the ends of pairs of edges 
# in the graph. this preserves degree distribution, but removes
# phenomenae such as age-correlation in barabasi-albert graphs
def randomise_graph(G, p):
  swap = []
  
  for e in G.edges() :
    s = 0
    if (p < 1) :
      s = rd.random()
    if s < p :
      swap.append(e[0])
      swap.append(e[1])
      G.remove_edge(*e)
  
  rd.shuffle(swap)
  
  while len(swap) > 0 :
    u = swap.pop()
    v = swap.pop(rd.randrange(len(swap)))
    if G.has_edge(u, v) or u == v :
      swap.append(u)
      swap.append(v)
    else :
      G.add_edge(u, v)

def random_scale_free_graph(n, m, p=1) :
  G = nx.barabasi_albert_graph(n, m)
  randomise_graph(G, p)
  return G

#  