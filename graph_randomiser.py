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
def randomise_graph(G):
  size = G.size()
  order = G.order()
  for i in range(size**2) :
    e1 = rd.choice(G.edges())
    e2 = rd.choice(G.edges())
    f1 = e1[0], e2[1]
    f2 = e2[0], e1[1]
    G.remove_edges_from([e1, e2])
    G.add_edges_from([f1, f2])

