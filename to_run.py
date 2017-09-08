# other script
import graph_randomiser as gr 
import gs_data as gsd

sim = gsd.full_sim_for_family('random_scale_free', gr.random_scale_free_graph, [1000, 2], 5, 5)
gsd.save_sim_data(sim)



reds_line = plt.plot(map(lambda g : g.graph['energy'], REDS_list_s05), map(lambda g : g.size(), REDS_list_s05), 'b-', label = 'REDS')
rgg_line = plt.plot(map(lambda g : g.graph['energy'], RGG_list), map(lambda g : g.size(), RGG_list), 'g-', label = 'RGG')
four_line = plt.plot(map(lambda g : g.graph['energy'], RGG_list), [4000 for x in range(20)], 'r-', label = '4000')
plt.xlabel('energy')
plt.ylabel('size')
plt.legend()
plt.title('Graph Size against Energy for S=0.5 REDS Networks. Converges to RGG network as Energy increases.')
plt.show()


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')


range1 = reds_range()
range1_df = pd.DataFrame(range1)
f = open(data_directory+'REDSrange1\\graph_df.redsgraph', 'w')
pickle.dump(range1_df, f)
f.close()


reds_sim = gsd.full_sim_for_family('REDS', rc.reds_graph, [1000, 0.1, 0.15, 1.0], 5, 5, 1e4, 1e3)
gsd.save_sim_data(reds_sim)


cProfile.run('full_sim(G, 1.6, 1e3, 1e2)', sort='tottime')


graph_dict = setup_graph_dict()
compare_degree_distribution_plot(graph_dict)

# Mockup
import reds_construct as rc
import gs_data as gsd
import gs_analysis as gsa
import networkx as nx
import matplotlib.pyplot as plt

erdos_sim = gsd.full_sim_for_family('erdos_renyi', nx.watts_strogatz_graph, [1000, 10, 1], 5, 5, 1e4, 1e3)
gsa.full_analysis(erdos_sim)
gsd.save_sim_data(erdos_sim)

barabasi_sim = gsd.full_sim_for_family('barabasi_albert', nx.barabasi_albert_graph, [1000, 5], 5, 5, 1e4, 1e3)
gsa.full_analysis(barabasi_sim)
gsd.save_sim_data(barabasi_sim)

rsf_sim = gsd.full_sim_for_family('random_scale_free', gr.random_scale_free_graph, [1000, 5], 5, 5, 1e4, 1e3)
gsa.full_analysis(rsf_sim)
gsd.save_sim_data(rsf_sim)

reds_sim = gsd.full_sim_for_family('reds', rc.reds_graph, [1000, 0.1, 0.146, 1.0], 5, 5, 1e4, 1e3)
gsa.full_analysis(reds_sim)
gsd.save_sim_data(reds_sim)

rgg_sim = gsd.full_sim_for_family('rgg', rc.RGG_md_graph, [1000, 10], 5, 5, 1e4, 1e3)
gsa.full_analysis(rgg_sim)
gsd.save_sim_data(rgg_sim)

sw_reds_sim = gsd.full_sim_for_family('smallworld_reds', rc.small_world_reds_graph, [1000, 0.1, 0.146, 1.0, 0.2], 5, 5, 1e4, 1e3)
gsa.full_analysis(sw_reds_sim)
gsd.save_sim_data(sw_reds_sim)

regular_sim = gsd.full_sim_for_family('regular', nx.watts_strogatz_graph, [1000, 10, 0], 5, 5, 1e4, 1e3)
gsa.full_analysis(regular_sim)
gsd.save_sim_data(regular_sim)

comm_rand_reds_sim = gsd.full_sim_for_family('comm_rand_reds', rc.small_world_reds_graph, [1000, 0.1, 0.146, 1.0, 1.0], 5, 5, 1e4, 1e3)
gsa.full_analysis(comm_rand_reds_sim)
gsd.save_sim_data(comm_rand_reds_sim)


comm_reds_sim = gsd.full_sim_for_family('comm_reds', rc.reds_graph, [1000, 0.1, 0.146, 1.0], 5, 5, 1e4, 1e3, comm_init=True)
gsa.full_analysis(comm_reds_sim)
gsd.save_sim_data(comm_reds_sim)


comm_erdos_sim = gsd.full_sim_for_family('comm_erdos_renyi', nx.watts_strogatz_graph, [1000, 10, 1], 5, 5, 1e4, 1e3, comm_init=True)
gsa.full_analysis(comm_erdos_sim)
gsd.save_sim_data(comm_erdos_sim)

comm_barabasi_sim = gsd.full_sim_for_family('comm_barabasi_albert', nx.barabasi_albert_graph, [1000, 5], 5, 5, 1e4, 1e3, comm_init=True)
gsa.full_analysis(comm_barabasi_sim)
gsd.save_sim_data(comm_barabasi_sim)

sim_dict1 = {'Regular': regular_sim, 'Erdos-Renyi': erdos_sim, 'Barabasi-Albert': barabasi_sim, 'Random scale-free': rsf_sim}

sim_dict2 = {'Regular': regular_sim, 'Erdos-Renyi': erdos_sim, 'REDS': reds_sim, 'Random scale-free': rsf_sim}

sim_dict3 = {'REDS': reds_sim, 'Small-world REDS': sw_reds_sim, 'RGG': rgg_sim}

mult_coop_by_b_plot(sim_dict1)
mult_coop_by_b_plot(sim_dict2)
mult_coop_by_b_plot(sim_dict3)



regular = gsd.load_sim_data('regular_params=1000-10-0_nets=5_sims=5.gsdata')
erdos = gsd.load_sim_data('erdos_renyi_params=1000-10-1_nets=5_sims=5.gsdata')
barabasi = gsd.load_sim_data('barabasi_albert_params=1000-5_nets=5_sims=5.gsdata')
r_scale_free = gsd.load_sim_data('random_scale_free_params=1000-5_nets=5_sims=5.gsdata')
reds = gsd.load_sim_data('reds_params=1000-0.1-0.146-1.0_nets=5_sims=5.gsdata')
sw_reds = gsd.load_sim_data('smallworld_reds_params=1000-0.1-0.146-1.0-0.2_nets=5_sims=5.gsdata')
rgg = gsd.load_sim_data('rgg_params=1000-10_nets=5_sims=5.gsdata')

reds_sim = gsd.load_sim_data('reds_params=1000-0.1-0.146-1.0_nets=5_sims=5.gsdata')
comm_reds_sim = gsd.load_sim_data('comm_reds_params=1000-0.1-0.146-1.0_nets=5_sims=5.gsdata')
rand_reds_sim = gsd.load_sim_data('rand_reds_params=1000-0.1-0.146-1.0-1.0_nets=5_sims=5.gsdata')
comm_rand_reds_sim = gsd.load_sim_data('comm_rand_reds_params=1000-0.1-0.146-1.0-1.0_nets=5_sims=5.gsdata')

sim_dict_redsold = {'REDS': reds, 'Erdos-Renyi': erdos, 'Barabasi-Albert': barabasi, 'Regular': regular}

sim_dict = {'Standard': reds_sim, 'Random': rand_reds_sim, 'Community-Random': comm_rand_reds_sim, 'Community': comm_reds_sim}

def reds_e_range(n, r, s, num, max_e, min_e=0, depth=1) :
  step = (max_e-min_e)/float(num)
  e_list = np.arange(min_e, max_e+step, step)
  cnt = 0
  total = depth * num
  dict = {}
  for e in e_list :
    dict[str(e)] = []
    arr = dict[str(e)]
    for x in range(depth) :
      print('graph '+str(cnt)+' of '+str(total))
      arr.append(reds_graph(n, r, e, s))
      cnt = cnt+1
  return dict
  
def con_size(dict) :
  ndict = {'size': {}, 'connect': {}}
  ndict['size'] = {k: map(lambda g: g.size(), l) for k, l in dict.iteritems()}
  ndict['connect'] = {k: map(nx.is_connected, l) for k, l in dict.iteritems()}
  return ndict
  
def connected_hist(connected) :
  my_dpi=300
  keys = sorted([float(k) for k, v in connected.iteritems()])
  rate_connected = [mean(connected[str(k)]) for k in keys]
  fig = plt.figure(figsize=(6, 2), dpi = my_dpi)
  predictor = [connected_prob(k) for k in keys]
  plt.plot(keys, predictor, 'r-', linewidth=3.0, label='Predictor')
  plt.plot(keys, rate_connected, 'bo', label='Observed')
  plt.legend()
  plt.tick_params(labelsize=9)
  plt.ylabel('Prob. Connected', fontsize=10, weight='bold')
  plt.xlabel('E', fontsize=10, weight='bold', fontstyle='italic')
  plt.xlim(min(keys), max(keys))
  plt.tight_layout()
  plt.savefig(data_directory+'connect_hist_pred.png', dpi=my_dpi)
  
def md_graph(size) :
  my_dpi=300
  keys = sorted([float(k) for k, v in size.iteritems()])
  md = {k: map(lambda s: s*2/float(1000), l) for k, l in size.iteritems()}
  mean_md = [mean(md[str(k)]) for k in keys]
  error = [mt.sqrt( var(md[str(k)]) / len(md[str(k)]) ) for k in keys]
  lbound = map(lambda x, y: x-y, mean_md, error)
  ubound = map(lambda x, y: x+y, mean_md, error)
  fig = plt.figure(figsize=(6, 3), dpi = my_dpi)
  plt.plot(keys, ubound, color = '#adadad', linestyle='--', label='Standard Error')
  plt.plot(keys, lbound, color = '#adadad', linestyle='--')
  plt.plot(keys, mean_md, color = '#494949', linewidth=3.0, label='Mean Degree')
  plt.ylabel('Mean Degree', fontsize=10, weight='bold')
  plt.xlabel('E', fontsize=10, weight='bold', fontstyle='italic')
  plt.tight_layout()
  plt.savefig(data_directory+'mean_deg_plot.png', dpi = my_dpi)

def connected_prob(E) :
  return(np.tanh(100.5*E - 13.65)+1)/2
  
def connected_quant(p) :
  if p >= 1 or p <= 0 :
    return None
  else :
    return (np.arctanh(2*p - 1) + 13.65)/100.5

def size_range(n, r, s, min_e, max_e, num, depth) :
  step = (max_e-min_e)/float(num)
  e_list = np.arange(min_e, max_e+step, step)
  cnt = 0
  total = depth * (num+1)
  dict = {'size': {}, 'max': {}}
  for e in e_list :
    dict['size'][str(e)] = []
    dict['max'][str(e)] = []
    s_arr = dict['size'][str(e)]
    m_arr = dict['max'][str(e)]
    for x in range(depth) :
      print('graph '+str(cnt)+' of '+str(total))
      G = reds_graph(n, r, e, s)
      R = RGG_from_REDS(G)
      s_arr.append(G.size())
      m_arr.append(R.size())
      del G
      del R
      cnt = cnt+1
  return dict
  
def box_unpack(size_dict) :
  s = size_dict['size']
  m = size_dict['max']
  E = []
  S = []
  M = []
  E = sorted([float(k) for k in s.keys()])
  for k in E:
    S.append([])
    M.append([])
    for i in range(len(s[str(k)])) :
      S[-1].append(s[str(k)][i])
      M[-1].append(m[str(k)][i])
  return {'E': np.array(E), 'S': np.array(S), 'M': np.array(M)}

def boxplot_size(box_dict) :
  my_dpi = 300
  s_means = box_dict['S'].mean(axis=1)
  m_means = box_dict['M'].mean(axis=1)
  e = box_dict['E']
  fig = plt.figure(figsize=(6, 3), dpi = my_dpi)
  plt.plot(e, m_means, 'r--', linewidth=2.0, label='Maximum Size')
  plt.plot(e, s_means, 'b-', linewidth=3.0, label='Actual Size')
  plt.xlim(min(e)-0.01, max(e)+0.01)
  locs, labels = plt.xticks()
  plt.boxplot(box_dict['S'].transpose(), showfliers=False, positions=e, widths = 0.01)
  plt.boxplot(box_dict['M'].transpose(), showfliers=False, positions=e, widths = 0.01)  
  #plt.xticks(locs)
  plt.xlim(min(e)-0.01, max(e)+0.01)
  plt.ylabel('Network Size', fontsize=10, weight='bold')
  plt.xlabel('E', fontsize=10, weight='bold', fontstyle='italic')
  plt.legend()
  plt.tight_layout()
  plt.savefig(data_directory+'size_box.png', dpi = my_dpi)
  
def size_prob(E) :
  return(np.tanh(15.95*E - 2.585)+1)/2
  
def size_quant(p) :
  return (np.arctanh(2*p-1)+2.585)/15.95

def size_predict(E, max=15600) :
  return np.random.binomial(max, size_prob(E))
  
def size_predict_exp(E, max=15600) :
  return max*size_prob(E)
  
def size_quant_exp(s, max=15600) :
  return size_quant(s/float(max))
 
def size_cdf(E, s, max=15600) :
  p = size_prob(E)
  n = max
  return stats.binom.cdf(s, n, p)
 
def size_quant(E, q, max=15600) :
  p = size_prob(E)
  n = max
  return stats.binom.ppf(q, n, p)
 
def size_pred_compare(df_sort, predictor=size_predict_exp) :
  my_dpi = 300
  predict = size_predict if predictor == None else predictor
  fig = plt.figure(figsize=(6, 3), dpi = my_dpi)
  e_pred = np.arange(0, 0.31, 0.01)
  s_pred = [predict(e) for e in e_pred]
  plt.plot(e_pred, s_pred, 'r-', linewidth=3.0, label = 'Predicted Size')
  plt.plot(df_sort['E'], df_sort['S'], 'bo', label='Observed Size')
  plt.ylabel('Network Size', fontsize=10, weight='bold')
  plt.xlabel('E', fontsize=10, weight='bold', fontstyle='italic')
  plt.legend()
  plt.tight_layout()
  plt.savefig(data_directory+'size_box_pred_comp2.png', dpi = my_dpi)
 
  
def unpack(size_dict) :
  s = size_dict['size']
  m = size_dict['max']
  E = []
  S = []
  M = []
  for k, v in s.iteritems():
    for i in range(len(v)) :
      E.append(float(k))
      S.append(s[k][i])
      M.append(m[k][i])
  df = pd.DataFrame({'E': E, 'S': S, 'M': M})
  return df
    
  
def plot_path_lengths(e_range) :
  lengths = [e.graph['char_path_length'] for e in e_range]
  energies = [e.graph['energy'] for e in e_range]
  my_dpi = 300
  fig = plt.figure(figsize=(6, 4), dpi = my_dpi)
  plt.plot(energies, lengths, 'r-', linewidth=6.0)
  plt.xlabel('Energy', fontsize=17, style='italic', weight='bold')
  plt.ylabel('Char. Path Length', fontsize=17, weight='bold')
  plt.tight_layout()
  plt.savefig(data_directory+'char_paths_narrow.png', dpi = my_dpi)

def plot_connected(e_range) :
  energies = [e.graph['energy'] for e in e_range]
  connected = []
  disconnected = []
  for i in range(len(e_range)) :
    if e_range[i].graph['char_path_length']>0 :
      connected.append(energies[i])
    else :
      disconnected.append(energies[i])
  my_dpi = 300
  m_size = 10
  fig = plt.figure(figsize=(6, 1.75), dpi = my_dpi)
  ax = plt.axes(frameon=False)
  ax.axes.get_yaxis().set_visible(False)
  plt.plot(connected, np.zeros_like(connected), 'bx', label='Connected', markersize=m_size)
  plt.plot(disconnected, np.zeros_like(disconnected), 'ro', label='Disconnected', markersize=m_size)
  plt.xlabel('Energy', fontsize=17, style='italic', weight='bold')
  plt.yticks([])
  plt.ylim(-0.05, 0.2)
  xmin = min(energies)
  xmax = max(energies)
  ymin, ymax = ax.get_yaxis().get_view_interval()
  ax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
  plt.legend()
  plt.tight_layout()
  plt.savefig(data_directory+'is_connected_narrow.png', dpi = my_dpi)
  
def svm(e_range, l=0, steps=80) :
  energies = [e.graph['energy'] for e in e_range]
  e_max = max(energies)
  e_min = min(energies)
  s = 10/float(e_max-e_min)
  step = (e_max-e_min)/steps
  D = [{'y' : 1 if e.graph['char_path_length']>0 else -1, 'e': e.graph['energy']} for e in e_range]
  W = np.arange(e_min, e_max+step, step)
  loss = [svm_loss(D, w, s, l) for w in W]
  return {'w' : W, 'loss' : loss}
  
def score(d, w, s) :
  return s*d['y']*(d['e']-w)

def svm_loss(D, w, s, l) :
  hinges = [max(0, 1-score(d, w, s)) for d in D]
  return (sum(hinges)/len(hinges)+l*s)

def mult_coop_by_b_plot_poster(sim_dict) :
  n = len(sim_dict)
  my_dpi=96
  fig = plt.figure(figsize=(10.1, 8.44), dpi=my_dpi)
  ax = fig.add_subplot('111')
  ax.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) + cycler('linestyle', ['-', '--', '-.', ':']))
  filename = 'coop_by_b'
  for name, sim in sim_dict.iteritems() :
    filename = filename + '_' + name
    gsa.cooperation_by_b_plot(sim, name)
  #plt.title('Co-operation Ratios for Random Graphs', fontsize=40)
  ax.tick_params(labelsize=14)
  plt.xlabel('Temptation to Defect', fontsize=17, weight='bold')
  plt.legend(loc=(0.7, 0.82), prop={'size': 16})
  plt.tight_layout()
  plt.savefig(data_directory+'POSTER_'+filename+'_k=10_small.png', dpi=my_dpi)

  

dict = {'barabasi-albert': [nx.barabasi_albert_graph(1000, 5) for x in range(10)], 'erdos-renyi': [nx.watts_strogatz_graph(1000, 10, 1) for x in range(10)], 'reds': [reds_graph(1000, 0.1, 0.146, 1.0) for x in range(10)]}
for k, v in dict.iteritems() :
  map(social_properties, v)

  
net_dict = {}
  
for k, s in sim_dict.iteritems() :
  net_dict[k] = []
  for m in s['networks'] :
    social_properties(m['network'])
    net_dict[k].append(m['network'])
    

def comp_deg(graphs) :
  degs = {}
  for k, l in graphs :
    degs[k] = []
    for g in graphs[k] :
      for n in range(len(g)) :
        degs[k].append(g.degree().values()[i])
  
  
  
  
  
  
  
  
  