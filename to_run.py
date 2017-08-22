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

erdos_sim = gsd.full_sim_for_family('erdos_renyi', nx.watts_strogatz_graph, [1000, 10, 1], 1, 1, 1e4, 1e3)
gsa.full_analysis(erdos_sim)
gsd.save_sim_data(erdos_sim)

barabasi_sim = gsd.full_sim_for_family('barabasi_albert', nx.barabasi_albert_graph, [1000, 5], 1, 1, 1e4, 1e3)
gsa.full_analysis(barabasi_sim)
gsd.save_sim_data(barabasi_sim)

reds_sim = gsd.full_sim_for_family('reds', rc.reds_graph, [1000, 0.1, 0.146, 1.0], 1, 1, 1e4, 1e3)
gsa.full_analysis(reds_sim)
gsd.save_sim_data(reds_sim)


regular = gsd.load_sim_data('regular_params=1000-4-0_nets=5_sims=5.gsdata')
erdos = gsd.load_sim_data('erdos_renyi_params=1000-4-1_nets=5_sims=5.gsdata')
barabasi = gsd.load_sim_data('barabasi_albert_params=1000-2_nets=5_sims=5.gsdata')
r_scale_free = gsd.load_sim_data('random_scale_free_params=1000-2_nets=5_sims=5.gsdata')

sim_dict = {'Regular': regular, 'Erdos-Renyi': erdos, 'Barabasi-Albert': barabasi, 'Random scale-free': r_scale_free}

mult_coop_by_b_plot(sim_dict)
