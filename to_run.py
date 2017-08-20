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