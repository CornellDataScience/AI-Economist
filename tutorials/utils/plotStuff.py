import plotting

# Play another episode. This time, tell the environment to do dense logging
play_random_episode(env, plot_every=100, do_dense_logging=True)

# Grab the dense log from the env
dense_log = env.previous_episode_dense_log
fig = plotting.vis_world_range(dense_log, t0=0, tN=200, N=5)
fig.savefig('vis_world_range.png')

# # Use the "breakdown" tool to visualize the world state, agent-wise quantities, movement, and trading events
# plotting.breakdown(dense_log);