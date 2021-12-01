import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from ai_economist.foundation.utils import load_episode_log
import plotting
import os

base = "../rllib/phase1/dense_logs/"
log_path = os.path.join(base, "logs_0000000024600000/env000.lz4")
dense_log = load_episode_log(log_path)
(fig0, fig1, fig2), incomes, endows, c_trades, all_builds = plotting.breakdown(dense_log)
fig = plotting.vis_world_range(dense_log, t0=0, tN=200, N=5)
fig.savefig('vis_world_range.png') 
fig0.savefig('breakdown0.png')
fig1.savefig('breakdown1.png')
fig2.savefig('breakdown2.png')



# from ai_economist import foundation

# def do_plot(env, ax, fig):
#     """Plots world state during episode sampling."""
#     plotting.plot_env_state(env, ax)
#     ax.set_aspect('equal')
#     display.display(fig)
#     display.clear_output(wait=True)

# def play_random_episode(env, plot_every=100, do_dense_logging=False):
#     """Plays an episode with randomly sampled actions.
    
#     Demonstrates gym-style API:
#         obs                  <-- env.reset(...)         # Reset
#         obs, rew, done, info <-- env.step(actions, ...) # Interaction loop
    
#     """
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))

#     # Reset
#     obs = env.reset(force_dense_logging=do_dense_logging)

#     # Interaction loop (w/ plotting)
#     for t in range(env.episode_length):
#         actions = sample_random_actions(env, obs)
#         obs, rew, done, info = env.step(actions)

#         if ((t+1) % plot_every) == 0:
#             do_plot(env, ax, fig)

#     if ((t+1) % plot_every) != 0:
#         do_plot(env, ax, fig)        

# env_config = {
#     # ===== SCENARIO CLASS =====
#     # Which Scenario class to use: the class's name in the Scenario Registry (foundation.scenarios).
#     # The environment object will be an instance of the Scenario class.
#     'scenario_name': 'layout_from_file/simple_wood_and_stone',
    
#     # ===== COMPONENTS =====
#     # Which components to use (specified as list of ("component_name", {component_kwargs}) tuples).
#     #   "component_name" refers to the Component class's name in the Component Registry (foundation.components)
#     #   {component_kwargs} is a dictionary of kwargs passed to the Component class
#     # The order in which components reset, step, and generate obs follows their listed order below.
#     'components': [
#         # (1) Building houses
#         ('Build', {'skill_dist': "pareto", 'payment_max_skill_multiplier': 3}),
#         # (2) Trading collectible resources
#         ('ContinuousDoubleAuction', {'max_num_orders': 5}),
#         # (3) Movement and resource collection
#         ('Gather', {}),
#     ],
    
#     # ===== SCENARIO CLASS ARGUMENTS =====
#     # (optional) kwargs that are added by the Scenario class (i.e. not defined in BaseEnvironment)
#     'env_layout_file': 'quadrant_25x25_20each_30clump.txt',
#     'starting_agent_coin': 10,
#     'fixed_four_skill_and_loc': True,
    
#     # ===== STANDARD ARGUMENTS ======
#     # kwargs that are used by every Scenario class (i.e. defined in BaseEnvironment)
#     'n_agents': 4,          # Number of non-planner agents (must be > 1)
#     'world_size': [25, 25], # [Height, Width] of the env world
#     'episode_length': 1000, # Number of timesteps per episode
    
#     # In multi-action-mode, the policy selects an action for each action subspace (defined in component code).
#     # Otherwise, the policy selects only 1 action.
#     'multi_action_mode_agents': False,
#     'multi_action_mode_planner': True,
    
#     # When flattening observations, concatenate scalar & vector observations before output.
#     # Otherwise, return observations with minimal processing.
#     'flatten_observations': False,
#     # When Flattening masks, concatenate each action subspace mask into a single array.
#     # Note: flatten_masks = True is required for masking action logits in the code below.
#     'flatten_masks': True,
# }

# env = foundation.make_env_instance(**env_config)
# # Play another episode. This time, tell the environment to do dense logging
# play_random_episode(env, plot_every=100, do_dense_logging=True)


# Grab the dense log from the env
# dense_log = env.previous_episode_dense_log
# dense_log = https://github.com/CornellDataScience/ai-economist/tree/training/gpu1/11_10_2021/tutorials/rllib/phase1/dense_logs/logs_0000000024600000
# fig = plotting.vis_world_range(dense_log, t0=0, tN=200, N=5)
# fig.savefig('vis_world_range.png')

# # Use the "breakdown" tool to visualize the world state, agent-wise quantities, movement, and trading events
# plotting.breakdown(dense_log); 

#CHANGES CHANGES CHANGES




