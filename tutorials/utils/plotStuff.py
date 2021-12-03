import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from ai_economist.foundation.utils import load_episode_log
import plotting
import os

base = "../rllib/phase2/dense_logs/"
log_path = os.path.join(base, "logs_0000000022446000/env002.lz4")
dense_log = load_episode_log(log_path)
(fig0, fig1, fig2), incomes, endows, c_trades, all_builds = plotting.breakdown(dense_log)
fig = plotting.vis_world_range(dense_log, t0=0, tN=200, N=5)
fig.savefig('vis_world_range_p2_5.png') 
fig0.savefig('p_2_breakdown15.png')
fig1.savefig('p_2_breakdown16.png')
fig2.savefig('p_2_breakdown17.png')
#plotting.get_all_taxes_from_all_logs(base)