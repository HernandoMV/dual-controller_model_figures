# %%

# Hernando M Vergara June 2022

import numpy as np
import pandas as pd
from os import path
import matplotlib.pylab as plt

repo_path = path.abspath(path.dirname(__file__))

# output path
out_path = path.join(repo_path, 'output_plots')

# read the data
rpe_model_file = path.join(repo_path, 'data/rewards_and_weights/RPE_alone_reward_100trails_500timesteps.csv')
# print(rpe_model_file)
ape_model_file = path.join(repo_path, 'data/rewards_and_weights/APE_alone_reward_100trails_500timesteps.csv')
# print(rpe_model_file)

rpe_model = pd.read_csv(rpe_model_file, header=None).to_numpy()
ape_model = pd.read_csv(ape_model_file, header=None).to_numpy()


# %%
# create plots
plt.plot(np.mean(rpe_model, axis=0))
plt.plot(np.mean(ape_model, axis=0))