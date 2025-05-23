#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# required to use Tune with Python >= 3.7
import sys

version_info = sys.version_info
print(f"Python version is {version_info}")

if sys.version_info >= (3, 7):
    get_ipython().run_line_magic('pip', 'uninstall -y dataclasses')
else:
    get_ipython().run_line_magic('pip', 'install -U dataclasses')


# ## Import Libraries

# In[ ]:


# Python logging
import logging

logging.basicConfig()
logger = logging.getLogger('LOGGING_SCIMAI-Gym_V1')
logger.setLevel(logging.WARN)


# In[ ]:


# importing Gym
import gym
from gym.spaces import Box


# In[ ]:


# importing Ray
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.pg as pg
import ray.rllib.agents.ppo as ppo

from ray.rllib.utils import try_import_torch
torch = try_import_torch


# In[ ]:


# importing Ax
from ax import optimize

from ax.plot.contour import interact_contour
from ax.plot.contour import plot_contour_plotly
from ax.plot.trace import optimization_trace_single_method_plotly

from ax.utils.notebook.plotting import render
from ax.utils.notebook.plotting import init_notebook_plotting
init_notebook_plotting()


# In[ ]:


# importing necessary libraries
from datetime import datetime
from itertools import chain
from tabulate import tabulate
from timeit import default_timer
from IPython.display import display

import collections
import dataframe_image as dfi
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import shutil

plt.style.use('seaborn')
sns.set(context='notebook')


# In[ ]:


# setting seed for reproducibility
seed = 2021
np.random.seed(seed)
random.seed(seed)


# In[ ]:


# setting output views only in case of debug
if logger.level == 10:
    verbose = 3
    plt.ion()
else:
    verbose = 0
    plt.ioff()


# In[ ]:


# getting the number of CPUs
import multiprocessing

try:
    num_cpus = multiprocessing.cpu_count()
except Exception as e:
    print(f"{e.__class__} occurred!")
    num_cpus = 0

print(f"num cpus is {num_cpus}")


# In[ ]:


# getting the number of GPUs
import GPUtil as GPU

try:
    num_gpus = len(GPU.getGPUs())
except Exception as e:
    print(f"{e.__class__} occurred!")
    num_gpus = 0

print(f"num gpus is {num_gpus}")


# # Reinforcement Learning Classes

# ## State Class

# In[ ]:


class State:
    """
    We choose the state vector to include all current stock levels for each 
    warehouse and product type, plus the last demand values.
    """

    def __init__(self, product_types_num, distr_warehouses_num, T,
                 demand_history, t=0):
        self.product_types_num = product_types_num
        self.factory_stocks = np.zeros(
            (self.product_types_num,),
            dtype=np.int32)
        self.distr_warehouses_num = distr_warehouses_num
        self.distr_warehouses_stocks = np.zeros(
            (self.distr_warehouses_num, self.product_types_num),
            dtype=np.int32)
        self.T = T
        self.demand_history = demand_history
        self.t = t

        logger.debug(f"\n--- State --- __init__"
                     f"\nproduct_types_num is "
                     f"{self.product_types_num}"
                     f"\nfactory_stocks is "
                     f"{self.factory_stocks}"
                     f"\ndistr_warehouses_num is "
                     f"{self.distr_warehouses_num}"
                     f"\ndistr_warehouses_stocks is "
                     f"{self.distr_warehouses_stocks}"
                     f"\nT is "
                     f"{self.T}"
                     f"\ndemand_history is "
                     f"{self.demand_history}"
                     f"\nt is "
                     f"{self.t}")

    def to_array(self):
        logger.debug(f"\n--- State --- to_array"
                     f"\nnp.concatenate is "
                     f"""{np.concatenate((
                         self.factory_stocks,
                         self.distr_warehouses_stocks.flatten(),
                         np.hstack(list(chain(*chain(*self.demand_history)))),
                         [self.t]))}""")

        return np.concatenate((
            self.factory_stocks,
            self.distr_warehouses_stocks.flatten(),
            np.hstack(list(chain(*chain(*self.demand_history)))),
            [self.t]))

    def stock_levels(self):
        logger.debug(f"\n--- State --- stock_levels"
                     f"\nnp.concatenate is "
                     f"""{np.concatenate((
                         self.factory_stocks,
                         self.distr_warehouses_stocks.flatten()))}""")

        return np.concatenate((
            self.factory_stocks,
            self.distr_warehouses_stocks.flatten()))


# ## Action Class

# In[ ]:


class Action:
    """
    The action vector consists of production and shipping controls.
    """

    def __init__(self, product_types_num, distr_warehouses_num):
        self.production_level = np.zeros(
            (product_types_num,),
            dtype=np.int32)
        self.shipped_stocks = np.zeros(
            (distr_warehouses_num, product_types_num),
            dtype=np.int32)

        logger.debug(f"\n--- Action --- __init__"
                     f"\nproduction_level is "
                     f"{self.production_level}"
                     f"\nshipped_stocks is "
                     f"{self.shipped_stocks}")


# ## Supply Chain Environment Class

# In[ ]:


class SupplyChainEnvironment:
    """
    We designed a divergent two-echelon supply chain that includes a single 
    factory, multiple distribution warehouses, and multiple product types over 
    a fixed number of time steps. At each time step, the agent is asked to find 
    the number of products to be produced and preserved at the factory, as well 
    as the number of products to be shipped to different distribution 
    warehouses. To make the supply chain more realistic, we set capacity 
    constraints on warehouses (and consequently, on how many units to produce 
    at the factory), along with storage and transportation costs. 
    """

    def __init__(self):
        # number of product types (e.g., 2 product types)
        self.product_types_num = 2
        # number of distribution warehouses (e.g., 2 distribution warehouses)
        self.distr_warehouses_num = 2
        # final time step (e.g., an episode takes 25 time steps)
        self.T = 25

        # maximum demand value, units (e.g., [3, 6])
        self.d_max = np.array(
            [3, 6],
            np.int32)
        # maximum demand variation according to a uniform distribution,
        # units (e.g., [2, 1])
        self.d_var = np.array(
            [2, 1],
            np.int32)

        # sale prices, per unit (e.g., [20, 10])
        self.sale_prices = np.array(
            [20, 10],
            np.int32)
        # production costs, per unit (e.g., [2, 1])
        self.production_costs = np.array(
            [2, 1],
            np.int32)

        # storage capacities for each product type at each warehouse,
        # units (e.g., [[3, 4], [6, 8], [9, 12]])
        self.storage_capacities = np.array(
            [[3, 4], [6, 8], [9, 12]],
            np.int32)

        # storage costs of each product type at each warehouse,
        # per unit (e.g., [[6, 3], [4, 2], [2, 1]])
        self.storage_costs = np.array(
            [[6, 3], [4, 2], [2, 1]],
            np.float32)
        # transportation costs of each product type for each distribution
        # warehouse, per unit (e.g., [[.1, .3], [.2, .6]])
        self.transportation_costs = np.array(
            [[.1, .3], [.2, .6]],
            np.float32)

        # penalty costs, per unit (e.g., [10, 5])
        self.penalty_costs = .5*self.sale_prices

        print(f"\n--- SupplyChainEnvironment --- __init__"
              f"\nproduct_types_num is "
              f"{self.product_types_num}"
              f"\ndistr_warehouses_num is "
              f"{self.distr_warehouses_num}"
              f"\nT is "
              f"{self.T}"
              f"\nd_max is "
              f"{self.d_max}"
              f"\nd_var is "
              f"{self.d_var}"
              f"\nsale_prices is "
              f"{self.sale_prices}"
              f"\nproduction_costs is "
              f"{self.production_costs}"
              f"\nstorage_capacities is "
              f"{self.storage_capacities}"
              f"\nstorage_costs is "
              f"{self.storage_costs}"
              f"\ntransportation_costs is "
              f"{self.transportation_costs}"
              f"\npenalty_costs is "
              f"{self.penalty_costs}")

        self.reset()

    def reset(self, demand_history_len=5):
        # (five) demand values observed
        self.demand_history = collections.deque(maxlen=demand_history_len)

        logger.debug(f"\n--- SupplyChainEnvironment --- reset"
                     f"\ndemand_history is "
                     f"{self.demand_history}")

        for d in range(demand_history_len):
            self.demand_history.append(np.zeros(
                (self.distr_warehouses_num, self.product_types_num),
                dtype=np.int32))
        self.t = 0

        logger.debug(f"\ndemand_history is "
                     f"{self.demand_history}"
                     f"\nt is "
                     f"{self.t}")

    def demand(self, j, i, t):
        # we simulate a seasonal behavior by representing the demand as a
        # co-sinusoidal function with a stochastic component (a random variable
        # assumed to be distributed according to a uniform distribution),
        # in order to evaluate the agent
        demand = np.round(
            self.d_max[i-1]/2 +
            self.d_max[i-1]/2*np.cos(4*np.pi*(2*j*i+t)/self.T) +
            np.random.randint(0, self.d_var[i-1]+1))

        logger.debug(f"\n--- SupplyChainEnvironment --- demand"
                     f"\nj is "
                     f"{j}"
                     f"\ni is "
                     f"{i}"
                     f"\nt is "
                     f"{t}"
                     f"\ndemand is "
                     f"{demand}")

        return demand

    def initial_state(self):
        logger.debug(f"\n--- SupplyChainEnvironment --- initial_state"
                     f"\nState is "
                     f"""{State(
                         self.product_types_num, self.distr_warehouses_num, 
                         self.T, list(self.demand_history))}""")

        return State(self.product_types_num, self.distr_warehouses_num,
                     self.T, list(self.demand_history))

    def step(self, state, action):
        demands = np.fromfunction(
            lambda j, i: self.demand(j+1, i+1, self.t),
            (self.distr_warehouses_num, self.product_types_num),
            dtype=np.int32)

        logger.debug(f"\n--- SupplyChainEnvironment --- step"
                     f"\nstate is "
                     f"{state}"
                     f"\nstate.factory_stocks is "
                     f"{state.factory_stocks}"
                     f"\nstate.distr_warehouses_stocks is "
                     f"{state.distr_warehouses_stocks}"
                     f"\naction is "
                     f"{action}"
                     f"\naction.production_level is "
                     f"{action.production_level}"
                     f"\naction.shipped_stocks is "
                     f"{action.shipped_stocks}"
                     f"\ndemands is "
                     f"{demands}")

        # next state
        next_state = State(self.product_types_num, self.distr_warehouses_num,
                           self.T, list(self.demand_history))

        next_state.factory_stocks = np.minimum(
            np.subtract(np.add(state.factory_stocks,
                               action.production_level),
                        np.sum(action.shipped_stocks, axis=0)
                        ),
            self.storage_capacities[0]
        )

        for j in range(self.distr_warehouses_num):
            next_state.distr_warehouses_stocks[j] = np.minimum(
                np.subtract(np.add(state.distr_warehouses_stocks[j],
                                   action.shipped_stocks[j]),
                            demands[j]
                            ),
                self.storage_capacities[j+1]
            )

        logger.debug(f"\n-- SupplyChainEnvironment -- next state"
                     f"\nnext_state is "
                     f"{next_state}"
                     f"\nnext_state.factory_stocks is "
                     f"{next_state.factory_stocks}"
                     f"\nnext_state.distr_warehouses_stocks is "
                     f"{next_state.distr_warehouses_stocks}"
                     f"\nnext_state.demand_history is "
                     f"{next_state.demand_history}"
                     f"\nnext_state.t is "
                     f"{next_state.t}")

        # revenues
        total_revenues = np.dot(self.sale_prices,
                                np.sum(demands, axis=0))
        # production costs
        total_production_costs = np.dot(self.production_costs,
                                        action.production_level)
        # transportation costs
        total_transportation_costs = np.dot(
            self.transportation_costs.flatten(),
            action.shipped_stocks.flatten())
        # storage costs
        total_storage_costs = np.dot(
            self.storage_costs.flatten(),
            np.maximum(next_state.stock_levels(),
                       np.zeros(
                           ((self.distr_warehouses_num+1) *
                            self.product_types_num),
                           dtype=np.int32)
                       )
        )
        # penalty costs (minus sign because stock levels would be already
        # negative in case of unfulfilled demand)
        total_penalty_costs = -np.dot(
            self.penalty_costs,
            np.add(
                np.sum(
                    np.minimum(next_state.distr_warehouses_stocks,
                               np.zeros(
                                   (self.distr_warehouses_num,
                                    self.product_types_num),
                                   dtype=np.int32)
                               ),
                    axis=0),
                np.minimum(next_state.factory_stocks,
                           np.zeros(
                               (self.product_types_num,),
                               dtype=np.int32)
                           )
            )
        )
        # reward function
        reward = total_revenues - total_production_costs - \
            total_transportation_costs - total_storage_costs - \
            total_penalty_costs

        logger.debug(f"\n-- SupplyChainEnvironment -- reward"
                     f"\ntotal_revenues is "
                     f"{total_revenues}"
                     f"\ntotal_production_costs is "
                     f"{total_production_costs}"
                     f"\ntotal_transportation_costs is "
                     f"{total_transportation_costs}"
                     f"\ntotal_storage_costs is "
                     f"{total_storage_costs}"
                     f"\ntotal_penalty_costs is "
                     f"{total_penalty_costs}"
                     f"\nreward is "
                     f"{reward}")

        # the actual demand for the current time step will not be known until
        # the next time step. This implementation choice ensures that the agent
        # may benefit from learning the demand pattern so as to integrate a
        # sort of demand forecasting directly into the policy
        self.demand_history.append(demands)
        # actual time step value is not observed (for now)
        self.t += 1

        logger.debug(f"\ndemand_history is "
                     f"{self.demand_history}"
                     f"\nt is "
                     f"{self.t}")

        logger.debug(f"\n-- SupplyChainEnvironment -- return"
                     f"\nnext_state is "
                     f"{next_state}, "
                     f"\nreward is "
                     f"{reward}, "
                     f"\ndone is "
                     f"{self.t == self.T-1}")

        return next_state, reward, self.t == self.T-1


# ## Supply Chain Gym Wrapper

# In[ ]:


class SupplyChain(gym.Env):
    """
    Gym environment wrapper.
    """

    def __init__(self, config):
        self.reset()

        # low values for action space (no negative actions)
        low_act = np.zeros(
            ((self.supply_chain.distr_warehouses_num+1) *
             self.supply_chain.product_types_num),
            dtype=np.int32)
        # high values for action space
        high_act = np.zeros(
            ((self.supply_chain.distr_warehouses_num+1) *
             self.supply_chain.product_types_num),
            dtype=np.int32)
        # high values for action space (factory)
        high_act[
            :self.supply_chain.product_types_num
        ] = np.sum(self.supply_chain.storage_capacities, axis=0)
        # high values for action space (distribution warehouses, according to
        # storage capacities)
        high_act[
            self.supply_chain.product_types_num:
        ] = (self.supply_chain.storage_capacities.flatten()[
            self.supply_chain.product_types_num:])
        # action space
        self.action_space = Box(low=low_act,
                                high=high_act,
                                dtype=np.int32)

        # low values for observation space
        low_obs = np.zeros(
            (len(self.supply_chain.initial_state().to_array()),),
            dtype=np.int32)
        # low values for observation space (factory, worst case scenario in
        # case of non-production and maximum demand)
        low_obs[
            :self.supply_chain.product_types_num
        ] = -np.sum(self.supply_chain.storage_capacities[1:], axis=0) * \
            self.supply_chain.T
        # low values for observation space (distribution warehouses, worst case
        # scenario in case of non-shipments and maximum demand)
        low_obs[
            self.supply_chain.product_types_num:
                (self.supply_chain.distr_warehouses_num+1) *
            self.supply_chain.product_types_num
        ] = np.array([
            -(self.supply_chain.d_max+self.supply_chain.d_var) *
            self.supply_chain.T
        ] * self.supply_chain.distr_warehouses_num).flatten()
        # high values for observation space
        high_obs = np.zeros(
            (len(self.supply_chain.initial_state().to_array()),),
            dtype=np.int32)
        # high values for observation space (factory and distribution
        # warehouses, according to storage capacities)
        high_obs[
            :(self.supply_chain.distr_warehouses_num+1) *
            self.supply_chain.product_types_num
        ] = self.supply_chain.storage_capacities.flatten()
        # high values for observation space (demand, according to the maximum
        # demand value)
        high_obs[
            (self.supply_chain.distr_warehouses_num+1) *
            self.supply_chain.product_types_num:
            len(high_obs)-1
        ] = np.array([
            self.supply_chain.d_max+self.supply_chain.d_var] *
            len(list(chain(*self.supply_chain.demand_history)))).flatten()
        # high values for observation space (episode, according to the final
        # time step)
        high_obs[len(high_obs)-1] = self.supply_chain.T
        # observation space
        self.observation_space = Box(low=low_obs,
                                     high=high_obs,
                                     dtype=np.int32)

        logger.debug(f"\n--- SupplyChain --- __init__"
                     f"\nlow_act is "
                     f"{low_act}"
                     f"\nhigh_act is "
                     f"{high_act}"
                     f"\naction_space is "
                     f"{self.action_space}"
                     f"\nlow_obs is "
                     f"{low_obs}"
                     f"\nhigh_obs is "
                     f"{high_obs}"
                     f"\nobservation_space is "
                     f"{self.observation_space}")

    def reset(self):
        self.supply_chain = SupplyChainEnvironment()
        self.state = self.supply_chain.initial_state()

        logger.debug(f"\n--- SupplyChain --- reset"
                     f"\nsupply_chain is "
                     f"{self.supply_chain}"
                     f"\nstate is "
                     f"{self.state}"
                     f"\nstate.to_array is "
                     f"{self.state.to_array()}")

        return self.state.to_array()

    def step(self, action):
        # casting to integer actions (units of product to produce and ship)
        action_obj = Action(
            self.supply_chain.product_types_num,
            self.supply_chain.distr_warehouses_num)
        action_obj.production_level = action[
            :self.supply_chain.product_types_num].astype(np.int32)
        action_obj.shipped_stocks = action[
            self.supply_chain.product_types_num:
        ].reshape((self.supply_chain.distr_warehouses_num,
                   self.supply_chain.product_types_num)).astype(np.int32)

        logger.debug(f"\n--- SupplyChain --- step"
                     f"\naction is "
                     f"{action}"
                     f"\naction_obj is "
                     f"{action_obj}"
                     f"\naction_obj.production_level is "
                     f"{action_obj.production_level}"
                     f"\naction_obj.shipped_stocks is "
                     f"{action_obj.shipped_stocks}")

        self.state, reward, done = self.supply_chain.step(
            self.state, action_obj)

        logger.debug(f"\n-- SupplyChain -- return"
                     f"\nstate.to_array is "
                     f"{self.state.to_array()}"
                     f"\nreward is "
                     f"{reward}"
                     f"\ndone is "
                     f"{done}")

        return self.state.to_array(), reward, done, {}


# # Global Parameters

# In[ ]:


# number of episodes for the simulations
num_episodes = 200


# In[ ]:


# name of the experiment (e.g., '2P2W' stands for two product types and two
# distribution warehouses)
now = datetime.now()
now_str = now.strftime('%Y-%m-%d_%H-%M-%S')
local_dir = f"2P2W_{now_str}"
# dir to save plots
plots_dir = 'plots'
# creating necessary dirs
if not os.path.exists(f"{local_dir}"):
    os.makedirs(f"{local_dir}")
if not os.path.exists(f"{local_dir+'/'+plots_dir}"):
    os.makedirs(f"{local_dir+'/'+plots_dir}")


# # Supply Chain Environment Initialization

# ## Visualize Demand Methods

# In[ ]:


def visualize_demand(num_episodes=1, local_dir=local_dir, plots_dir=plots_dir):
    """
    Visualize demand behavior for each distribution warehouses and for each 
    product type.
    """
    if env.distr_warehouses_num <= 3 and env.product_types_num <= 2:
        demands = []
        for n in range(num_episodes):
            # generating demands
            demands.append(np.fromfunction(
                lambda j, i, t: env.demand(j+1, i+1, t),
                (env.distr_warehouses_num, env.product_types_num, env.T),
                dtype=np.int32))

        # mean of demands
        demands_mean = np.array([np.mean(demand, axis=0)
                                for demand in zip(*demands)])
        # std of demands
        demands_std = np.array([np.std(demand, axis=0)
                               for demand in zip(*demands)])

        logger.debug(f"\n-- visualize_demand --"
                     f"\ndemands is "
                     f"{demands}"
                     f"\ndemands_mean is "
                     f"{demands_mean}"
                     f"\ndemands_std is "
                     f"{demands_std}")

        plt.figure(figsize=(15, 5))
        plt.xlabel('Time Steps')
        plt.ylabel('Demand Value')

        plt.xticks(np.arange(min(range(env.T)),
                             max(range(env.T))+1))
        plt.tick_params(axis='x', which='both',
                        top=True, bottom=True,
                        labelbottom=True)
        plt.ticklabel_format(axis='y', style='plain',
                             useOffset=False)
        plt.tight_layout()

        # same color for the same distribution warehouse, but different line
        # style according to the different product type
        color = [['b', 'b'], ['g', 'g'], ['r', 'r']]
        line_style = [['b-', 'b--'], ['g-', 'g--'], ['r-', 'r--']]

        for j in range(env.distr_warehouses_num):
            for i in range(env.product_types_num):
                plt.plot(range(env.T),
                         demands_mean[j][i].T,
                         line_style[j][i])
                plt.fill_between(range(env.T),
                                 (demands_mean[j][i] -
                                  demands_std[j][i]).flatten(),
                                 (demands_mean[j][i] +
                                  demands_std[j][i]).flatten(),
                                 color=color[j][i], alpha=.2)

        # plotting legend
        plt.legend([f"WH {j+1}, Prod {i+1}"
                    for j in range(env.distr_warehouses_num)
                    for i in range(env.product_types_num)])

        # saving plot
        plt.savefig(f"{local_dir}/{plots_dir}"
                    f"/demand.pdf",
                    format='pdf', bbox_inches='tight')


def save_env_settings(env, local_dir=local_dir, plots_dir=plots_dir):
    """
    Save the Supply Chain Environment settings.
    """
    f = open(f"{local_dir}/{plots_dir}"
             f"/env_settings.txt",
             'w', encoding='utf-8')
    f.write(f"--- SupplyChainEnvironment ---"
            f"\nproduct_types_num is "
            f"{env.product_types_num}"
            f"\ndistr_warehouses_num is "
            f"{env.distr_warehouses_num}"
            f"\nT is "
            f"{env.T}"
            f"\nd_max is "
            f"{env.d_max}"
            f"\nd_var is "
            f"{env.d_var}"
            f"\nsale_prices is "
            f"{env.sale_prices}"
            f"\nproduction_costs is "
            f"{env.production_costs}"
            f"\nstorage_capacities is "
            f"{env.storage_capacities}"
            f"\nstorage_costs is "
            f"{env.storage_costs}"
            f"\ntransportation_costs is "
            f"{env.transportation_costs}"
            f"\npenalty_costs is "
            f"{env.penalty_costs}")
    f.close()


# ## Initialization

# In[ ]:


# supply chain env
env = SupplyChainEnvironment()


# In[ ]:


# saving env settings
save_env_settings(env)


# In[ ]:


visualize_demand(num_episodes)


# # Methods

# ## Simulator Methods

# In[ ]:


def simulate_oracle(env, num_episodes):
    """
    Oracle simulator.
    """
    rewards_trace = []
    for n in range(num_episodes):
        # generating demands (oracle knows the episodes' demands a priori)
        demands = []
        demands = np.fromfunction(
            lambda j, i, t: env.demand(j+1, i+1, t),
            (env.distr_warehouses_num, env.product_types_num, env.T),
            dtype=np.int32)

        # calculating demands for each product type
        demands_product_types = np.sum(np.sum(demands,
                                              axis=0),
                                       axis=1)
        # calculating demands of each product type for each warehouse
        demands_distr_warehouses = np.sum(demands,
                                          axis=2)

        # revenues
        total_revenues = np.dot(env.sale_prices,
                                demands_product_types)
        # production costs
        total_production_costs = np.dot(env.production_costs,
                                        demands_product_types)
        # transportation costs
        total_transportation_costs = np.dot(
            env.transportation_costs.flatten(),
            demands_distr_warehouses.flatten())
        # reward
        reward = total_revenues - total_production_costs - \
            total_transportation_costs
        # append reward for each episode
        rewards_trace.append(reward)

        logger.debug(f"\n-- simulate_oracle --"
                     f"\ndemands is "
                     f"{demands}"
                     f"\ndemands_product_types is "
                     f"{demands_product_types}"
                     f"\ndemands_distr_warehouses is "
                     f"{demands_distr_warehouses}"
                     f"\ntotal_revenues is "
                     f"{total_revenues}"
                     f"\ntotal_production_costs is "
                     f"{total_production_costs}"
                     f"\ntotal_transportation_costs is "
                     f"{total_transportation_costs}"
                     f"\nreward is "
                     f"{reward}")

    print(f"reward: mean "
          f"{np.mean(rewards_trace)}, "
          f"std "
          f"{np.std(rewards_trace)}, "
          f"max "
          f"{np.max(rewards_trace)}, "
          f"min "
          f"{np.min(rewards_trace)}")

    return np.array(rewards_trace)


def simulate_episode(env, policy):
    """
    Single episode simulator.
    """
    env.reset()
    state = env.initial_state()
    transitions = []

    logger.debug(f"\n-- simulate_episode --"
                 f"\nstate is "
                 f"{state}"
                 f"\ntransitions is "
                 f"{transitions}")

    if not isinstance(policy, SQPolicy):
        for t in range(env.T):
            action = policy.compute_single_action(state.to_array(),
                                                  normalize_actions=True,
                                                  explore=False)[0].astype(
                np.int32)
            action_obj = Action(env.product_types_num,
                                env.distr_warehouses_num)
            action_obj.production_level = action[:env.product_types_num]
            action_obj.shipped_stocks = action[
                env.product_types_num:
            ].reshape((env.distr_warehouses_num, env.product_types_num))

            state, reward, _ = env.step(state, action_obj)
            transitions.append(np.array(
                [state, action_obj, reward],
                dtype=object))

            logger.debug(f"\naction is "
                         f"{action}"
                         f"\naction_obj is "
                         f"{action_obj}"
                         f"\naction_obj.production_level is "
                         f"{action_obj.production_level}"
                         f"\naction_obj.shipped_stocks is "
                         f"{action_obj.shipped_stocks}"
                         f"\nstate is "
                         f"{state}"
                         f"\nstate.factory_stocks is "
                         f"{state.factory_stocks}"
                         f"\nstate.distr_warehouses_stocks is "
                         f"{state.distr_warehouses_stocks}"
                         f"\nstate.demand_history is "
                         f"{state.demand_history}"
                         f"\nt is "
                         f"{t}"
                         f"\nreward is "
                         f"{reward}")
    else:
        for t in range(env.T):
            action = policy.select_action(state)
            state, reward, _ = env.step(state, action)
            transitions.append(np.array(
                [state, action, reward],
                dtype=object))

            logger.debug(f"\naction is "
                         f"{action}"
                         f"\naction.production_level is "
                         f"{action.production_level}"
                         f"\naction.shipped_stocks is "
                         f"{action.shipped_stocks}"
                         f"\nstate is "
                         f"{state}"
                         f"\nstate.factory_stocks is "
                         f"{state.factory_stocks}"
                         f"\nstate.distr_warehouses_stocks is "
                         f"{state.distr_warehouses_stocks}"
                         f"\nstate.demand_history is "
                         f"{state.demand_history}"
                         f"\nt is "
                         f"{t}"
                         f"\nreward is "
                         f"{reward}")

    logger.debug(f"\ntransitions [state, action, reward] is "
                 f"{transitions}")

    return transitions


def simulate(env, policy, num_episodes=1):
    """
    Simulator.
    """
    returns_trace = []

    if not isinstance(policy, SQPolicy):
        # initializing Ray
        ray.shutdown()
        ray.init(log_to_driver=False)

    for episode in range(num_episodes):
        returns_trace.append(np.array(
            simulate_episode(env, policy)))

    if not isinstance(policy, SQPolicy):
        # stopping Ray
        ray.shutdown()

    logger.debug(f"\n-- simulate --"
                 f"\nreturns_trace is "
                 f"{returns_trace}")

    return returns_trace


# ## Visualize Transitions Methods

# In[ ]:


def prepare_metric_plot(ylabel, n,
                        plots_n=10 if env.product_types_num == 1 else 26):
    """
    Auxiliary function.
    """
    plt.subplot(plots_n, 1, n)
    plt.ylabel(ylabel, fontsize=10)

    plt.xticks(np.arange(min(range(env.T)),
                         max(range(env.T))+1))
    plt.tick_params(axis='x', which='both',
                    top=True, bottom=True,
                    labelbottom=False)


def visualize_transitions(returns_trace, algorithm,
                          local_dir=local_dir, plots_dir=plots_dir):
    """
    Visualize transitions (stock levels, production and shipping controls,
    reward) along the episodes.
    """
    if env.distr_warehouses_num <= 3 and env.product_types_num <= 2:
        transitions = np.array(
            [(return_trace)
             for return_trace in zip(*returns_trace)])
        states_trace, actions_trace, rewards_trace = (transitions.T[0],
                                                      transitions.T[1],
                                                      transitions.T[2])

        logger.debug(f"\n-- visualize_transitions --"
                     f"\nstates_trace is "
                     f"{states_trace}"
                     f"\nactions_trace is "
                     f"{actions_trace}"
                     f"\nrewards_trace is "
                     f"{rewards_trace}")

        plt.figure(figsize=(10, 30))

        # states transitions
        states = np.array(
            [(state_trace)
             for state_trace in zip(*states_trace)])

        logger.debug(f"\nstates is "
                     f"{states}")

        # factory stocks
        prepare_metric_plot('Stocks,\nFactory',
                            1)
        tmp_mean = []
        for t in range(len(states)):
            tmp_mean.append(
                np.mean(
                    [np.sum(state.factory_stocks)
                     for state in states[t]], axis=0))
        tmp_std = []
        for t in range(len(states)):
            tmp_std.append(
                np.std(
                    [np.sum(state.factory_stocks)
                     for state in states[t]], axis=0))
        plt.plot(range(env.T),
                 tmp_mean,
                 color='purple', alpha=.5)
        plt.fill_between(range(env.T),
                         list(np.array(tmp_mean) -
                              np.array(tmp_std)),
                         list(np.array(tmp_mean) +
                              np.array(tmp_std)),
                         color='purple', alpha=.2)

        logger.debug(f"\nfactory_stocks (mean) is "
                     f"{tmp_mean}"
                     f"\nfactory_stocks (std) is "
                     f"{tmp_std}")

        if env.product_types_num >= 2:
            for i in range(env.product_types_num):
                prepare_metric_plot(f"Stocks,\nFactory,\nProd {i+1}",
                                    2+i)
                tmp_mean = []
                for t in range(len(states)):
                    tmp_mean.append(
                        np.mean(
                            [np.sum(state.factory_stocks[i])
                             for state in states[t]], axis=0))
                tmp_std = []
                for t in range(len(states)):
                    tmp_std.append(
                        np.std(
                            [np.sum(state.factory_stocks[i])
                             for state in states[t]], axis=0))
                plt.plot(range(env.T),
                         tmp_mean,
                         '--',
                         color='purple', alpha=.5)
                plt.fill_between(range(env.T),
                                 list(np.array(tmp_mean) -
                                      np.array(tmp_std)),
                                 list(np.array(tmp_mean) +
                                      np.array(tmp_std)),
                                 color='purple', alpha=.2)

        logger.debug(f"\nfactory_stocks (mean for product) is "
                     f"{tmp_mean}"
                     f"\nfactory_stocks (std for product) is "
                     f"{tmp_std}")

        # distribution warehouses stocks
        for j in range(env.distr_warehouses_num):
            prepare_metric_plot(f"Stocks,\nWH {j+1}",
                                2*env.product_types_num+j)
            tmp_mean = []
            for t in range(len(states)):
                tmp_mean.append(
                    np.mean(
                        [np.sum(state.distr_warehouses_stocks[j])
                         for state in states[t]], axis=0))
            tmp_std = []
            for t in range(len(states)):
                tmp_std.append(
                    np.std(
                        [np.sum(state.distr_warehouses_stocks[j])
                         for state in states[t]], axis=0))
            plt.plot(range(env.T),
                     tmp_mean,
                     color='purple', alpha=.5)
            plt.fill_between(range(env.T),
                             list(np.array(tmp_mean) -
                                  np.array(tmp_std)),
                             list(np.array(tmp_mean) +
                                  np.array(tmp_std)),
                             color='purple', alpha=.2)

        logger.debug(f"\ndistr_warehouses_stocks (mean) is "
                     f"{tmp_mean}"
                     f"\ndistr_warehouses_stocks (std) is "
                     f"{tmp_std}")

        if env.product_types_num >= 2:
            cont = 0
            for j in range(env.distr_warehouses_num):
                for i in range(env.product_types_num):
                    prepare_metric_plot(f"Stocks,\nWH {j+1},\nProd {i+1}",
                                        4+env.distr_warehouses_num+cont)
                    tmp_mean = []
                    for t in range(len(states)):
                        tmp_mean.append(
                            np.mean(
                                [np.sum(state.distr_warehouses_stocks[j][i])
                                 for state in states[t]], axis=0))
                    tmp_std = []
                    for t in range(len(states)):
                        tmp_std.append(
                            np.std(
                                [np.sum(state.distr_warehouses_stocks[j][i])
                                 for state in states[t]], axis=0))
                    plt.plot(range(env.T),
                             tmp_mean,
                             '--',
                             color='purple', alpha=.5)
                    plt.fill_between(range(env.T),
                                     list(np.array(tmp_mean) -
                                          np.array(tmp_std)),
                                     list(np.array(tmp_mean) +
                                          np.array(tmp_std)),
                                     color='purple', alpha=.2)
                    cont += 1

        logger.debug(f"\ndistr_warehouses_stocks (mean for product) is "
                     f"{tmp_mean}"
                     f"\ndistr_warehouses_stocks (std for product) is "
                     f"{tmp_std}")

        # actions (transitions)
        actions = np.array(
            [(action_trace)
             for action_trace in zip(*actions_trace)])

        logger.debug(f"\nactions is "
                     f"{actions}")

        # production level
        prepare_metric_plot('Production,\nFactory',
                            2+env.distr_warehouses_num if
                            env.product_types_num == 1
                            else
                            4+env.distr_warehouses_num +
                            env.distr_warehouses_num *
                            env.product_types_num)
        tmp_mean = []
        for t in range(len(actions)):
            tmp_mean.append(
                np.mean(
                    [np.sum(action.production_level)
                     for action in actions[t]], axis=0))
        tmp_std = []
        for t in range(len(actions)):
            tmp_std.append(
                np.std(
                    [np.sum(action.production_level)
                     for action in actions[t]], axis=0))
        plt.plot(range(env.T),
                 tmp_mean,
                 color='blue', alpha=.5)
        plt.fill_between(range(env.T),
                         list(np.array(tmp_mean) -
                              np.array(tmp_std)),
                         list(np.array(tmp_mean) +
                              np.array(tmp_std)),
                         color='blue', alpha=.2)

        logger.debug(f"\nproduction_level (mean) is "
                     f"{tmp_mean}"
                     f"\nproduction_level (std) is "
                     f"{tmp_std}")

        if env.product_types_num >= 2:
            for i in range(env.product_types_num):
                prepare_metric_plot(f"Production,\nFactory,\nProd {i+1}",
                                    5+env.distr_warehouses_num +
                                    env.distr_warehouses_num *
                                    env.product_types_num+i)
                tmp_mean = []
                for t in range(len(actions)):
                    tmp_mean.append(
                        np.mean(
                            [np.sum(action.production_level[i])
                             for action in actions[t]], axis=0))
                tmp_std = []
                for t in range(len(actions)):
                    tmp_std.append(
                        np.std(
                            [np.sum(action.production_level[i])
                             for action in actions[t]], axis=0))
                plt.plot(range(env.T),
                         tmp_mean,
                         '--',
                         color='blue', alpha=.5)
                plt.fill_between(range(env.T),
                                 list(np.array(tmp_mean) -
                                      np.array(tmp_std)),
                                 list(np.array(tmp_mean) +
                                      np.array(tmp_std)),
                                 color='blue', alpha=.2)

        logger.debug(f"\nproduction_level (mean for product) is "
                     f"{tmp_mean}"
                     f"\nproduction_level (std for product) is "
                     f"{tmp_std}")

        # shipped stocks
        for j in range(env.distr_warehouses_num):
            prepare_metric_plot(f"Shipments,\nWH {j+1}",
                                3+env.distr_warehouses_num+j
                                if env.product_types_num == 1
                                else
                                7+env.distr_warehouses_num +
                                env.distr_warehouses_num *
                                env.product_types_num+j)
            tmp_mean = []
            for t in range(len(actions)):
                tmp_mean.append(
                    np.mean(
                        [np.sum(action.shipped_stocks[j])
                         for action in actions[t]], axis=0))
            tmp_std = []
            for t in range(len(actions)):
                tmp_std.append(
                    np.std(
                        [np.sum(action.shipped_stocks[j])
                         for action in actions[t]], axis=0))
            plt.plot(range(env.T),
                     tmp_mean,
                     color='blue', alpha=.5)
            plt.fill_between(range(env.T),
                             list(np.array(tmp_mean) -
                                  np.array(tmp_std)),
                             list(np.array(tmp_mean) +
                                  np.array(tmp_std)),
                             color='blue', alpha=.2)

        logger.debug(f"\nshipped_stocks (mean) is "
                     f"{tmp_mean}"
                     f"\nshipped_stocks (std) is "
                     f"{tmp_std}")

        if env.product_types_num >= 2:
            cont = 0
            for j in range(env.distr_warehouses_num):
                for i in range(env.product_types_num):
                    prepare_metric_plot(f"Shipments,\nWH {j+1},\nProd {i+1}",
                                        7+(2*env.distr_warehouses_num) +
                                        env.distr_warehouses_num *
                                        env.product_types_num+cont)
                    tmp_mean = []
                    for t in range(len(actions)):
                        tmp_mean.append(
                            np.mean(
                                [np.sum(action.shipped_stocks[j][i])
                                 for action in actions[t]], axis=0))
                    tmp_std = []
                    for t in range(len(actions)):
                        tmp_std.append(
                            np.std(
                                [np.sum(action.shipped_stocks[j][i])
                                 for action in actions[t]], axis=0))
                    plt.plot(range(env.T),
                             tmp_mean,
                             '--',
                             color='blue', alpha=.5)
                    plt.fill_between(range(env.T),
                                     list(np.array(tmp_mean) -
                                          np.array(tmp_std)),
                                     list(np.array(tmp_mean) +
                                          np.array(tmp_std)),
                                     color='blue', alpha=.2)
                    cont += 1

        logger.debug(f"\nshipped_stocks (mean for product) is "
                     f"{tmp_mean}"
                     f"\nshipped_stocks (std for product) is "
                     f"{tmp_std}")

        # profit
        prepare_metric_plot('Profit',
                            3+(2*env.distr_warehouses_num)
                            if env.product_types_num == 1
                            else
                            7+(2*env.distr_warehouses_num) +
                            2*env.distr_warehouses_num*env.product_types_num)
        reward_mean = np.array(
            np.mean(rewards_trace, axis=0),
            dtype=np.int32)
        reward_std = np.array(
            np.std(rewards_trace.astype(np.int32), axis=0),
            dtype=np.int32)
        plt.plot(range(env.T),
                 reward_mean,
                 linewidth=2,
                 color='red', alpha=.5)
        plt.fill_between(range(env.T),
                         reward_mean -
                         reward_std,
                         reward_mean +
                         reward_std,
                         color='red', alpha=.2)

        logger.debug(f"\nprofit (mean) is "
                     f"{reward_mean}"
                     f"\nprofit (std) is "
                     f"{reward_std}")

        # cumulative profit
        prepare_metric_plot('Cum\nProfit',
                            4+(2*env.distr_warehouses_num)
                            if env.product_types_num == 1
                            else
                            8+(2*env.distr_warehouses_num) +
                            2*env.distr_warehouses_num*env.product_types_num)
        cum_reward = np.array(
            [np.cumsum(reward_trace)
             for reward_trace in rewards_trace])
        cum_reward_mean = np.array(
            np.mean(cum_reward, axis=0),
            dtype=np.int32)
        cum_reward_std = np.array(
            np.std(cum_reward.astype(np.int32), axis=0),
            dtype=np.int32)
        plt.plot(range(env.T),
                 cum_reward_mean,
                 linewidth=2,
                 color='red', alpha=.5)
        plt.fill_between(range(env.T),
                         cum_reward_mean -
                         cum_reward_std,
                         cum_reward_mean +
                         cum_reward_std,
                         color='red', alpha=.2)

        logger.debug(f"\ncumulative profit (mean) is "
                     f"{cum_reward_mean}"
                     f"\ncumulative profit (std) is "
                     f"{cum_reward_std}")

        plt.xlabel('Time Steps', labelpad=10)
        plt.ticklabel_format(axis='y', style='plain',
                             useOffset=False)
        plt.tight_layout()

        # creating necessary subdir and saving plot
        if not os.path.exists(f"{local_dir}/{plots_dir}/{algorithm}"):
            os.makedirs(f"{local_dir}/{plots_dir}/{algorithm}")
        plt.savefig(f"{local_dir}/{plots_dir}/{algorithm}"
                    f"/transitions_{algorithm}.pdf",
                    format='pdf', bbox_inches='tight')


# ## Visualize Cumulative Profit Methods

# In[ ]:


def calculate_cum_profit(returns_trace, print_reward=True):
    """
    Calculate the cumulative profit for each episode.
    """
    rewards_trace = []
    for return_trace in returns_trace:
        rewards_trace.append(
            np.sum(return_trace.T[2]))

    if print_reward:
        print(f"reward: mean "
              f"{np.mean(rewards_trace)}, "
              f"std "
              f"{np.std(rewards_trace)}, "
              f"max "
              f"{np.max(rewards_trace)}, "
              f"min "
              f"{np.min(rewards_trace)}")

    return rewards_trace


def visualize_cum_profit(rewards_trace, algorithm,
                         local_dir=local_dir, plots_dir=plots_dir):
    """
    Visualize the cumulative profit boxplot along the episodes.
    """
    xticks = []
    if not isinstance(algorithm, list):
        xticks.append(algorithm)
    else:
        xticks = algorithm

    plt.figure(figsize=(15, 5))
    plt.boxplot(rewards_trace)

    plt.ylabel('Cumulative Profit')
    plt.xticks(np.arange(1,
                         len(xticks)+1),
               xticks)
    plt.tick_params(axis='x', which='both',
                    top=False, bottom=True,
                    labelbottom=True)
    plt.ticklabel_format(axis='y', style='plain',
                         useOffset=False)
    plt.tight_layout()

    # creating necessary subdir and saving plot
    if not os.path.exists(f"{local_dir}/{plots_dir}/{algorithm}"):
        os.makedirs(f"{local_dir}/{plots_dir}/{algorithm}")
    plt.savefig(f"{local_dir}/{plots_dir}/{algorithm}"
                f"/cum_profit_{algorithm}.pdf",
                format='pdf', bbox_inches='tight')

    # saving the cumulative profit as text
    if not isinstance(algorithm, list):
        f = open(f"{local_dir}/{plots_dir}/{algorithm}"
                 f"/cum_profit_{algorithm}.txt",
                 'w', encoding='utf-8')
        f.write(f"reward: mean "
                f"{np.mean(rewards_trace)}, "
                f"std "
                f"{np.std(rewards_trace)}, "
                f"max "
                f"{np.max(rewards_trace)}, "
                f"min "
                f"{np.min(rewards_trace)}")
        f.close()


# # Oracle

# In[ ]:


# cumulative profit of the oracle
cum_profit_oracle = simulate_oracle(env, num_episodes)


# In[ ]:


visualize_cum_profit(cum_profit_oracle, 'Oracle')


# # (s, Q)-Policy Class

# In[ ]:


class SQPolicy:
    """
    To assess and compare performances achieved by the adopted DRL algorithms, 
    we implement a static reorder policy known in the specialized literature as 
    the (s, Q)-policy. This policy can be expressed by a rule, which can be 
    summarized as follows: at each time step t, the current stock level for a 
    specific warehouse and product type is compared to the reorder point s. 
    If the stock level falls below the reorder point s, then the (s, Q)-policy 
    orders Q units of product; otherwise, it does not take any action.
    """

    def __init__(self, factory_s, factory_Q, warehouses_s, warehouses_Q):
        self.factory_s = factory_s
        self.factory_Q = factory_Q
        self.warehouses_s = warehouses_s
        self.warehouses_Q = warehouses_Q

        logger.debug(f"\n--- SQPolicy --- __init__"
                     f"\nfactory_s is "
                     f"{self.factory_s}"
                     f"\nfactory_Q is "
                     f"{self.factory_Q}"
                     f"\nwarehouses_s is "
                     f"{self.warehouses_s}"
                     f"\nwarehouses_Q is "
                     f"{self.warehouses_Q}")

    def select_action(self, state):
        action = Action(state.product_types_num, state.distr_warehouses_num)

        # reordering decisions are made independently for factory and
        # distribution warehouses, so policy parameters s and Q can be
        # different for each warehouse
        for j in range(state.distr_warehouses_num):
            for i in range(state.product_types_num):
                if state.distr_warehouses_stocks[j][i] < \
                        self.warehouses_s[j][i]:
                    action.shipped_stocks[j][i] = \
                        self.warehouses_Q[j][i]

        for i in range(state.product_types_num):
            if (state.factory_stocks[i] -
                    np.sum(action.shipped_stocks, axis=0)[i]) < \
                    self.factory_s[i]:
                action.production_level[i] = \
                    self.factory_Q[i]

        logger.debug(f"\n--- SQPolicy --- select_action"
                     f"\nstate is "
                     f"{state}"
                     f"\naction is "
                     f"{action}"
                     f"\nstate.distr_warehouses_stocks is "
                     f"{state.distr_warehouses_stocks}"
                     f"\nwarehouses_s is "
                     f"{self.warehouses_s}"
                     f"\nwarehouses_Q is "
                     f"{self.warehouses_Q}"
                     f"\naction.shipped_stocks is "
                     f"{action.shipped_stocks}"
                     f"\nstate.factory_stocks is "
                     f"{state.factory_stocks}"
                     f"\nnp.sum(action.shipped_stocks, axis=0) is "
                     f"{np.sum(action.shipped_stocks, axis=0)}"
                     f"\nstate.factory_stocks - "
                     f"np.sum(action.shipped_stocks, axis=0) is "
                     f"""{state.factory_stocks -
                         np.sum(action.shipped_stocks, axis=0)}"""
                     f"\nfactory_s is "
                     f"{self.factory_s}"
                     f"\nfactory_Q is "
                     f"{self.factory_Q}"
                     f"\naction.production_level is "
                     f"{action.production_level}")

        return action


# # (s, Q)-Policy Config [Ax]

# ## Parameters [Ax]

# In[ ]:


# total trials for Ax optimization
total_trials_Ax = 200
# number of episodes for each trial
num_episodes_Ax = [25, 75, 200]
# number of iterations for each number of episodes
iterations_Ax = 3


# ## Parameters Methods [Ax]

# In[ ]:


def create_parameters_Ax(env):
    """
    Create Ax (s, Q)-policy parameters (s and Q) for the factory, the 
    distribution warehouses and for each product type.
    """
    # factory parameters
    factory_parameters = [
        {'name': 'factory_s_',
         'type': 'range',
         'value_type': 'int', },
        {'name': 'factory_Q_',
         'type': 'range',
         'value_type': 'int', },
    ]

    factory_parameters_Ax = []

    # factory parameters (s and Q) for each product type, according to storage
    # capacities
    for factory_parameter in factory_parameters:
        for i in range(env.product_types_num):
            factory_parameters_Ax.append(
                {**factory_parameter,
                 'name': factory_parameter['name'] + str(i+1),
                 'bounds': [0, env.storage_capacities[0][i].item(0)], })

    # distribution warehouses parameters
    w_parameters = [
        {'name': 'w',
         'type': 'range',
         'value_type': 'int', },
    ]

    w_parameters_Ax = []

    # distribution warehouses parameters (s and Q) for each product type,
    # according to storage capacities
    for w_parameter in w_parameters:
        for j in range(env.distr_warehouses_num):
            for i in range(env.product_types_num):
                w_parameters_Ax.append(
                    {**w_parameter,
                     'name': w_parameter['name'] + str(j+1) + '_s_' + str(i+1),
                     'bounds': [0, env.storage_capacities[j+1][i].item(0)], })
                w_parameters_Ax.append(
                    {**w_parameter,
                     'name': w_parameter['name'] + str(j+1) + '_Q_' + str(i+1),
                     'bounds': [0, env.storage_capacities[j+1][i].item(0)], })

    # final Ax parameters
    parameters_Ax = factory_parameters_Ax + w_parameters_Ax

    logger.debug(f"\n-- create_parameters_Ax --"
                 f"\nparameters_Ax is "
                 f"{parameters_Ax}")

    return parameters_Ax


def save_checkpoint(checkpoint, algorithm,
                    local_dir=local_dir, plots_dir=plots_dir):
    """
    Save Ax (s, Q)-policy parameters or RLib Agent checkpoint.
    """
    f = open(f"{local_dir}/{plots_dir}/{algorithm}"
             f"/best_checkpoint_{algorithm}.txt",
             'w', encoding='utf-8')
    f.write(checkpoint)
    f.close()


# # (s, Q)-Policy Methods [Ax]

# ## Optimize Methods [Ax]

# In[ ]:


def opt_func_Ax(p):
    """
    Evaluation function to optimize (to maximize).
    """
    args = [[],
            [],
            [],
            []]

    # nested list for warehouses parameters
    args[2] = [[] for _ in range(env.distr_warehouses_num)]
    args[3] = [[] for _ in range(env.distr_warehouses_num)]

    for i in range(env.product_types_num):
        args[0].append(p[f"factory_s_{i+1}"])
        args[1].append(p[f"factory_Q_{i+1}"])
        for j in range(env.distr_warehouses_num):
            args[2][j].append(p[f"w{j+1}_s_{i+1}"])
            args[3][j].append(p[f"w{j+1}_Q_{i+1}"])

    policy = SQPolicy(*args)

    return np.mean(calculate_cum_profit(simulate(env, policy, num_episodes),
                                        print_reward=False))


def optimize_Ax(num_episodes_Ax, iterations_Ax, parameters_Ax, total_trials_Ax,
                seed):
    """
    Brute force search through the parameter space using the Adaptive
    Experimentation Platform developed by Facebook. This framework provides a
    very convenient API and uses Bayesian optimization internally.
    """
    best_mean_cum_profit_Ax = None
    for num_episodes in num_episodes_Ax:
        for iteration in range(iterations_Ax):
            # Ax optimization runs total_trials times, each trial sees a given
            # number of episodes and this procedure is repeated for each
            # iteration, searching the best parameters for the Ax (s, Q)-policy
            start_Ax = default_timer()
            parameters, values, experiment, model = optimize(
                parameters=parameters_Ax,
                evaluation_function=opt_func_Ax,
                objective_name='episode_reward_mean',
                minimize=False,
                total_trials=total_trials_Ax,
                random_seed=seed)
            end_Ax = default_timer()

            # setting the optimised parameters for current num episodes and
            # iteration
            policy_Ax = SQPolicy(
                [parameters[f"factory_s_{i+1}"]
                 for i in range(env.product_types_num)],
                [parameters[f"factory_Q_{i+1}"]
                 for i in range(env.product_types_num)],
                [[parameters[f"w{j+1}_s_{i+1}"]
                  for i in range(env.product_types_num)]
                 for j in range(env.distr_warehouses_num)],
                [[parameters[f"w{j+1}_Q_{i+1}"]
                  for i in range(env.product_types_num)]
                 for j in range(env.distr_warehouses_num)])

            # evaluating the Ax (s, Q)-policy for current num episodes and
            # iteration
            returns_trace_Ax = simulate(env, policy_Ax, num_episodes)

            # printing current num episodes and iteration
            print(f"--num episodes: {num_episodes}, "
                  f"iteration: {iteration+1}")

            # cumulative profit of the Ax (s, Q)-policy for current
            # num episodes and iteration
            cum_profit_Ax = calculate_cum_profit(returns_trace_Ax)
            visualize_cum_profit(cum_profit_Ax,
                                 f"sQ_{num_episodes}_{iteration+1}")

            # finding the best parameters for the Ax (s, Q)-policy
            if (best_mean_cum_profit_Ax is None or
                    np.mean(cum_profit_Ax) > best_mean_cum_profit_Ax):
                best_mean_cum_profit_Ax = np.mean(cum_profit_Ax)
                best_num_episodes = num_episodes
                best_iteration = iteration
                best_parameters = parameters
                best_values = values
                best_experiment = experiment
                best_model = model
                time_Ax = int((end_Ax-start_Ax) // 60)
                best_policy_Ax = policy_Ax
                best_returns_trace_Ax = returns_trace_Ax
                best_cum_profit_Ax = cum_profit_Ax

    logger.debug(f"\n-- optimize_Ax --"
                 f"\nbest_mean_cum_profit_Ax is "
                 f"{best_mean_cum_profit_Ax}"
                 f"\nbest_num_episodes is "
                 f"{best_num_episodes}"
                 f"\nbest_iteration is "
                 f"{best_iteration}"
                 f"\nbest_parameters is "
                 f"{best_parameters}"
                 f"\nbest_values is "
                 f"{best_values}"
                 f"\nbest_experiment is "
                 f"{best_experiment}"
                 f"\nbest_model is "
                 f"{best_model}"
                 f"\ntime_Ax is "
                 f"{time_Ax}"
                 f"\nbest_policy_Ax is "
                 f"{best_policy_Ax}"
                 f"\nbest_returns_trace_Ax is "
                 f"{best_returns_trace_Ax}"
                 f"\nbest_cum_profit_Ax is "
                 f"{best_cum_profit_Ax}")

    return (best_mean_cum_profit_Ax, best_num_episodes, best_iteration,
            best_parameters, best_values, best_experiment, best_model,
            time_Ax, best_policy_Ax, best_returns_trace_Ax, best_cum_profit_Ax)


# ## Visualize Rewards Methods [Ax]

# In[ ]:


def visualize_optimization_trace_Ax(experiment, verbose,
                                    local_dir=local_dir, plots_dir=plots_dir):
    """
    Plot the mean reward along the trials iterations.
    """
    try:
        best_objectives = np.array(
            [[trial.objective_mean
              for trial in experiment.trials.values()]])
        best_objective_plot = optimization_trace_single_method_plotly(
            y=np.maximum.accumulate(best_objectives, axis=1),
            ylabel='Reward Mean',
            title='sQ Performance vs. Trials Iterations')
        if verbose == 3:
            best_objective_plot.show()
        # creating necessary subdir and saving plot
        if not os.path.exists(f"{local_dir}/{plots_dir}"
                              f"/sQ"):
            os.makedirs(f"{local_dir}/{plots_dir}"
                        f"/sQ")
        best_objective_plot.write_image(f"{local_dir}/{plots_dir}"
                                        f"/sQ"
                                        f"/optimization_trace.pdf")
    except Exception as e:
        print(f"{e.__class__} occurred!")


def visualize_contour_Ax(model, parameters, verbose,
                         local_dir=local_dir, plots_dir=plots_dir):
    """
    Plot the contours, showing the episode reward mean as a function of two
    selected parameters (e.g., 'factory_s_1' and 'factory_Q_1').
    """
    try:
        # creating necessary subdir
        if not os.path.exists(f"{local_dir}/{plots_dir}"
                              f"/sQ/contours"):
            os.makedirs(f"{local_dir}/{plots_dir}"
                        f"/sQ/contours")
        # saving plots
        for p in range(0, len(parameters), 2):
            contour = plot_contour_plotly(model=model,
                                          metric_name='episode_reward_mean',
                                          param_x=list(parameters)[p],
                                          param_y=list(parameters)[p+1])
            contour.write_image(f"{local_dir}/{plots_dir}"
                                f"/sQ/contours"
                                f"/contour_"
                                f"{list(parameters)[p]}_"
                                f"{list(parameters)[p+1]}.pdf")
            if verbose == 3:
                contour.show()

        # interactive contour plot
        if verbose == 3:
            render(interact_contour(model=best_model_Ax,
                   metric_name='episode_reward_mean'))
    except Exception as e:
        print(f"{e.__class__} occurred!")


def move_dir_Ax(local_dir=local_dir, plots_dir=plots_dir):
    """
    Move dirs whose name starts with 'sQ_' (related to all optimizations) in
    the main sQ dir.
    """
    try:
        src_dir = f"{local_dir}/{plots_dir}/"
        dst_dir = f"{local_dir}/{plots_dir}/sQ"

        pattern = src_dir + "sQ_*"
        for file in glob.iglob(pattern, recursive=True):
            shutil.move(file, dst_dir)
            print('moved:', file)
    except Exception as e:
        print(f"{e.__class__} occurred!")


# # (s, Q)-Policy Optimize [Ax]

# In[ ]:


# Ax parameters
parameters_Ax = create_parameters_Ax(env)
parameters_Ax


# In[ ]:


# Ax optimization
(best_mean_cum_profit_Ax, best_num_episodes_Ax, best_iteration_Ax,
 best_parameters_Ax, best_values_Ax, best_experiment_Ax, best_model_Ax,
 time_Ax, best_policy_Ax_Ax, best_returns_trace_Ax, best_cum_profit_Ax) = \
    optimize_Ax(num_episodes_Ax, iterations_Ax, parameters_Ax, total_trials_Ax,
                seed)


# In[ ]:


# printing the best Ax mean cum profit with related num episodes and iteration
print(f"best num episodes is {best_num_episodes_Ax} "
      f"at iteration {best_iteration_Ax+1} "
      f"\nmean cum profit: {best_mean_cum_profit_Ax}")


# In[ ]:


visualize_optimization_trace_Ax(best_experiment_Ax, verbose)


# In[ ]:


visualize_contour_Ax(best_model_Ax, best_parameters_Ax, verbose)


# In[ ]:


# printing the Ax optimization time
print(f"sQ optimization time (in minutes) is {time_Ax}")


# In[ ]:


# displaying and saving the Ax best parameters
display(best_parameters_Ax)
save_checkpoint(str(best_parameters_Ax), 'sQ')


# In[ ]:


visualize_transitions(best_returns_trace_Ax, 'sQ')


# In[ ]:


visualize_cum_profit(best_cum_profit_Ax, 'sQ')


# In[ ]:


# moving Ax optimization dirs in the main Ax dir
move_dir_Ax()


# # Reinforcement Learning Config [Tune]

# ## Parameters [Tune]

# In[ ]:


# number of episodes for RLib agents
num_episodes_ray = 50000
# stop trials at least from this number of episodes
grace_period_ray = num_episodes_ray / 10


# In[ ]:


# dir for saving Ray results
ray_dir = 'ray_results'
# creating necessary dir
if not os.path.exists(f"{local_dir+'/'+ray_dir}"):
    os.makedirs(f"{local_dir+'/'+ray_dir}")


# ## Algorithms [Tune]

# In[ ]:


# https://docs.ray.io/en/latest/rllib-algorithms.html
# https://docs.ray.io/en/master/rllib-training.html#common-parameters
# adopted algorithms
algorithms = {
    'A3C': a3c.A3CTrainer,
    'PG': pg.PGTrainer,
    'PPO': ppo.PPOTrainer,
}


# ## A3C Config [Tune]

# In[ ]:


# https://docs.ray.io/en/master/rllib-algorithms.html#a3c
config_A3C = a3c.DEFAULT_CONFIG.copy()
config_A3C['seed'] = seed
config_A3C['log_level'] = 'WARN'

config_A3C['env'] = SupplyChain
config_A3C['horizon'] = env.T-1

config_A3C['model']['fcnet_hiddens'] = tune.grid_search([[64, 64],
                                                         [128, 128]])
config_A3C['lr'] = tune.grid_search([1e-3,
                                     1e-4])
config_A3C['gamma'] = .99

config_A3C['rollout_fragment_length'] = tune.grid_search([10,
                                                          100])
config_A3C['train_batch_size'] = tune.grid_search([200,
                                                   2000])

config_A3C['grad_clip'] = tune.grid_search([20.0,
                                            40.0])

config_A3C['evaluation_num_episodes'] = 1000
config_A3C['sample_async'] = False

config_A3C['num_workers'] = num_cpus-1
config_A3C['num_gpus'] = num_gpus

config_A3C['framework'] = 'torch'


# ## PG Config [Tune]

# In[ ]:


# https://docs.ray.io/en/master/rllib-algorithms.html#policy-gradients
config_PG = pg.DEFAULT_CONFIG.copy()
config_PG['seed'] = seed
config_PG['log_level'] = 'WARN'

config_PG['env'] = SupplyChain
config_PG['horizon'] = env.T-1

config_PG['model']['fcnet_hiddens'] = tune.grid_search([[64, 64],
                                                        [128, 128]])
config_PG['lr'] = tune.grid_search([4e-3,
                                    4e-4])
config_PG['gamma'] = .99

config_PG['rollout_fragment_length'] = tune.grid_search([10,
                                                         100])
config_PG['train_batch_size'] = tune.grid_search([200,
                                                  2000])

config_PG['evaluation_num_episodes'] = 1000
config_PG['sample_async'] = False

config_PG['num_workers'] = num_cpus-1
config_PG['num_gpus'] = num_gpus

config_PG['framework'] = 'torch'


# ## PPO Config [Tune]

# In[ ]:


# https://docs.ray.io/en/master/rllib-algorithms.html#ppo
config_PPO = ppo.DEFAULT_CONFIG.copy()
config_PPO['seed'] = seed
config_PPO['log_level'] = 'WARN'

config_PPO['env'] = SupplyChain
config_PPO['horizon'] = env.T-1

config_PPO['model']['fcnet_hiddens'] = tune.grid_search([[64, 64],
                                                         [128, 128]])
config_PPO['lr'] = tune.grid_search([5e-3,
                                     5e-4])
config_PPO['gamma'] = .99

config_PPO['rollout_fragment_length'] = tune.grid_search([20,
                                                          200])
config_PPO['train_batch_size'] = tune.grid_search([400,
                                                   4000])

config_PPO['grad_clip'] = tune.grid_search([None,
                                            20.0])
config_PPO['num_sgd_iter'] = tune.grid_search([15,
                                               30])
config_PPO['sgd_minibatch_size'] = tune.grid_search([64,
                                                     128])

config_PPO['evaluation_num_episodes'] = 1000
config_PPO['sample_async'] = False

config_PPO['num_workers'] = num_cpus-1
config_PPO['num_gpus'] = num_gpus

config_PPO['framework'] = 'torch'


# # Reinforcement Learning Methods [Tune]

# ## Train Agents Methods [Tune]

# In[ ]:


def train(algorithm, config, verbose,
          num_episodes_ray=num_episodes_ray, grace_period_ray=grace_period_ray,
          local_dir=local_dir, ray_dir=ray_dir):
    """
    Train a RLib Agent.
    """
    # initializing Ray
    ray.shutdown()
    ray.init(log_to_driver=False)

    logger.debug(f"\n-- train --"
                 f"\nalgorithm is "
                 f"{algorithm}"
                 f"\nconfig is "
                 f"{config}")

    # https://docs.ray.io/en/latest/tune/api_docs/execution.html
    # https://docs.ray.io/en/master/tune/api_docs/schedulers.html#summary
    # https://docs.ray.io/en/master/tune/api_docs/analysis.html#id1
    analysis = tune.run(algorithm,
                        config=config,
                        metric='episode_reward_mean',
                        mode='max',
                        scheduler=ASHAScheduler(
                            time_attr='episodes_total',
                            max_t=num_episodes_ray,
                            grace_period=grace_period_ray,
                            reduction_factor=5),
                        checkpoint_freq=1,
                        keep_checkpoints_num=1,
                        checkpoint_score_attr='episode_reward_mean',
                        progress_reporter=tune.JupyterNotebookReporter(
                            overwrite=True),
                        verbose=verbose,
                        local_dir=os.getcwd()+'/'+local_dir+'/'+ray_dir)

    trial_dataframes = analysis.trial_dataframes
    best_result_df = analysis.best_result_df
    best_config = analysis.best_config
    best_checkpoint = analysis.best_checkpoint
    print(f"\ncheckpoint saved at {best_checkpoint}")

    # stopping Ray
    ray.shutdown()

    return trial_dataframes, best_result_df, best_config, best_checkpoint


def result_df_as_image(result_df, algorithm,
                       local_dir=local_dir, plots_dir=plots_dir):
    """
    Visualize the (DataFrame) RLib Agent's result as an image.
    """
    # creating necessary subdir and saving plot
    if not os.path.exists(f"{local_dir}/{plots_dir}/{algorithm}"):
        os.makedirs(f"{local_dir}/{plots_dir}/{algorithm}")
    dfi.export(result_df.iloc[:, np.r_[:3, 9]],
               f"{local_dir}/{plots_dir}/{algorithm}"
               f"/best_result_{algorithm}.png",
               table_conversion='matplotlib')


def calculate_training_time(result_df):
    """
    Calculate a RLib Agent training time (minutes).
    """
    return int(result_df.time_total_s[0]//60)


def calculate_training_episodes(result_df):
    """
    Calculate a RLib Agent training episodes (number).
    """
    return round(result_df.episodes_total[0], -3)


# ## Policy Methods [Tune]

# In[ ]:


def load_policy(algorithm, config, checkpoint):
    """
    Load a RLib Agent policy.
    """
    # initializing Ray
    ray.shutdown()
    ray.init(log_to_driver=False)

    # loading policy
    trainer = algorithm(config=config)
    trainer.restore(f"{checkpoint}")
    policy = trainer.get_policy()

    # stopping Ray
    ray.shutdown()

    logger.debug(f"\n-- load_policy --"
                 f"\nalgorithm is "
                 f"{algorithm}"
                 f"\nconfig is "
                 f"{config}"
                 f"\ncheckpoint is "
                 f"{checkpoint}"
                 f"\ntrainer is "
                 f"{trainer}"
                 f"\npolicy is "
                 f"{policy}")

    return policy


def fix_best_checkpoint(checkpoint):
    """
    Fix a RLib Agent best checkpoint path.
    """
    # searching all checkpoints related to the best agent's result
    checkpoint_dir = checkpoint.rsplit('/', 2)[0]
    sub_dirs = [sub_dir for sub_dir in os.listdir(checkpoint_dir)
                if os.path.isdir(os.path.join(checkpoint_dir, sub_dir))]
    # finding the most recent checkpoint (the best one)
    sub_dirs.sort(reverse=True)

    # creating the fixed best checkpoint path
    fixed_checkpoint_dir = checkpoint_dir + '/' + sub_dirs[0] + '/'
    fixed_checkpoint_file = os.listdir(fixed_checkpoint_dir)[0].split('.')[0]
    best_checkpoint = fixed_checkpoint_dir + fixed_checkpoint_file

    logger.debug(f"\n-- fix_best_checkpoint --"
                 f"\nfixed_checkpoint_dir is "
                 f"{fixed_checkpoint_dir}"
                 f"\nfixed_checkpoint_file is "
                 f"{fixed_checkpoint_file}"
                 f"\nbest_checkpoint is "
                 f"{best_checkpoint}")

    return best_checkpoint


# ## Visualize Rewards Methods [Tune]

# In[ ]:


def apply_style_plot(ax, episodes_total):
    """
    Auxiliary function.
    """
    labels = [str(0),
              str(episodes_total//2),
              str(episodes_total)]
    ax.xaxis.set_major_locator(ticker.LinearLocator(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Episodes')
    ax.ticklabel_format(axis='y', style='plain',
                        useOffset=False)


def visualize_rewards(results, best_result, algorithm, legend=[],
                      local_dir=local_dir, plots_dir=plots_dir):
    """
    Visualize the min, mean and max rewards along the episodes.
    """
    # creating necessary subdir and saving plot
    if not os.path.exists(f"{local_dir}/{plots_dir}/{algorithm}"):
        os.makedirs(f"{local_dir}/{plots_dir}/{algorithm}")
    episodes_total = calculate_training_episodes(best_result)

    # min reward
    fig, ax = plt.subplots(figsize=(15, 5))
    apply_style_plot(ax, episodes_total)
    ax.set_ylabel('Min Reward')
    for result in results.values():
        ax = result.episode_reward_min.plot(ax=ax)
    ax.legend(legend, bbox_to_anchor=(1.04, .5), borderaxespad=0,
              frameon=False, loc='center left', fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(f"{local_dir}/{plots_dir}/{algorithm}"
                f"/episode_reward_min_{algorithm}.pdf",
                format='pdf', bbox_inches='tight')

    # mean reward
    fig, ax = plt.subplots(figsize=(15, 5))
    apply_style_plot(ax, episodes_total)
    ax.set_ylabel('Mean Reward')
    for result in results.values():
        ax = result.episode_reward_mean.plot(ax=ax)
    ax.legend(legend, bbox_to_anchor=(1.04, .5), borderaxespad=0,
              frameon=False, loc='center left', fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(f"{local_dir}/{plots_dir}/{algorithm}"
                f"/episode_reward_mean_{algorithm}.pdf",
                format='pdf', bbox_inches='tight')

    # max reward
    fig, ax = plt.subplots(figsize=(15, 5))
    apply_style_plot(ax, episodes_total)
    ax.set_ylabel('Max Reward')
    for result in results.values():
        ax = result.episode_reward_max.plot(ax=ax)
    ax.legend(legend, bbox_to_anchor=(1.04, .5), borderaxespad=0,
              frameon=False, loc='center left', fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(f"{local_dir}/{plots_dir}/{algorithm}"
                f"/episode_reward_max_{algorithm}.pdf",
                format='pdf', bbox_inches='tight')


# # Reinforcement Learning Train Agents [Tune]

# ## A3C Agent [Tune]

# In[ ]:


# training an A3C agent
(results_A3C, best_result_A3C,
 best_config_A3C, checkpoint_A3C) = train(algorithms['A3C'],
                                          config_A3C,
                                          verbose)


# In[ ]:


# saving and showing the best result of the A3C agent
result_df_as_image(best_result_A3C, 'A3C')
best_result_A3C


# In[ ]:


visualize_rewards(results_A3C, best_result_A3C, 'A3C')


# In[ ]:


# calculating and printing the A3C training time
time_A3C = calculate_training_time(best_result_A3C)
print(f"total training time A3C (in minutes) is {time_A3C}")


# In[ ]:


# (fixing and) saving the A3C best checkpoint
best_checkpoint_A3C = fix_best_checkpoint(checkpoint_A3C)
save_checkpoint(best_checkpoint_A3C, 'A3C')


# In[ ]:


# loading the best A3C agent's policy
policy_A3C = load_policy(algorithms['A3C'],
                         best_config_A3C,
                         best_checkpoint_A3C)


# In[ ]:


# evaluating the best A3C agent's policy
returns_trace_A3C = simulate(env, policy_A3C, num_episodes)


# In[ ]:


visualize_transitions(returns_trace_A3C, 'A3C')


# In[ ]:


# cumulative profit of the best A3C agent's policy
cum_profit_A3C = calculate_cum_profit(returns_trace_A3C)


# In[ ]:


visualize_cum_profit(cum_profit_A3C, 'A3C')


# ## PG Agent [Tune]

# In[ ]:


# training a PG agent
(results_PG, best_result_PG,
 best_config_PG, checkpoint_PG) = train(algorithms['PG'],
                                        config_PG,
                                        verbose)


# In[ ]:


# saving and showing the best result of the PG agent
result_df_as_image(best_result_PG, 'PG')
best_result_PG


# In[ ]:


visualize_rewards(results_PG, best_result_PG, 'PG')


# In[ ]:


# calculating and printing the PG training time
time_PG = calculate_training_time(best_result_PG)
print(f"total training time PG (in minutes) is {time_PG}")


# In[ ]:


# (fixing and) saving the PG best checkpoint
best_checkpoint_PG = fix_best_checkpoint(checkpoint_PG)
save_checkpoint(best_checkpoint_PG, 'PG')


# In[ ]:


# loading the best PG agent's policy
policy_PG = load_policy(algorithms['PG'],
                        best_config_PG,
                        best_checkpoint_PG)


# In[ ]:


# evaluating the best PG agent's policy
returns_trace_PG = simulate(env, policy_PG, num_episodes)


# In[ ]:


visualize_transitions(returns_trace_PG, 'PG')


# In[ ]:


# cumulative profit of the best PG agent's policy
cum_profit_PG = calculate_cum_profit(returns_trace_PG)


# In[ ]:


visualize_cum_profit(cum_profit_PG, 'PG')


# ## PPO Agent [Tune]

# In[ ]:


# training a PPO agent
(results_PPO, best_result_PPO,
 best_config_PPO, checkpoint_PPO) = train(algorithms['PPO'],
                                          config_PPO,
                                          verbose)


# In[ ]:


# saving and showing the best result of the PPO agent
result_df_as_image(best_result_PPO, 'PPO')
best_result_PPO


# In[ ]:


visualize_rewards(results_PPO, best_result_PPO, 'PPO')


# In[ ]:


# calculating and printing the PPO training time
time_PPO = calculate_training_time(best_result_PPO)
print(f"total training time PPO (in minutes) is {time_PPO}")


# In[ ]:


# (fixing and) saving the PPO best checkpoint
best_checkpoint_PPO = fix_best_checkpoint(checkpoint_PPO)
save_checkpoint(best_checkpoint_PPO, 'PPO')


# In[ ]:


# loading the best PPO agent's policy
policy_PPO = load_policy(algorithms['PPO'],
                         best_config_PPO,
                         best_checkpoint_PPO)


# In[ ]:


# evaluating the best PPO agent's policy
returns_trace_PPO = simulate(env, policy_PPO, num_episodes)


# In[ ]:


visualize_transitions(returns_trace_PPO, 'PPO')


# In[ ]:


# cumulative profit of the best PPO agent's policy
cum_profit_PPO = calculate_cum_profit(returns_trace_PPO)


# In[ ]:


visualize_cum_profit(cum_profit_PPO, 'PPO')


# # Final Results

# ## Cumulative Profit

# In[ ]:


visualize_cum_profit([cum_profit_A3C,
                      cum_profit_PG,
                      cum_profit_PPO],
                     ['A3C',
                      'PG',
                      'PPO'])


# In[ ]:


visualize_cum_profit([cum_profit_A3C,
                      cum_profit_PG,
                      cum_profit_PPO,
                      cum_profit_oracle],
                     ['A3C',
                      'PG',
                      'PPO',
                      'Oracle'])


# In[ ]:


visualize_cum_profit([cum_profit_A3C,
                      cum_profit_PG,
                      cum_profit_PPO,
                      best_cum_profit_Ax],
                     ['A3C',
                      'PG',
                      'PPO',
                      'sQ'])


# In[ ]:


visualize_cum_profit([cum_profit_A3C,
                      cum_profit_PG,
                      cum_profit_PPO,
                      best_cum_profit_Ax,
                      cum_profit_oracle],
                     ['A3C',
                      'PG',
                      'PPO',
                      'sQ',
                      'Oracle'])


# ## Training Time

# In[ ]:


# training time of all policies
times_total = {'Algorithm':
               ['A3C',
                'PG',
                'PPO',
                f"sQ_{best_num_episodes_Ax}_{best_iteration_Ax+1}"],
               'Training Time \n(in minutes)':
               [time_A3C,
                time_PG,
                time_PPO,
                time_Ax]}
# creating pandas DataFrame
times_total_df = pd.DataFrame(data=times_total)
times_total_df.set_index('Algorithm', inplace=True)
# saving pandas DataFrame as an image
dfi.export(times_total_df,
           f"{local_dir}/{plots_dir}"
           f"/times_total_df.png",
           table_conversion='matplotlib')
# printing training time of all policies
print(tabulate(times_total_df, headers='keys', tablefmt='grid'))


# # Compress Final Results

# In[ ]:


# creating a tar file containing plots and Ray results
try:
    cmd = f"tar -zcvf {local_dir}.tar.gz ./{local_dir}"
    print(f"cmd is {cmd}")
    os.system(cmd)
except Exception as e:
    print(f"{e.__class__} occurred!")


# # TensorBoard

# In[ ]:


# checking if A3C best checkpoint is defined
try:
    best_checkpoint_A3C
except Exception as e:
    print(f"{e.__class__} occurred!")
    best_checkpoint_A3C = None


# In[ ]:


# checking if PG best checkpoint is defined
try:
    best_checkpoint_PG
except Exception as e:
    print(f"{e.__class__} occurred!")
    best_checkpoint_PG = None


# In[ ]:


# checking if PPO best checkpoint is defined
try:
    best_checkpoint_PPO
except Exception as e:
    print(f"{e.__class__} occurred!")
    best_checkpoint_PPO = None


# In[ ]:


# TensorBoard dir for Ray results (the first best checkpoint not None)
tb_dir = next(checkpoint for checkpoint in [best_checkpoint_A3C,
                                            best_checkpoint_PG,
                                            best_checkpoint_PPO]
              if checkpoint is not None).rsplit('/', 4)[0]
tb_dir


# In[ ]:


# loading TensorBoard
try:
    get_ipython().run_line_magic('load_ext', 'tensorboard')
    get_ipython().run_line_magic('tensorboard', '--logdir $tb_dir')
except Exception as e:
    print(f"{e.__class__} occurred!")


# In[ ]:




