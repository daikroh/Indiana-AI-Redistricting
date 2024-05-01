# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 17:19:56 2024

@author: itsam
"""

from gerrychain import (GeographicPartition, Graph, 
                        updaters, Election)
import numpy as np
import statistics
import matplotlib.pyplot as plt
from gerrychain import Graph, Partition, proposals, updaters, constraints, accept, MarkovChain, Election
from gerrychain.updaters import cut_edges, Tally
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from functools import partial
import geopandas as gpd
import pandas as pd


gdf = gpd.read_file('./IN/IN.shp')

gdf = gdf.fillna(value={'ALL_TOT20': 0})
gdf = gdf.fillna(value={'VAP_TOT20': 0})
graph = Graph.from_geodataframe(gdf)

tot_pop = sum([graph.nodes()[v]['ALL_TOT20'] for v in graph.nodes()])
num_dist = 9 # Number of Congressional Districts in Indiana
ideal_pop = tot_pop/num_dist
pop_tolerance = 0.02

# Define a function to determine the number of districts won by the Republican party
def republican_wins(partition):
    republican_won = 0
    for district in partition.parts:
        democratic_votes = partition["dem votes"][district]
        republican_votes = partition["rep votes"][district]
        total_votes = democratic_votes + republican_votes
        if republican_votes / total_votes > 0.5:
            republican_won += 1
    return republican_won

# Define a function to determine the mean median difference
def mean_median_difference(partition):
    republican_voting_population = sum(partition["rep votes"].values())
    democratic_voting_population = sum(partition["dem votes"].values())

    rep_median = 0
    dem_median = 0

    total_population = sum(partition["population"].values())

    for district in partition.parts:
        rep_median += partition["rep votes"][district] / republican_voting_population * partition["population"][district]
        dem_median += partition["dem votes"][district] / democratic_voting_population * partition["population"][district]

    mean_median_diff = (rep_median - dem_median) / total_population

    return mean_median_diff

# Define a function to determine the efficiency gap
def efficiency_gap(partition):
    republican_wasted_votes = 0
    democratic_wasted_votes = 0

    for district in partition.parts:
        democratic_votes = partition["dem votes"][district]
        republican_votes = partition["rep votes"][district]
        total_votes = democratic_votes + republican_votes

        if democratic_votes > republican_votes:
            democratic_wasted_votes += (democratic_votes - (total_votes / 2))
            republican_wasted_votes += republican_votes
        else:
            republican_wasted_votes += (republican_votes - (total_votes / 2))
            democratic_wasted_votes += democratic_votes

    total_votes = sum(partition["population"].values())

    efficiency_gap_value = (republican_wasted_votes - democratic_wasted_votes) / total_votes

    return efficiency_gap_value


initial_partition = Partition(
      graph, 
      assignment="CD",
      updaters={
          "cut_edges": cut_edges, 
          "population": Tally("ALL_TOT20", alias="population"),  
          "rep votes": Tally("PRES20R", alias = "rep votes"),
          "dem votes": Tally("PRES20D", alias = "dem votes"),
          "republican_won": republican_wins,
          "efficiency_gap": efficiency_gap,
          "mean_median_difference": mean_median_difference,
      }
)

# Define proposal and constraints
rw_proposal = partial(recom, ## how you choose a next districting plan
                      pop_col = "ALL_TOT20", ## What data describes population? 
                      pop_target = ideal_pop, ## What the target/ideal population is for each district 
                                             
                      epsilon = pop_tolerance,  ## how far from ideal population you can deviate
                                              
                      node_repeats = 1 ## number of times to repeat bipartition.  Can increase if you get a BipartitionWarning
                      )

population_constraint = constraints.within_percent_of_ideal_population(
    initial_partition, 
    pop_tolerance, 
    pop_key="population")

# Creating the Markov Chain
our_random_walk = MarkovChain(
    proposal = rw_proposal, 
    constraints = [population_constraint],
    accept = always_accept, # Accept every proposed plan that meets the population constraints
    initial_state = initial_partition, 
    total_steps = 8000) 

# Lists to store data for histograms
cutedge_ensemble = []
republican_won_ensemble = []
efficiency_gap_ensemble = []
mean_median_difference_ensemble = []

# Running the Markov Chain
for part in our_random_walk:
    # Add cutedges to cutedges ensemble
    cutedge_ensemble.append(len(part["cut_edges"]))
    
    republican_won_ensemble.append(part["republican_won"])
    
    efficiency_gap_ensemble.append(part["efficiency_gap"])
    
    mean_median_difference_ensemble.append(part["mean_median_difference"])

plt.figure()
plt.hist(cutedge_ensemble, align = 'left')
plt.title("Histogram of Cut Edges")
plt.show()
#plt.savefig('histogram_cut_edges.png')

plt.figure()
plt.hist(republican_won_ensemble, align = 'left')
plt.title("Histogram of Republican-won districts")
plt.show()
#plt.savefig('histogram_republican_won.png')

plt.figure()
plt.hist(efficiency_gap_ensemble, align = 'left')
plt.title("Histogram of Efficiency Gap")
plt.show()
#plt.savefig('histogram_efficiency_gap.png')

plt.figure()
plt.hist(mean_median_difference_ensemble, align = 'left')
plt.title("Histogram of Mean Median Difference")
plt.show()
#plt.savefig('histogram_mean_median.png')