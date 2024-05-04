# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 19:50:53 2024

@author: itsam
"""

from gerrychain.random import random
random.seed(12345678)
import matplotlib.pyplot as plt
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from functools import partial
import pandas as pd
import geopandas as gpd
import csv
import os
import datetime  #For keeping track of runtime
import tqdm # for progress bar

beginrun = datetime.datetime.now()
print ("\nBegin date and time : ", beginrun.strftime("%Y-%m-%d %H:%M:%S"))

total_steps_in_run=500
save_district_graph_mod=1
save_district_plot_mod=100

# Reading in the shapefile
outdir="./IN_recom_2020_CD/"
os.makedirs(outdir, exist_ok=True)
gdf = gpd.read_file('./IN/IN.shp')

gdf = gdf.fillna(value={'ALL_TOT20': 0})
gdf = gdf.fillna(value={'VAP_TOT20': 0})
graph = Graph.from_geodataframe(gdf)

# Setting up the election updaters
elections = [
    Election("PRES20", {"Republican": "PRES20R", "Democratic": "PRES20D"}),
]

my_updaters = {"population": updaters.Tally("ALL_TOT20", alias="population")}
election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

# Setting up the Partition
initial_partition = GeographicPartition(graph, 
                                        assignment= "CD", updaters=my_updaters)

ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

proposal = partial(recom,
                   pop_col="ALL_TOT20",
                   pop_target=ideal_population,
                   epsilon=0.02,
                   node_repeats=2
                  )
# Constraints
compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]),
    2*len(initial_partition["cut_edges"])
)

pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.02)

# Running a Markov Chain
chain = MarkovChain(
    proposal=proposal,
    constraints=[
        pop_constraint,
        compactness_bound
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=total_steps_in_run
    )

# Creating the Box Plot
data = pd.DataFrame(
    sorted(partition["PRES20"].percents("Republican"))
    for partition in chain.with_progress_bar()
)

fig, ax = plt.subplots(figsize=(8, 6))

# Draw 50% line
ax.axhline(0.5, color="#cccccc")

# Draw boxplot
data.boxplot(ax=ax, positions=range(len(data.columns)))

# Draw initial plan's Democratic vote %s (.iloc[0] gives the first row, which corresponds to the initial plan)
plt.plot(data.iloc[0], "ro")

# Annotate plot
ax.set_title("Comparing the 2020 plan to an ensemble")
ax.set_ylabel("Republican vote % (Presidnetial 2020)")
ax.set_xlabel("Sorted districts")
ax.set_ylim(0.25, 0.75)
ax.set_yticks([0.25, 0.5, 0.75])

plt.show()
plt.savefig('marginal_box_plot20.png')