# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 19:04:32 2024

@author: itsam
"""

from gerrychain import (GeographicPartition, Graph, 
                        updaters, Election, metrics)
import geopandas as gpd

gdf = gpd.read_file('./IN/IN.shp')

gdf = gdf.fillna(value={'ALL_TOT20': 0})
gdf = gdf.fillna(value={'VAP_TOT20': 0})
graph = Graph.from_geodataframe(gdf)

# Population updater, for computing how close to equality the district
# populations are. "TOT_POP" is the population column from our shapefile.
my_updaters = {"population": updaters.Tally("ALL_TOT20", alias="population")}

# Election updaters, for computing election results using the vote totals
# from our shapefile.
elections = [
    Election("PRES20", {"Republican": "PRES20R", "Democratic": "PRES20D"}),
]

election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

initial_partition = GeographicPartition(graph, 
                                        assignment= "CD", updaters=my_updaters)


print("gerrychain calculates mean median to be ", metrics.partisan.mean_median(initial_partition[elections[0].name]))

print("gerrychain calculates EG to be ", metrics.partisan.efficiency_gap(initial_partition[elections[0].name]))





    