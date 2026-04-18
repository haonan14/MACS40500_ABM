*Code adapted from Mesa Examples project*

# SugarScape with Bourdieusian Habitus

This is a modification of Epstein and Axtell's (1996) SugarScape model, extending it with a Bourdieusian habitus mechanism: agents develop an internalized disposition shaped by their trajectory of resource accumulation, which in turn shapes their future movement decisions.
In the original SugarScape, agents are pure sugar-maximizers: at every step, they move to the visible cell with the most sugar, breaking ties by distance. Inequality emerges from the interaction between agent heterogeneity (vision, metabolism, endowment) and the spatial distribution of sugar.
In this modified version, each agent additionally carries a habitus variable in [0, 1] that determines how it weighs sugar-gain versus travel-distance when moving:
utility(cell) = habitus * sugar(cell) - (1 - habitus) * distance(cell)

Habitus updates each step based on the agent's recent sugar trajectory (change over the memory window). Agents experiencing sustained accumulation drift toward exploration; agents experiencing sustained loss drift toward risk-aversion. 

# Research Question
Does an internalized disposition shaped by one's trajectory of resource accumulation reproduce initial inequalities, independently of structural differences in agents' fixed abilities (vision, metabolism)?

## How to Run

To run the model interactively once you have a complete agents file, run the following code in this directory:

```
    $ solara run app.py
```

## Files

* ``agents.py``: Contains the agent class, added habitus, sugar_history, memory_length, habitus_update_rate attributes. Replaced the movement rule with a habitus-weighted utility calculation. Added an update_habitus() method that updates habitus based on recent sugar trajectory.
* ``model.py``: Contains the model class, added memory_length and habitus_update_rate parameters. Added habitus-related data reporters (mean, variance, correlation with sugar, cumulative deaths). Added update_habitus to the step sequence
* ``app.py``: Defines classes for visualizing the model in the browser via Solara, and instantiates a visualization server. Added sliders for memory_length and habitus_update_rate. Agent color now maps habitus to a red-to-blue gradient. Added new plots for habitus dynamics and cumulative deaths.
* ``sugar-map.txt``: Raster file with the default sugar distibution for the model
