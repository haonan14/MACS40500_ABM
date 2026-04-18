from pathlib import Path

import numpy as np

import mesa
from agents import SugarAgent
## Using discrete cell space for this model that enforces von Neumann neighborhoods
from mesa.discrete_space import OrthogonalVonNeumannGrid
from mesa.discrete_space.property_layer import PropertyLayer

class SugarScapeModel(mesa.Model):
    ## Helper function to calculate Gini coefficient, used in plot
    def calc_gini(self):
        agent_sugars = [a.sugar for a in self.agents]
        if len(agent_sugars) == 0 or sum(agent_sugars) == 0:
            return 0
        sorted_sugars = sorted(agent_sugars)
        n = len(sorted_sugars)
        x = sum(el * (n - ind) for ind, el in enumerate(sorted_sugars)) / (n * sum(sorted_sugars))
        return 1 + (1 / n) - 2 * x
    
    #Mean habitus tracks the overall disposition climate
    def mean_habitus(self):
        if len(self.agents) == 0:
            return 0
        return sum(a.habitus for a in self.agents) / len(self.agents)
    
    #Habitus variance tracks behavioral polarization (a high variance means agents have sorted into distinct dispositional "classes")
    def habitus_variance(self):
        if len(self.agents) < 2:
            return 0
        habs = [a.habitus for a in self.agents]
        mean = sum(habs) / len(habs)
        return sum((h - mean) ** 2 for h in habs) / len(habs)
    
    #The habitus-sugar correlation is the signal for cumulative advantage / reproduction of inequality
    def habitus_sugar_correlation(self):
        if len(self.agents) < 2:
            return 0
        habs = [a.habitus for a in self.agents]
        sugars = [a.sugar for a in self.agents]
        mean_h = sum(habs) / len(habs)
        mean_s = sum(sugars) / len(sugars)
        num = sum((habs[i] - mean_h) * (sugars[i] - mean_s) for i in range(len(habs)))
        den_h = (sum((h - mean_h) ** 2 for h in habs)) ** 0.5
        den_s = (sum((s - mean_s) ** 2 for s in sugars)) ** 0.5
        if den_h == 0 or den_s == 0:
            return 0
        return num / (den_h * den_s)

    ## Define initiation, inherit seed property from parent class
    def __init__(
        self,
        width = 50,
        height = 50,
        initial_population=200,
        endowment_min=25,
        endowment_max=50,
        metabolism_min=1,
        metabolism_max=5,
        vision_min=1,
        vision_max=5,
        memory_len=10,
        habitus_update_rate=0.1,
        seed = None
    ):
        super().__init__(rng=seed)
        ## Instantiate model parameters
        self.width = width
        self.height = height
        self.memory_len = memory_len
        self.habitus_update_rate = habitus_update_rate
        self.cumulative_deaths = 0
        ## Set model to run continuously
        self.running = True
        ## Create grid
        self.grid = OrthogonalVonNeumannGrid(
            (self.width, self.height), torus=False, random=self.random
        )
        ## Define datacollector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Gini": self.calc_gini,
                "Population": lambda m: len(m.agents),
                "Mean Habitus": self.mean_habitus,
                "Habitus Variance": self.habitus_variance,
                "Habitus-Sugar Corr": self.habitus_sugar_correlation,
                "Cumulative Deaths": lambda m: m.cumulative_deaths,
            },
        )
        ## Import sugar distribution from raster, define grid property
        self.sugar_distribution = np.genfromtxt(Path(__file__).parent / "sugar-map.txt")
        self.grid.add_property_layer(
            PropertyLayer.from_data("sugar", self.sugar_distribution)
        )

        ## Create agents, give them random properties, and place them randomly on the map
        SugarAgent.create_agents(
            self,
            initial_population,
            self.random.choices(self.grid.all_cells.cells, k=initial_population),
            sugar=self.rng.integers(
                endowment_min, endowment_max, (initial_population,), endpoint=True
            ),
            metabolism=self.rng.integers(
                metabolism_min, metabolism_max, (initial_population,), endpoint=True
            ),
            vision=self.rng.integers(
                vision_min, vision_max, (initial_population,), endpoint=True
            ),
            memory_len=memory_len,
            habitus_update_rate=habitus_update_rate,
        )
        ## Initialize datacollector
        self.datacollector.collect(self)
    ## Define step: Sugar grows back at constant rate of 1, all agents move, then all agents consume, then all see if they die. Then model calculated Gini coefficient.
    def step(self):
        self.grid.sugar.data = np.minimum(
            self.grid.sugar.data + 1, self.sugar_distribution
        )
        self.agents.shuffle_do("move")
        self.agents.shuffle_do("gather_and_eat")
        self.agents.shuffle_do("update_habitus")
        ## Record population before death and count deaths
        pop_before = len(self.agents)
        self.agents.shuffle_do("see_if_die")
        self.cumulative_deaths += pop_before - len(self.agents)
        self.datacollector.collect(self)
    
