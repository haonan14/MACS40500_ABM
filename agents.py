import math
## Using experimental agent type with native "cell" property that saves its current position in cellular grid
from mesa.discrete_space import CellAgent

## Helper function to get distance between two cells
def get_distance(cell_1, cell_2):
    x1, y1 = cell_1.coordinate
    x2, y2 = cell_2.coordinate
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx**2 + dy**2)

class SugarAgent(CellAgent):
    ## Initiate agent, inherit model property from parent class
    def __init__(self, model, cell, sugar=0, metabolism=0, vision = 0, memory_len = 0, habitus_update_rate = 0.1):
        super().__init__(model)
        ## Set variable traits based on model parameters
        self.cell = cell
        self.sugar = sugar
        self.metabolism = metabolism
        self.vision = vision
        
        # Here I introduced a new paramter which is habitus, as a continuous variable in [0, 1] 
        # that modulates how the agent weighs sugar-gain versus travel-distance when moving:
        # as moving to  1, it signals the agent is more exploratory that ignores distance and seeks maximum sugar
        # in contrast, when moving to 0, it signals the agent is more conservative, that minimizes distance and tolerates low sugar in nearby area
        self.habitus = 0.5
        
        # habitus is a "sedimented history" of past experience, so I introduced a sugar_history as a list to store the sugar history.
        self.sugar_history = []
        self.memory_len = memory_len
        
        #Speed at which habitus drifts toward the target implied by recent history. 
        #Small values produce strong hysteresis (habitus lags the environment)
        self.habitus_update_rate = habitus_update_rate
        
    ## Define movement action
    def move(self):
        ## Determine currently empty cells within line of sight
        possibles = [
            cell
            for cell in self.cell.get_neighborhood(self.vision, include_center=True)
            if cell.is_empty 
        ]
        if not possibles:
            return
        
        # compute habitus-weighted utility for every visible empty cell
        utilities = [
            self.habitus * cell.sugar - (1 - self.habitus) * get_distance(self.cell, cell)
            for cell in possibles
        ]
 
        ## Identify cell(s) tied for maximum utility
        max_utility = max(utilities)
        final_candidates = [
            possibles[i]
            for i in range(len(utilities))
            if math.isclose(utilities[i], max_utility, rel_tol=1e-02)
        ]
 
        ## Choose one of the best cells (randomly if more than one)
        self.cell = self.random.choice(final_candidates)
    # consumer sugar in current cell, depleting it, then consumer metabolism
    def gather_and_eat(self):
        self.sugar += self.cell.sugar
        self.cell.sugar = 0
        self.sugar -= self.metabolism
        
    def update_habitus(self):
        # Record current sugar, trim to memory window
        self.sugar_history.append(self.sugar)
        if len(self.sugar_history) > self.memory_len:
            self.sugar_history.pop(0)
 
        if len(self.sugar_history) < 2:
            return
 
        # Trajectory signal: per-step change in sugar over the memory, positive = accumulating, negative = losing
        trajectory = (self.sugar_history[-1] - self.sugar_history[0]) / len(self.sugar_history)
        
        # Map trajectory to target habitus through a sigmoid, so that target_habitus smoothly varies in [0, 1] around 0.5
        k = 1.0
        target_habitus = 1.0 / (1.0 + math.exp(-k * trajectory))
        # update the habitus
        self.habitus += self.habitus_update_rate * (target_habitus - self.habitus)
    ## If an agent has zero or negative suger, it dies and is removed from the model
    def see_if_die(self):
        if self.sugar <= 0:
            self.remove()
    
        
