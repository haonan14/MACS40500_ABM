from model import SugarScapeModel
from mesa.visualization import Slider, SolaraViz, make_plot_component
from mesa.visualization.components.matplotlib_components import make_mpl_space_component
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle

## Define agent portrayal (color, size, shape)
#Agents are colored on a red-to-blue gradient based on their habitus:red = risk-averse (habitus near 0), blue = exploratory (habitus near 1)
#purple in between, it is also the original state
def agent_portrayal(agent):
    ## Interpolate from red (risk-averse) to blue (exploratory)
    h = getattr(agent, "habitus", 0.5)
    h = max(0.0, min(1.0, h))
    
    r = int(255 * (1 - h))
    b = int(255 * h)
    color_hex = f"#{r:02x}00{b:02x}"
    return AgentPortrayalStyle(
        color=color_hex,
        marker="o",
        size=10,
    )


## Sugar cells rendered as yellower = more sugar
def propertylayer_portrayal(layer):
    return PropertyLayerStyle(
        color="yellow", alpha=0.8, colorbar=True, vmin=0, vmax=10
    )

## Define model space component
sugarscape_space = make_mpl_space_component(
    agent_portrayal=agent_portrayal,
    propertylayer_portrayal=propertylayer_portrayal,
    post_process=None,
    draw_grid=False,
)

GiniPlot = make_plot_component("Gini")
PopulationPlot = make_plot_component("Population")
MeanHabitusPlot = make_plot_component("Mean Habitus")
HabitusVariancePlot = make_plot_component("Habitus Variance")
HabitusSugarCorrPlot = make_plot_component("Habitus-Sugar Corr")

## Define variable model parameters
model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "width": 50,
    "height": 50,
    "initial_population": Slider(
        "Initial Population", value=200, min=50, max=500, step=10
    ),
    # Agent endowment parameters
    "endowment_min": Slider("Min Initial Endowment", value=25, min=5, max=30, step=1),
    "endowment_max": Slider("Max Initial Endowment", value=50, min=30, max=100, step=1),
    # Metabolism parameters
    "metabolism_min": Slider("Min Metabolism", value=1, min=1, max=3, step=1),
    "metabolism_max": Slider("Max Metabolism", value=5, min=3, max=8, step=1),
    # Vision parameters
    "vision_min": Slider("Min Vision", value=1, min=1, max=3, step=1),
    "vision_max": Slider("Max Vision", value=5, min=3, max=8, step=1),
    #memory length parameter
    "memory_len": Slider(
        "Memory Length (steps)", value=10, min=1, max=30, step=1
    ),
    #habitus_update_rate
    #   0.0 => habitus never updates (stays at initial 0.5 for all agents)
    #   1.0 => habitus reactively matches recent history each step
    "habitus_update_rate": Slider(
        "Habitus Update Rate", value=0.1, min=0.0, max=1.0, step=0.01
    ),
}

##Instantiate model
model1 = SugarScapeModel()

## Define all aspects of page
page = SolaraViz(
    model1,
    components=[
        sugarscape_space,
        GiniPlot,
        PopulationPlot,
        MeanHabitusPlot,
        HabitusVariancePlot,
        HabitusSugarCorrPlot,
    ],
    model_params=model_params,
    name="Sugarscape",
    play_interval=150,
)
## Return page
page
