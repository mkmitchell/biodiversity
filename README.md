# biodiversity
Ducks Unlimited Incorporated and Ducks Unlimited Canada have been working on species distribution modeling
to calculate probability of species occurrence over a given area of interest.

We plan to follow the work already done by Dr. James Paterson (DUC) et al. - https://www.sciencedirect.com/science/article/pii/S0006320724003161
with a focus on different areas and species.

At present (Jan 2026) we're utilizing eBird for avian species and Global Biodiveristy Information Facility (GBIF) for mammals, reptiles, and amphibians.
Google Earth Engine and local processing were used to create the necessary data and run the models.

For model parameters we've creating area coverage percentages around species points based on date of observation.  We have cover percentages within
100m and 10km radiuses for all Google Dynamic World classes (water, trees, grass, flooded vegetation, crops, shrub and scrub, built, bare, snow and ice 
- https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1).  This is done with the geeDataFromPoints and GEEcreateEnvRaster 
notebooks.  We also calculate environmental means per season using Daymet v4 (https://developers.google.com/earth-engine/datasets/catalog/NASA_ORNL_DAYMET_V4) day 
length, precipitation, maximum air temperature and minimum air temperature within 10km.
Seasons:
	Winter: Dec, Jan, Feb
	Spring: Mar, Apr, May
	Summer: Jun, Jul, Aug
	Fall:   Sep, Oct, Nov

All parameters above are used to create maxent models using the elapid Python library for the following species:
    "Protonotaria citrea",
    "Limnothlypis swainsonii",
    "Setophaga americana",
    "Empidonax virescens",
    "Coccyzus americanus",
    "Vireo griseus",
    "Setophaga cerulea",
    "Hylocichla mustelina",
    "Parkesia motacilla",
    "Geothlypis formosa",
    "Archilochus colubris",
    "Elanoides forficatus",
    "Vireo flavifrons",
    "Buteo lineatus",
    "Setophaga dominica",
    "Setophaga citrina",
    "Dryocopus pileatus",
    "Meleagris gallopavo",
    "Sphyrapicus varius",
    "Odocoileus virginianus",     # White tailed deer
    "Ursus americanus",           # Black bear
    "Anaxyrus americanus",        # American Toad
    "Anaxyrus fowleri",           # Fowler's Toad
    "Gastrophryne carolinensis",  # Eastern Narrow-mouthed Toad
    "Hyla avivoca",               # Bird-voiced Treefrog
    "Hyla chrysoscelis",          # Cope's Gray Treefrog
    "Hyla cinerea",               # Green Treefrog
    "Hyla squirella",             # Squirrel Treefrog    
    "Hyla versicolor",            # Gray Treefrog
    "Lithobates catesbeianus",    # American Bullfrog
    "Lithobates clamitans",       # Bronze Frog
    "Lithobates palustris",       # Pickerel Frog
    "Lithobates sphenocephalus",  # Southern Leopard Frog
    "Pseudacris crucifer",        # Spring Peeper
    "Pseudacris fouquettei",      # Cajun Chorus Frog
    "Kinosternon subrubrum",      # Eastern Mud Turtle
    "Apalone spinifera",          # Spiny Softshell Turtle   
    "Macrochelys temmincki"       # Alligator Snapping Turtle 


To create "current" parameter values we've taken the mode of all classes within 2025 from Google Dynamic World and
since Daymet v4 does not have 2025 data yet we took the mean from 2017 - 2024.  We used the species models and created
distribution probabilities for each season using the 2025 landcover data and Daymet 2017-2024 means.