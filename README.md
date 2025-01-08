# robust_resource_sharing
Code used to conduct the simulations in the paper entitled "Robust Resource Sharing in Network Slicing via Hypothesis Testing"

1) In the same folder of the git repository create a folder named "data"
2) Download the .csv files from the [IMDEA Networks dataset](https://git2.networks.imdea.org/wng/madrid-lte-dataset/-/tree/main/dataset?ref_type=heads) and place them in the "data" folder
3) Download the corresponding .parquet raw dataset files from the [IMDEA Networks box](https://box.networks.imdea.org/s/wxiZamiEXA5aVGx) and also place them in the "data" folder
4) Go to the git repository
5) Create a new test scenario parameter file to select which entries of the dataset correspond to a network slice and how many network slices are considered OR use one of the existing files (parameters_ts1.py, parameters_ts2.py, etc...)
6) Configure parameters.py to point at the desired parameters file
7) Sequentially run python scripts starting from a_load_csv_data.py to g_plot.py (do not run scripts that end with "alternative.py" or "original.py")
