import importlib


# Set Test Scenario to load the appropriate parameter file
test_scenario = 1

# Find the name of the parameter file
parameter_file = f"parameters_ts{test_scenario}"

# Load the parameter file
parameter_module = importlib.import_module(parameter_file)

# Inherit its global variables
for name, value in parameter_module.__dict__.items():
    if not name.startswith("__"):
        globals()[name] = value
