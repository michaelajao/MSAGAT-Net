import os
import sys
import subprocess

# Define the configurations to run
# Each configuration is a dictionary with dataset, sim_mat, and a list of horizons
configurations = [
    {
        "dataset": "japan",
        "sim_mat": "japan-adj",
        "horizons": [3, 5, 10, 15]
    },
    {
        "dataset": "australia-covid",
        "sim_mat": "australia-adj",
        "horizons": [3, 7, 14]
    },
    {
        "dataset": "spain-covid",
        "sim_mat": "spain-adj",
        "horizons": [3, 7, 14]
    },
    {
        "dataset": "state360",
        "sim_mat": "state-adj-49",
        "horizons": [3, 7, 14]
    }
]

# Common arguments for the optimization script
common_args = {
    "trials": 50,
    "epochs": 1500,
    "pruner": "hyperband",
    "parallel": 1, # Set to > 1 for parallel execution if your machine supports it
    "gpu": 0
}

def run_optimization():
    """
    Runs the optimization script for all defined configurations.
    """
    # Get the path to the optimization script
    script_path = os.path.join(os.path.dirname(__file__), "optimize.py")

    for config in configurations:
        dataset = config["dataset"]
        sim_mat = config["sim_mat"]
        for horizon in config["horizons"]:
            print(f"===== Running optimization for dataset: {dataset}, horizon: {horizon} =====")
            
            # Construct the command
            study_name = f"{dataset}_h{horizon}_msagat_t{common_args['trials']}_hb"
            
            command = [
                sys.executable,
                script_path,
                "--dataset", dataset,
                "--sim-mat", sim_mat,
                "--horizon", str(horizon),
                "--study-name", study_name,
                "--trials", str(common_args["trials"]),
                "--epochs", str(common_args["epochs"]),
                "--pruner", common_args["pruner"],
                "--parallel", str(common_args["parallel"]),
                "--gpu", str(common_args["gpu"])
            ]
            
            try:
                # Run the command
                subprocess.run(command, check=True)
                print(f"===== Finished optimization for dataset: {dataset}, horizon: {horizon} =====")
            except subprocess.CalledProcessError as e:
                print(f"!!!!! Error running optimization for dataset: {dataset}, horizon: {horizon} !!!!!")
                print(e)
                print("Continuing with the next run...")
            except FileNotFoundError:
                print(f"Error: The script at {script_path} was not found.")
                return

if __name__ == "__main__":
    run_optimization()
