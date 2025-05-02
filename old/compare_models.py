import os
import subprocess
import argparse
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "comparative")
FIGURES_DIR = os.path.join(BASE_DIR, "report", "figures", "comparative")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Datasets for comparison
DATASETS = [
    {
        "name": "japan",
        "sim_mat": "japan-adj",
        "description": "Japan COVID-19 dataset (47 prefectures)",
    },
    {
        "name": "region785",
        "sim_mat": "region-adj",
        "description": "US region dataset (785 regions)",
    },
]

# Forecast horizons to test
HORIZONS = [3, 5, 10]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare original MSTAGAT-Net with LocationAwareMSAGAT-Net"
    )
    parser.add_argument("--window", type=int, default=20, help="Input window size")
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs for quick comparison",
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="Early stopping patience"
    )
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--save_dir", type=str, default="save", help="Directory to save models"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--run_comparisons", action="store_true", help="Run the model comparison tests"
    )
    parser.add_argument(
        "--analyze_results", action="store_true", help="Analyze and visualize results"
    )
    return parser.parse_args()


def run_model(
    dataset,
    horizon,
    window,
    model_type,
    epochs,
    patience,
    batch,
    save_dir,
    gpu,
    seed,
    use_adjacency=False,
):
    """
    Run a model training with specified configuration
    """
    cmd = [
        "python",
        "src/train.py",
        f"--dataset={dataset['name']}",
        f"--sim_mat={dataset['sim_mat']}",
        f"--window={window}",
        f"--horizon={horizon}",
        f"--train=0.5",
        f"--val=0.2",
        f"--test=0.3",
        f"--epochs={epochs}",
        f"--batch={batch}",
        f"--patience={patience}",
        f"--save_dir={save_dir}",
        f"--gpu={gpu}",
        f"--seed={seed}",
        f"--model={model_type}",
        "--cuda",
    ]

    # Add adjacency matrix flag if needed
    if use_adjacency and model_type == "location_aware":
        cmd.append("--use_adjacency")

    # For dynamic models, don't set fixed hyperparameters
    if model_type == "location_aware":
        # Allow the model to dynamically set parameters based on dataset
        print(f"Running location-aware model with dynamic parameters...")
    else:
        # For original model, set the optimal hyperparameters for each dataset
        if dataset["name"] == "japan":
            cmd.extend(
                [
                    "--hidden_dim=16",
                    "--attention_heads=4",
                    "--attention_regularization_weight=3.15e-4",
                    "--num_scales=4",
                    "--kernel_size=5",
                    "--feature_channels=12",
                    "--bottleneck_dim=8",
                    "--dropout=0.318",
                    "--lr=0.001893",
                    "--weight_decay=6.72e-5",
                ]
            )
        elif dataset["name"] == "region785":
            cmd.extend(
                [
                    "--hidden_dim=32",
                    "--attention_heads=8",
                    "--attention_regularization_weight=3.96e-5",
                    "--num_scales=5",
                    "--kernel_size=9",
                    "--feature_channels=8",
                    "--bottleneck_dim=12",
                    "--dropout=0.223",
                    "--lr=0.004865",
                    "--weight_decay=1.0e-5",
                ]
            )

    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate()

    # Parse the output to extract metrics
    metrics = extract_metrics_from_output(stdout)
    metrics["dataset"] = dataset["name"]
    metrics["horizon"] = horizon
    metrics["model_type"] = model_type

    return metrics


def extract_metrics_from_output(output):
    """Extract metrics from the training script output"""
    metrics = {}

    # Look for final test metrics line
    for line in output.split("\n"):
        if line.startswith("Final TEST MAE"):
            # Parse metrics from line like:
            # Final TEST MAE 12.3456 std 1.2345 RMSE 23.4567 RMSEs 34.5678 PCC 0.9876 PCCs 0.8765 ...
            parts = line.split()
            metrics["mae"] = float(parts[3])
            metrics["std_mae"] = float(parts[5])
            metrics["rmse"] = float(parts[7])
            metrics["rmse_states"] = float(parts[9])
            metrics["pcc"] = float(parts[11])
            metrics["pcc_states"] = float(parts[13])
            metrics["mape"] = float(parts[15])
            metrics["r2"] = float(parts[17])
            metrics["r2_states"] = float(parts[19])
            metrics["var"] = float(parts[21])
            metrics["var_states"] = float(parts[23])
            metrics["peak_mae"] = float(parts[25])
            break

    # Look for parameter count
    for line in output.split("\n"):
        if line.startswith("#params:"):
            metrics["params"] = int(line.split(":")[1].strip())
            break

    # Look for best epoch
    for line in output.split("\n"):
        if line.startswith("Best validation epoch:"):
            metrics["best_epoch"] = int(line.split(":")[1].strip().split()[0])
            break

    return metrics


def run_comparisons(args):
    """Run comparison tests between original and location-aware models"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(RESULTS_DIR, f"comparison_results_{timestamp}.csv")

    # Initialize results file
    with open(results_path, "w", newline="") as csvfile:
        fieldnames = [
            "dataset",
            "horizon",
            "model_type",
            "mae",
            "rmse",
            "pcc",
            "rmse_states",
            "pcc_states",
            "r2",
            "r2_states",
            "params",
            "best_epoch",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Run comparisons for each dataset and horizon
        for dataset in DATASETS:
            print(f"\n========== Testing on {dataset['description']} ==========")

            for horizon in HORIZONS:
                print(f"\n--- Horizon {horizon} ---")

                # Run original model
                print(f"Running original MSTAGAT-Net...")
                msagat_metrics = run_model(
                    dataset,
                    horizon,
                    args.window,
                    "msagat",
                    args.epochs,
                    args.patience,
                    args.batch,
                    args.save_dir,
                    args.gpu,
                    args.seed,
                )

                # Write results to CSV
                writer.writerow(
                    {
                        "dataset": dataset["name"],
                        "horizon": horizon,
                        "model_type": "msagat",
                        "mae": msagat_metrics.get("mae", "N/A"),
                        "rmse": msagat_metrics.get("rmse", "N/A"),
                        "pcc": msagat_metrics.get("pcc", "N/A"),
                        "rmse_states": msagat_metrics.get("rmse_states", "N/A"),
                        "pcc_states": msagat_metrics.get("pcc_states", "N/A"),
                        "r2": msagat_metrics.get("r2", "N/A"),
                        "r2_states": msagat_metrics.get("r2_states", "N/A"),
                        "params": msagat_metrics.get("params", "N/A"),
                        "best_epoch": msagat_metrics.get("best_epoch", "N/A"),
                    }
                )

                # Run location-aware model without adjacency
                print(f"Running LocationAwareMSAGAT-Net without adjacency...")
                la_metrics = run_model(
                    dataset,
                    horizon,
                    args.window,
                    "location_aware",
                    args.epochs,
                    args.patience,
                    args.batch,
                    args.save_dir,
                    args.gpu,
                    args.seed,
                    use_adjacency=False,
                )

                # Write results to CSV
                writer.writerow(
                    {
                        "dataset": dataset["name"],
                        "horizon": horizon,
                        "model_type": "location_aware",
                        "mae": la_metrics.get("mae", "N/A"),
                        "rmse": la_metrics.get("rmse", "N/A"),
                        "pcc": la_metrics.get("pcc", "N/A"),
                        "rmse_states": la_metrics.get("rmse_states", "N/A"),
                        "pcc_states": la_metrics.get("pcc_states", "N/A"),
                        "r2": la_metrics.get("r2", "N/A"),
                        "r2_states": la_metrics.get("r2_states", "N/A"),
                        "params": la_metrics.get("params", "N/A"),
                        "best_epoch": la_metrics.get("best_epoch", "N/A"),
                    }
                )

                # Run location-aware model with adjacency
                print(f"Running LocationAwareMSAGAT-Net with adjacency...")
                la_adj_metrics = run_model(
                    dataset,
                    horizon,
                    args.window,
                    "location_aware",
                    args.epochs,
                    args.patience,
                    args.batch,
                    args.save_dir,
                    args.gpu,
                    args.seed,
                    use_adjacency=True,
                )

                # Write results to CSV
                writer.writerow(
                    {
                        "dataset": dataset["name"],
                        "horizon": horizon,
                        "model_type": "location_aware_adj",
                        "mae": la_adj_metrics.get("mae", "N/A"),
                        "rmse": la_adj_metrics.get("rmse", "N/A"),
                        "pcc": la_adj_metrics.get("pcc", "N/A"),
                        "rmse_states": la_adj_metrics.get("rmse_states", "N/A"),
                        "pcc_states": la_adj_metrics.get("pcc_states", "N/A"),
                        "r2": la_adj_metrics.get("r2", "N/A"),
                        "r2_states": la_adj_metrics.get("r2_states", "N/A"),
                        "params": la_adj_metrics.get("params", "N/A"),
                        "best_epoch": la_adj_metrics.get("best_epoch", "N/A"),
                    }
                )

                # Print comparison
                print("\nComparison Results:")
                print(
                    f"Original MSAGAT-Net - RMSE: {msagat_metrics.get('rmse', 'N/A')}, PCC: {msagat_metrics.get('pcc', 'N/A')}"
                )
                print(
                    f"LocationAwareMSAGAT-Net - RMSE: {la_metrics.get('rmse', 'N/A')}, PCC: {la_metrics.get('pcc', 'N/A')}"
                )
                print(
                    f"LocationAwareMSAGAT-Net with Adj - RMSE: {la_adj_metrics.get('rmse', 'N/A')}, PCC: {la_adj_metrics.get('pcc', 'N/A')}"
                )

                # Show improvement percentage
                if isinstance(msagat_metrics.get("rmse"), float) and isinstance(
                    la_adj_metrics.get("rmse"), float
                ):
                    rmse_improvement = (
                        (msagat_metrics["rmse"] - la_adj_metrics["rmse"])
                        / msagat_metrics["rmse"]
                        * 100
                    )
                    print(f"RMSE Improvement: {rmse_improvement:.2f}%")

                if isinstance(msagat_metrics.get("pcc"), float) and isinstance(
                    la_adj_metrics.get("pcc"), float
                ):
                    pcc_improvement = (
                        (la_adj_metrics["pcc"] - msagat_metrics["pcc"])
                        / msagat_metrics["pcc"]
                        * 100
                    )
                    print(f"PCC Improvement: {pcc_improvement:.2f}%")

    print(f"\nResults saved to {results_path}")
    return results_path


def analyze_results(results_path):
    """Analyze and visualize comparison results"""
    if not os.path.exists(results_path):
        print(f"Results file {results_path} not found.")
        return

    # Load results
    df = pd.read_csv(results_path)

    # Create nice model type labels
    model_type_mapping = {
        "msagat": "MSTAGAT-Net",
        "location_aware": "LocationAware-Net",
        "location_aware_adj": "LocationAware-Net+Adj",
    }
    df["model_label"] = df["model_type"].map(model_type_mapping)

    # Add dataset labels
    dataset_mapping = {
        "japan": "Japan (47 nodes)",
        "region785": "US Regions (785 nodes)",
    }
    df["dataset_label"] = df["dataset"].map(dataset_mapping)

    # Set seaborn style
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 10))

    # Plot RMSE by dataset, horizon, and model type
    plt.subplot(2, 2, 1)
    rmse_plot = sns.barplot(x="horizon", y="rmse", hue="model_label", data=df)
    plt.title("RMSE by Horizon and Model Type", fontsize=14)
    plt.xlabel("Prediction Horizon", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.legend(title="Model Type")

    # Plot PCC by dataset, horizon, and model type
    plt.subplot(2, 2, 2)
    pcc_plot = sns.barplot(x="horizon", y="pcc", hue="model_label", data=df)
    plt.title("PCC by Horizon and Model Type", fontsize=14)
    plt.xlabel("Prediction Horizon", fontsize=12)
    plt.ylabel("Pearson Correlation Coefficient", fontsize=12)
    plt.legend(title="Model Type")

    # Plot RMSE by dataset and model type
    plt.subplot(2, 2, 3)
    ds_rmse_plot = sns.barplot(x="dataset_label", y="rmse", hue="model_label", data=df)
    plt.title("RMSE by Dataset and Model Type", fontsize=14)
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.legend(title="Model Type")

    # Plot parameter count by dataset and model type
    plt.subplot(2, 2, 4)
    param_plot = sns.barplot(x="dataset_label", y="params", hue="model_label", data=df)
    plt.title("Parameter Count by Dataset and Model Type", fontsize=14)
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Number of Parameters", fontsize=12)
    plt.legend(title="Model Type")

    # Adjust layout and save
    plt.tight_layout()
    fig_path = os.path.join(
        FIGURES_DIR, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Comparison figure saved to {fig_path}")

    # Calculate improvement statistics
    print("\nImprovement Statistics:")

    # Group by dataset and horizon
    for ds in df["dataset"].unique():
        for horizon in df["horizon"].unique():
            subset = df[(df["dataset"] == ds) & (df["horizon"] == horizon)]

            if len(subset) >= 2:
                msagat_metrics = subset[subset["model_type"] == "msagat"].iloc[0]
                la_adj_metrics = (
                    subset[subset["model_type"] == "location_aware_adj"].iloc[0]
                    if "location_aware_adj" in subset["model_type"].values
                    else None
                )

                if la_adj_metrics is not None:
                    rmse_improvement = (
                        (msagat_metrics["rmse"] - la_adj_metrics["rmse"])
                        / msagat_metrics["rmse"]
                        * 100
                    )
                    pcc_improvement = (
                        (la_adj_metrics["pcc"] - msagat_metrics["pcc"])
                        / msagat_metrics["pcc"]
                        * 100
                    )

                    print(
                        f"\nDataset: {dataset_mapping.get(ds, ds)}, Horizon: {horizon}"
                    )
                    print(f"RMSE Improvement: {rmse_improvement:.2f}%")
                    print(f"PCC Improvement: {pcc_improvement:.2f}%")

                    # Parameter differences
                    param_change = (
                        (la_adj_metrics["params"] - msagat_metrics["params"])
                        / msagat_metrics["params"]
                    ) * 100
                    print(f"Parameter Change: {param_change:.2f}%")


def main():
    args = parse_args()

    if args.run_comparisons:
        results_path = run_comparisons(args)
    else:
        # Find the most recent results file if not running comparisons
        results_files = [
            f for f in os.listdir(RESULTS_DIR) if f.startswith("comparison_results_")
        ]
        results_files.sort(reverse=True)
        if results_files:
            results_path = os.path.join(RESULTS_DIR, results_files[0])
        else:
            print("No results file found. Run with --run_comparisons first.")
            return

    if args.analyze_results:
        analyze_results(results_path)


if __name__ == "__main__":
    main()
