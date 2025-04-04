import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import pearsonr
from pathlib import Path

# Add utils for metric calculations
def calculate_efficiency_score(rmse, pcc, param_count, 
                              rmse_weight=0.65, 
                              pcc_weight=0.25, 
                              size_weight=0.1, 
                              size_threshold=100000):
    """Calculate an efficiency score balancing RMSE, PCC and model size."""
    normalized_pcc = 1.0 - pcc
    size_penalty = np.log10(max(param_count, 1000)) / np.log10(size_threshold)
    size_penalty = min(size_penalty, 1.0)
    
    score = (rmse * rmse_weight) + \
            (normalized_pcc * pcc_weight) + \
            (size_penalty * size_weight)
    
    return score

class OptimizationAnalyzer:
    """Class to analyze optimization results from multiple runs."""
    
    def __init__(self, results_path):
        """
        Initialize the analyzer with a path to optimization results.
        
        Args:
            results_path: Path to directory containing optimization results
        """
        self.results_path = Path(results_path)
        self.results_df = None
        self.history_data = {}
        self.load_data()
        
    def load_data(self):
        """Load optimization results data."""
        # Find all CSV files
        csv_files = list(self.results_path.glob("*_results.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No results CSV files found in {self.results_path}")
            
        # Load all results files into a single dataframe
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Add study name
            study_name = csv_file.stem.replace("_results", "")
            df['study'] = study_name
            dfs.append(df)
            
        self.results_df = pd.concat(dfs, ignore_index=True)
        
        # Clean up column names for Optuna output format
        # Convert all column names to strings
        self.results_df.columns = [str(col) for col in self.results_df.columns]
        
        # Find columns that hold user attributes - they'll be formatted as "user_attrs_{name}"
        user_attr_cols = [col for col in self.results_df.columns if col.startswith("user_attrs_")]
        param_cols = [col for col in self.results_df.columns if col.startswith("params_")]
        
        # Add a dictionary column for user attributes for compatibility
        if "user_attrs" not in self.results_df.columns:
            self.results_df['user_attrs'] = self.results_df.apply(
                lambda row: {
                    col.replace("user_attrs_", ""): row[col] 
                    for col in user_attr_cols if pd.notna(row[col])
                }, 
                axis=1
            )
        
        # Load trial history data
        history_files = list(self.results_path.glob("trial_*_history.json"))
        
        for history_file in history_files:
            trial_id = history_file.stem.split("_")[1]
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                    self.history_data[trial_id] = history_data
            except Exception as e:
                print(f"Error loading {history_file}: {e}")

        # Print summary of loaded data
        print(f"Loaded {len(self.results_df)} trials from {len(csv_files)} studies")
        print(f"Loaded {len(self.history_data)} trial history files")
        print(f"Found {len(user_attr_cols)} user attribute columns and {len(param_cols)} parameter columns")
    
    def get_best_trials(self, metric='value', n=5, ascending=True):
        """
        Get the top N trials based on a specific metric.
        
        Args:
            metric: Column name to sort by
            n: Number of trials to return
            ascending: If True, lower values are better (e.g., for RMSE or efficiency score)
        
        Returns:
            DataFrame of top N trials
        """
        # First check if metric is directly in dataframe
        if metric in self.results_df.columns:
            sorted_df = self.results_df.sort_values(metric, ascending=ascending)
            return sorted_df.head(n)
        
        # Check if metric is in user_attrs_ prefixed columns
        user_attr_metric = f"user_attrs_{metric}"
        if user_attr_metric in self.results_df.columns:
            sorted_df = self.results_df.sort_values(user_attr_metric, ascending=ascending)
            return sorted_df.head(n)
        
        # If using the user_attrs dictionary column
        try:
            # This will work if user_attrs is a dictionary column
            sorted_df = self.results_df.sort_values(
                by=lambda x: self.results_df['user_attrs'].apply(
                    lambda attrs: attrs.get(metric, float('inf') if ascending else float('-inf'))
                ),
                ascending=ascending
            )
            return sorted_df.head(n)
        except Exception as e:
            print(f"Error getting best trials: {e}")
            return self.results_df.head(n)  # Return first N as fallback
            
    def get_parameter_importance(self):
        """
        Calculate parameter importance based on correlation with performance metrics.
        
        Returns:
            DataFrame of parameter importance scores
        """
        # Extract parameters
        params = [c for c in self.results_df.columns if c.startswith('params_')]
        
        # Extract targets - use both direct columns and user_attrs columns
        target_candidates = ['best_rmse', 'best_pcc', 'value']
        targets = []
        target_columns = {}
        
        for target in target_candidates:
            # Check if direct column
            if target in self.results_df.columns:
                targets.append(target)
                target_columns[target] = target
            # Check if in user_attrs_
            elif f"user_attrs_{target}" in self.results_df.columns:
                targets.append(target)
                target_columns[target] = f"user_attrs_{target}"
        
        if not targets:
            print("No target metrics found for importance calculation")
            return pd.DataFrame(columns=['parameter', 'target', 'importance'])
        
        importance_data = []
        
        # For each parameter and target, calculate correlation
        for param in params:
            param_name = param.replace('params_', '')
            
            for target in targets:
                target_col = target_columns.get(target, target)
                
                # Get target values
                if target_col in self.results_df.columns:
                    target_values = self.results_df[target_col]
                else:
                    # Try extracting from user_attrs
                    try:
                        target_values = self.results_df['user_attrs'].apply(
                            lambda attrs: attrs.get(target, np.nan)
                        )
                    except:
                        print(f"Could not extract {target} values")
                        continue
                
                # Get parameter values
                param_values = self.results_df[param]
                
                # Skip if all values are NaN
                if target_values.isna().all() or param_values.isna().all():
                    continue
                
                # If parameter is categorical (low number of unique values), treat specially
                if len(param_values.unique()) < 10:
                    # For categorical parameters, use ANOVA or similar approach
                    # Here we'll use a simple approach: for each category, calculate the average performance
                    try:
                        grouped = pd.DataFrame({
                            'param': param_values,
                            'target': target_values
                        }).groupby('param').agg({'target': 'mean'})
                        
                        # Calculate how much variance is explained by the grouping
                        # Higher values mean the parameter has more impact
                        overall_variance = target_values.var()
                        if overall_variance == 0:
                            importance = 0
                        else:
                            # Calculate weighted variance of group means
                            group_means_var = grouped['target'].var()
                            importance = group_means_var / overall_variance
                    except:
                        # Fall back to correlation if grouping fails
                        try:
                            importance, _ = pearsonr(param_values, target_values)
                            importance = abs(importance)
                        except:
                            importance = 0
                else:
                    # For numerical parameters, use correlation
                    try:
                        importance, _ = pearsonr(param_values, target_values)
                        # Take absolute value for importance
                        importance = abs(importance)
                    except:
                        importance = 0
                
                importance_data.append({
                    'parameter': param_name,
                    'target': target,
                    'importance': importance
                })
        
        return pd.DataFrame(importance_data)
    
    def plot_parameter_importance(self, save_path=None):
        """
        Plot parameter importance chart.
        
        Args:
            save_path: Path to save the figure (if None, just display)
        """
        importance_df = self.get_parameter_importance()
        
        if importance_df.empty:
            print("No parameter importance data available to plot")
            return
        
        # Pivot for better visualization
        try:
            pivot_df = importance_df.pivot(index='parameter', columns='target', values='importance')
            
            # Sort by average importance
            pivot_df['avg_importance'] = pivot_df.mean(axis=1)
            pivot_df = pivot_df.sort_values('avg_importance', ascending=False)
            
            # Drop average column
            plot_df = pivot_df.drop('avg_importance', axis=1)
            
            # Plot
            plt.figure(figsize=(12, 8))
            sns.heatmap(plot_df, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Importance Score'})
            plt.title('Parameter Importance', fontsize=16)
            plt.ylabel('Parameter', fontsize=14)
            plt.xlabel('Target Metric', fontsize=14)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300)
                print(f"Parameter importance plot saved to {save_path}")
            else:
                plt.show()
        except Exception as e:
            print(f"Error creating parameter importance plot: {e}")
            
    def plot_parallel_coordinates(self, best_n=10, save_path=None):
        """
        Create parallel coordinates plot to visualize the relationships between
        hyperparameters and performance metrics.
        
        Args:
            best_n: Number of best trials to include
            save_path: Path to save the figure (if None, just display)
        """
        # Get best trials
        best_trials = self.get_best_trials(n=best_n)
        
        try:
            # Prepare data for parallel coordinates
            # Extract parameters
            params = [c for c in best_trials.columns if c.startswith('params_')]
            
            # Add performance metrics from user_attrs or direct columns
            plot_data = best_trials[params].copy()
            
            # Add performance metrics
            metrics_to_add = {
                'RMSE': 'best_rmse',
                'PCC': 'best_pcc',
                'Model Size': 'param_count'
            }
            
            for display_name, metric_name in metrics_to_add.items():
                # Try getting from user_attrs column first
                if f"user_attrs_{metric_name}" in best_trials.columns:
                    plot_data[display_name] = best_trials[f"user_attrs_{metric_name}"]
                # Then try from direct columns
                elif metric_name in best_trials.columns:
                    plot_data[display_name] = best_trials[metric_name]
                # Finally from user_attrs dictionary if it exists
                elif 'user_attrs' in best_trials.columns:
                    try:
                        plot_data[display_name] = best_trials['user_attrs'].apply(
                            lambda x: x.get(metric_name, np.nan)
                        )
                    except:
                        pass
            
            # Add efficiency score
            if 'value' in best_trials.columns:
                plot_data['Efficiency'] = best_trials['value']
            
            # Rename columns to remove 'params_' prefix
            plot_data.columns = [c.replace('params_', '') if c.startswith('params_') else c for c in plot_data.columns]
            
            # Drop any columns with all NaN values
            plot_data = plot_data.dropna(axis=1, how='all')
            
            # Scale the columns for better visualization
            scaled_data = plot_data.copy()
            for col in plot_data.columns:
                if plot_data[col].dtype in [np.float64, np.int64] and not plot_data[col].isna().all():
                    if plot_data[col].max() != plot_data[col].min():
                        scaled_data[col] = (plot_data[col] - plot_data[col].min()) / (plot_data[col].max() - plot_data[col].min())
            
            # Create a column for coloring, preferring 'Efficiency' if available
            color_col = 'Efficiency' if 'Efficiency' in scaled_data.columns else scaled_data.columns[-1]
            
            # Plot
            plt.figure(figsize=(15, 10))
            pd.plotting.parallel_coordinates(
                scaled_data, color_col, 
                colormap=plt.cm.coolwarm_r,
                linewidth=3,
                alpha=0.7
            )
            plt.title('Parallel Coordinates Plot of Top Configurations', fontsize=16)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.ylabel('Normalized Value', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300)
                print(f"Parallel coordinates plot saved to {save_path}")
            else:
                plt.show()
        except Exception as e:
            print(f"Error creating parallel coordinates plot: {e}")
    
    def plot_learning_curves(self, trial_ids=None, metric='rmse', smoothing=0, save_path=None):
        """
        Plot learning curves for selected trials.
        
        Args:
            trial_ids: List of trial IDs to plot (if None, plot best 5)
            metric: Which metric to plot ('rmse', 'pcc', etc.)
            smoothing: Window size for moving average smoothing
            save_path: Path to save the figure (if None, just display)
        """
        if not self.history_data:
            print("No trial history data available")
            return
            
        if trial_ids is None:
            # Get best trials
            best_trials = self.get_best_trials(n=5)
            trial_ids = [str(v) for v in best_trials['number'].values]
        
        plt.figure(figsize=(12, 8))
        curves_plotted = 0
        
        for trial_id in trial_ids:
            if trial_id not in self.history_data:
                print(f"No history data for trial {trial_id}")
                continue
                
            history = self.history_data[trial_id]
            
            if metric not in history:
                print(f"Metric {metric} not found in trial {trial_id} history")
                continue
                
            values = history[metric]
            
            # Apply smoothing if needed
            if smoothing > 0 and len(values) > smoothing:
                kernel = np.ones(smoothing) / smoothing
                values = np.convolve(values, kernel, mode='valid')
                epochs = range(1 + smoothing//2, len(values) + 1 + smoothing//2)
            else:
                epochs = range(1, len(values) + 1)
            
            label = f"Trial {trial_id} - Final: {values[-1]:.4f}"
            plt.plot(epochs, values, linewidth=2, label=label)
            curves_plotted += 1
        
        if curves_plotted == 0:
            print(f"No learning curves available for metric {metric}")
            return
            
        plt.title(f'Learning Curves - {metric.upper()}', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(metric.upper(), fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Learning curves plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self, output_dir=None):
        """
        Generate a comprehensive report of optimization results.
        
        Args:
            output_dir: Directory to save report figures (if None, use results_path/report)
        """
        if output_dir is None:
            output_dir = self.results_path / "report"
        else:
            output_dir = Path(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating optimization report in {output_dir}")
        
        # 1. Parameter importance
        param_importance_path = output_dir / "parameter_importance.png"
        self.plot_parameter_importance(save_path=param_importance_path)
        
        # 2. Parallel coordinates for best trials
        parallel_coords_path = output_dir / "parallel_coordinates.png"
        self.plot_parallel_coordinates(best_n=10, save_path=parallel_coords_path)
        
        # 3. Learning curves for best trials
        if self.history_data:
            rmse_curves_path = output_dir / "rmse_learning_curves.png"
            self.plot_learning_curves(metric='rmse', smoothing=5, save_path=rmse_curves_path)
            
            pcc_curves_path = output_dir / "pcc_learning_curves.png"
            self.plot_learning_curves(metric='pcc', smoothing=5, save_path=pcc_curves_path)
        
        # 4. Create a summary table of best trials
        best_trials = self.get_best_trials(n=10)
        best_trials_path = output_dir / "best_trials.csv"
        
        try:
            # Extract relevant columns
            # Start with number and value
            summary_cols = []
            if 'number' in best_trials.columns:
                summary_cols.append('number')
            if 'value' in best_trials.columns:
                summary_cols.append('value')
                
            # Add parameters
            param_cols = [c for c in best_trials.columns if c.startswith('params_')]
            summary_cols.extend(param_cols)
            
            # Create summary dataframe
            summary_df = best_trials[summary_cols].copy()
            
            # Add metrics either from user_attrs columns or user_attrs dictionary
            metrics_to_add = {
                'rmse': 'best_rmse',
                'pcc': 'best_pcc',
                'model_size': 'param_count',
                'test_rmse': 'test_rmse',
                'test_pcc': 'test_pcc'
            }
            
            for col_name, metric_name in metrics_to_add.items():
                # Try to get from user_attrs_ prefixed column
                if f"user_attrs_{metric_name}" in best_trials.columns:
                    summary_df[col_name] = best_trials[f"user_attrs_{metric_name}"]
                # Or from direct column
                elif metric_name in best_trials.columns:
                    summary_df[col_name] = best_trials[metric_name]
                # Or from user_attrs dictionary
                elif 'user_attrs' in best_trials.columns:
                    try:
                        summary_df[col_name] = best_trials['user_attrs'].apply(
                            lambda x: x.get(metric_name, np.nan)
                        )
                    except:
                        pass
            
            summary_df.to_csv(best_trials_path, index=False)
            print(f"Best trials summary saved to {best_trials_path}")
        except Exception as e:
            print(f"Error creating best trials summary: {e}")
        
        # Return summary of best configuration
        try:
            best_trial = self.get_best_trials(n=1).iloc[0]
            
            # Extract parameters
            params = {}
            for col in best_trial.index:
                if col.startswith('params_'):
                    param_name = col.replace('params_', '')
                    params[param_name] = best_trial[col]
            
            # Get metrics from user attributes
            metrics = {}
            attr_metrics = ['best_rmse', 'best_pcc', 'test_rmse', 'test_pcc', 'param_count']
            
            for metric in attr_metrics:
                # Try user_attrs_ prefixed column
                if f"user_attrs_{metric}" in best_trial.index:
                    metrics[metric] = best_trial[f"user_attrs_{metric}"]
                # Try direct column
                elif metric in best_trial.index:
                    metrics[metric] = best_trial[metric]
                # Try user_attrs dictionary
                elif 'user_attrs' in best_trial.index:
                    try:
                        metrics[metric] = best_trial['user_attrs'].get(metric, np.nan)
                    except:
                        metrics[metric] = np.nan
            
            # Create the best config dictionary
            best_config = {
                'trial_number': int(best_trial['number']) if 'number' in best_trial.index else 0,
                'efficiency_score': float(best_trial['value']) if 'value' in best_trial.index else np.nan,
                'rmse': float(metrics.get('best_rmse', np.nan)),
                'pcc': float(metrics.get('best_pcc', np.nan)),
                'test_rmse': float(metrics.get('test_rmse', np.nan)),
                'test_pcc': float(metrics.get('test_pcc', np.nan)),
                'param_count': int(metrics.get('param_count', 0)),
                'hyperparameters': params,
                '_analyzer': self  # Add reference to analyzer for comparison
            }
            
            return best_config
        except Exception as e:
            print(f"Error extracting best configuration: {e}")
            return None
        
def print_best_config(config):
    """Print a nicely formatted best configuration."""
    if config is None:
        print("No valid configuration found")
        return
        
    print("\n" + "="*80)
    print(f"BEST MODEL CONFIGURATION (Trial #{config['trial_number']})")
    print("="*80)
    
    print(f"\nPerformance Metrics:")
    print(f"  Validation RMSE: {config['rmse']:.4f}")
    print(f"  Validation PCC:  {config['pcc']:.4f}")
    print(f"  Test RMSE:       {config['test_rmse']:.4f}")
    print(f"  Test PCC:        {config['test_pcc']:.4f}")
    print(f"  Model Size:      {config['param_count']} parameters")
    print(f"  Efficiency:      {config['efficiency_score']:.4f}")
    
    print(f"\nHyperparameters:")
    for name, value in config['hyperparameters'].items():
        print(f"  {name}: {value}")
    
    # Print info about missing low_rank_dim if needed
    if 'low_rank_dim' not in config['hyperparameters'] or pd.isna(config['hyperparameters'].get('low_rank_dim')):
        print("\nNOTE: low_rank_dim was not optimized in this trial (using default value of 8)")
    
    print("\n" + "="*80)
    print("COMMAND FOR TRAINING BEST CONFIGURATION:")
    print("="*80)
    
    cmd = "python src/train.py"
    cmd += f" --dataset japan --sim_mat japan-adj --window 20 --horizon 5"
    for name, value in config['hyperparameters'].items():
        # Skip None or NaN values
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        cmd += f" --{name} {value}"
    
    # Add low_rank_dim with default if missing
    if 'low_rank_dim' not in config['hyperparameters'] or pd.isna(config['hyperparameters'].get('low_rank_dim')):
        cmd += " --low_rank_dim 8"  # Use the default value
        
    cmd += " --cuda"
    
    print(f"\n{cmd}\n")
    print("="*80 + "\n")

    # Add a section comparing with the most recent trial that uses low_rank_dim parameter
    print("COMPARISON WITH RECENT OPTIMIZATION INCLUDING low_rank_dim:")
    print("="*80)
    try:
        # Get the most recent optimization results
        analyzer = config.get('_analyzer')
        if analyzer and hasattr(analyzer, 'results_df'):
            # Find trials that have low_rank_dim parameter
            low_rank_trials = analyzer.results_df[analyzer.results_df['params_low_rank_dim'].notna()]
            
            if not low_rank_trials.empty:
                # Get the best low_rank_dim trial
                best_low_rank = low_rank_trials.sort_values('value').iloc[0]
                
                print(f"\nBest Trial With low_rank_dim Parameter (Trial #{int(best_low_rank['number'])}):")
                print(f"  Validation RMSE: {best_low_rank.get('user_attrs_best_rmse', 'N/A')}")
                print(f"  Validation PCC: {best_low_rank.get('user_attrs_best_pcc', 'N/A')}")
                print(f"  Test RMSE: {best_low_rank.get('user_attrs_test_rmse', 'N/A')}")
                print(f"  Test PCC: {best_low_rank.get('user_attrs_test_pcc', 'N/A')}")
                print(f"  Model Size: {best_low_rank.get('user_attrs_param_count', 'N/A')} parameters")
                print(f"  Efficiency: {best_low_rank.get('value', 'N/A')}")
                
                print("\n  Hyperparameters:")
                for col in [c for c in best_low_rank.index if c.startswith('params_')]:
                    print(f"    {col.replace('params_', '')}: {best_low_rank[col]}")
                
                # Print a recommendation
                if best_low_rank.get('value', float('inf')) > config['efficiency_score']:
                    print("\nRECOMMENDATION: The earlier optimization without explicit low_rank_dim tuning")
                    print("produced a better model. Consider keeping low_rank_dim=8 (default) for best results.")
                else:
                    print("\nRECOMMENDATION: The newer optimization with explicit low_rank_dim tuning")
                    print(f"produced a better model. Consider using low_rank_dim={int(best_low_rank.get('params_low_rank_dim', 8))}")
            else:
                print("\nNo trials with optimized low_rank_dim parameter found for comparison.")
        else:
            print("\nNo analyzer available for comparison.")
    except Exception as e:
        print(f"\nError comparing with recent optimizations: {str(e)}")
    
    print("="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze optimization results')
    parser.add_argument('--results-dir', type=str, default='optim_results',
                       help='Directory with optimization results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for report (default: results_dir/report)')
    args = parser.parse_args()
    
    # Ensure path is absolute
    if not os.path.isabs(args.results_dir):
        # Get repo root
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(repo_root, args.results_dir)
    else:
        results_dir = args.results_dir
    
    if args.output_dir is not None and not os.path.isabs(args.output_dir):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(repo_root, args.output_dir)
    else:
        output_dir = args.output_dir
    
    analyzer = OptimizationAnalyzer(results_dir)
    best_config = analyzer.generate_report(output_dir=output_dir)
    print_best_config(best_config)