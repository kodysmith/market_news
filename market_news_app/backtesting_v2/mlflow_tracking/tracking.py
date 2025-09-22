#!/usr/bin/env python3
"""
MLflow Experiment Tracking

Tracks backtest experiments, parameters, metrics, and artifacts.
Provides experiment comparison and model registry functionality.
"""

import mlflow
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

class BacktestTracker:
    """MLflow-based experiment tracking for backtests"""

    def __init__(self, experiment_name: str = "quant_backtests", tracking_uri: str = None):
        """
        Initialize the backtest tracker

        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI (optional)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def start_run(self, run_name: str, tags: Dict[str, Any] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run

        Args:
            run_name: Name for the run
            tags: Additional tags for the run

        Returns:
            MLflow ActiveRun object
        """
        run = mlflow.start_run(run_name=run_name)

        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        return run

    def log_backtest_params(self, params: Dict[str, Any]):
        """
        Log backtest parameters

        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)
            elif isinstance(value, (list, dict)):
                mlflow.log_param(key, json.dumps(value))
            else:
                mlflow.log_param(key, str(value))

    def log_backtest_metrics(self, metrics: Dict[str, float]):
        """
        Log backtest performance metrics

        Args:
            metrics: Dictionary of metrics
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                mlflow.log_metric(key, value)

    def log_portfolio_results(self, portfolio_results: Any, framework: str = "vectorbt"):
        """
        Log portfolio results based on framework

        Args:
            portfolio_results: Portfolio object (VectorBT or Backtrader)
            framework: Framework used ('vectorbt' or 'backtrader')
        """
        try:
            if framework == "vectorbt":
                self._log_vectorbt_results(portfolio_results)
            elif framework == "backtrader":
                self._log_backtrader_results(portfolio_results)
            else:
                print(f"Unsupported framework: {framework}")
        except Exception as e:
            print(f"Failed to log portfolio results: {e}")

    def _log_vectorbt_results(self, portfolio):
        """Log VectorBT portfolio results"""
        try:
            # Basic metrics
            metrics = {
                'total_return': float(portfolio.total_return()),
                'sharpe_ratio': float(portfolio.sharpe_ratio()),
                'max_drawdown': float(portfolio.max_drawdown()),
                'win_rate': float(portfolio.trades.win_rate()),
                'total_trades': len(portfolio.trades),
                'avg_trade_return': float(portfolio.trades.pnl.mean()),
            }

            # Filter out NaN values
            metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}

            self.log_backtest_metrics(metrics)

            # Log equity curve as artifact
            equity_df = pd.DataFrame({
                'equity': portfolio.value().values,
                'date': portfolio.value().index
            })

            equity_path = "equity_curve.csv"
            equity_df.to_csv(equity_path, index=False)
            mlflow.log_artifact(equity_path)

            # Log trades
            if len(portfolio.trades) > 0:
                trades_df = portfolio.trades.records_readable
                trades_path = "trades.csv"
                trades_df.to_csv(trades_path, index=False)
                mlflow.log_artifact(trades_path)

        except Exception as e:
            print(f"Failed to log VectorBT results: {e}")

    def _log_backtrader_results(self, strategy_result):
        """Log Backtrader strategy results"""
        try:
            # Extract analyzer results
            returns_analysis = strategy_result.analyzers.returns.get_analysis()
            drawdown_analysis = strategy_result.analyzers.drawdown.get_analysis()
            sharpe_analysis = strategy_result.analyzers.sharpe.get_analysis()
            trades_analysis = strategy_result.analyzers.trades.get_analysis()

            metrics = {
                'total_return': float(returns_analysis.get('rtot', 0)),
                'sharpe_ratio': float(sharpe_analysis.get('sharperatio', 0)),
                'max_drawdown': float(drawdown_analysis.get('max', {}).get('drawdown', 0)),
                'total_trades': trades_analysis.get('total', {}).get('total', 0),
                'winning_trades': trades_analysis.get('won', {}).get('total', 0),
                'losing_trades': trades_analysis.get('lost', {}).get('total', 0),
            }

            # Calculate win rate
            total_trades = metrics['total_trades']
            if total_trades > 0:
                metrics['win_rate'] = metrics['winning_trades'] / total_trades

            # Filter out invalid metrics
            metrics = {k: v for k, v in metrics.items()
                      if isinstance(v, (int, float)) and not np.isnan(v)}

            self.log_backtest_metrics(metrics)

            # Log equity curve if available
            if hasattr(strategy_result.analyzers, 'pyfolio'):
                try:
                    pyfolio_analysis = strategy_result.analyzers.pyfolio.get_analysis()
                    returns_df = pd.DataFrame({
                        'returns': pyfolio_analysis['returns']
                    })
                    returns_path = "returns.csv"
                    returns_df.to_csv(returns_path)
                    mlflow.log_artifact(returns_path)
                except:
                    pass

        except Exception as e:
            print(f"Failed to log Backtrader results: {e}")

    def log_feature_importance(self, feature_names: List[str], importance_scores: List[float]):
        """
        Log feature importance scores

        Args:
            feature_names: List of feature names
            importance_scores: List of importance scores
        """
        try:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)

            importance_path = "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)

            # Log top features as parameters
            top_features = importance_df.head(10)
            for i, (_, row) in enumerate(top_features.iterrows()):
                mlflow.log_param(f"top_feature_{i+1}", f"{row['feature']}: {row['importance']:.4f}")

        except Exception as e:
            print(f"Failed to log feature importance: {e}")

    def log_model_artifact(self, model, model_name: str = "backtest_model"):
        """
        Log machine learning model

        Args:
            model: Trained model object
            model_name: Name for the model
        """
        try:
            # Try sklearn logging first
            try:
                import sklearn
                mlflow.sklearn.log_model(model, model_name)
            except ImportError:
                # Fallback: log as pickle artifact
                import pickle
                with open(f"{model_name}.pkl", "wb") as f:
                    pickle.dump(model, f)
                mlflow.log_artifact(f"{model_name}.pkl")
                print(f"Logged model as pickle artifact (sklearn not available)")
        except Exception as e:
            print(f"Failed to log model: {e}")

    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()

class ExperimentComparator:
    """Compare and analyze multiple backtest experiments"""

    def __init__(self, experiment_name: str = "quant_backtests", tracking_uri: str = None):
        """
        Initialize experiment comparator

        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.experiment_name = experiment_name

    def get_experiment_runs(self, n_runs: int = 50) -> pd.DataFrame:
        """
        Get recent experiment runs

        Args:
            n_runs: Number of recent runs to retrieve

        Returns:
            DataFrame with run information
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                print(f"Experiment '{self.experiment_name}' not found")
                return pd.DataFrame()

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=n_runs
            )

            return runs
        except Exception as e:
            print(f"Failed to get experiment runs: {e}")
            return pd.DataFrame()

    def compare_strategies(self, metric: str = "total_return",
                          group_by: str = "params.strategy") -> pd.DataFrame:
        """
        Compare strategies by performance metric

        Args:
            metric: Metric to compare
            group_by: Parameter to group by

        Returns:
            DataFrame with strategy comparison
        """
        runs_df = self.get_experiment_runs(200)  # Get more runs for comparison

        if runs_df.empty:
            return pd.DataFrame()

        # Extract strategy parameter
        if group_by == "params.strategy":
            runs_df['strategy'] = runs_df['params.strategy']
        else:
            runs_df['strategy'] = runs_df['tags'].apply(
                lambda x: x.get('strategy', 'unknown') if isinstance(x, dict) else 'unknown'
            )

        # Group by strategy and calculate statistics
        comparison = runs_df.groupby('strategy').agg({
            f'metrics.{metric}': ['mean', 'std', 'count', 'min', 'max']
        }).round(4)

        # Flatten column names
        comparison.columns = [f'{metric}_{stat}' for stat in ['mean', 'std', 'count', 'min', 'max']]
        comparison = comparison.sort_values(f'{metric}_mean', ascending=False)

        return comparison

    def find_best_hyperparameters(self, strategy: str, metric: str = "total_return") -> Dict[str, Any]:
        """
        Find best hyperparameters for a strategy

        Args:
            strategy: Strategy name
            metric: Metric to optimize

        Returns:
            Dictionary with best parameters and score
        """
        runs_df = self.get_experiment_runs(500)

        if runs_df.empty:
            return {}

        # Filter by strategy
        strategy_runs = runs_df[runs_df['params.strategy'] == strategy]

        if strategy_runs.empty:
            return {}

        # Find best run
        best_run = strategy_runs.loc[strategy_runs[f'metrics.{metric}'].idxmax()]

        best_params = {}
        for col in best_run.index:
            if col.startswith('params.'):
                param_name = col.replace('params.', '')
                best_params[param_name] = best_run[col]

        best_params['best_score'] = best_run[f'metrics.{metric}']
        best_params['run_id'] = best_run['run_id']

        return best_params

    def generate_performance_report(self, output_path: str = "performance_report.html"):
        """
        Generate comprehensive performance report

        Args:
            output_path: Path to save the report
        """
        runs_df = self.get_experiment_runs(100)

        if runs_df.empty:
            print("No runs found for report generation")
            return

        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quant Backtest Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric-positive {{ color: green; }}
                .metric-negative {{ color: red; }}
                h1, h2 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>Quant Backtest Performance Report</h1>
            <p>Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Strategy Performance Comparison</h2>
        """

        # Strategy comparison table
        comparison = self.compare_strategies()
        if not comparison.empty:
            html_content += """
            <table>
                <tr>
                    <th>Strategy</th>
                    <th>Avg Return</th>
                    <th>Std Dev</th>
                    <th>Sample Size</th>
                    <th>Best Return</th>
                    <th>Worst Return</th>
                </tr>
            """

            for strategy, row in comparison.iterrows():
                avg_return = row['total_return_mean']
                color_class = "metric-positive" if avg_return > 0 else "metric-negative"

                html_content += f"""
                <tr>
                    <td>{strategy}</td>
                    <td class="{color_class}">{avg_return:.2%}</td>
                    <td>{row['total_return_std']:.2%}</td>
                    <td>{int(row['total_return_count'])}</td>
                    <td class="metric-positive">{row['total_return_max']:.2%}</td>
                    <td class="metric-negative">{row['total_return_min']:.2%}</td>
                </tr>
                """

            html_content += "</table>"

        # Recent runs table
        html_content += "<h2>Recent Backtest Runs</h2>"
        recent_runs = runs_df.head(20)

        if not recent_runs.empty:
            html_content += """
            <table>
                <tr>
                    <th>Run Name</th>
                    <th>Strategy</th>
                    <th>Total Return</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown</th>
                    <th>Start Time</th>
                </tr>
            """

            for _, run in recent_runs.iterrows():
                return_val = run.get('metrics.total_return', 0)
                sharpe_val = run.get('metrics.sharpe_ratio', 0)
                dd_val = run.get('metrics.max_drawdown', 0)

                return_class = "metric-positive" if return_val > 0 else "metric-negative"

                html_content += f"""
                <tr>
                    <td>{run.get('tags.mlflow.runName', 'Unknown')}</td>
                    <td>{run.get('params.strategy', 'Unknown')}</td>
                    <td class="{return_class}">{return_val:.2%}</td>
                    <td>{sharpe_val:.2f}</td>
                    <td class="metric-negative">{dd_val:.2%}</td>
                    <td>{run['start_time'].strftime('%Y-%m-%d %H:%M')}</td>
                </tr>
                """

            html_content += "</table>"

        html_content += """
        </body>
        </html>
        """

        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"Performance report saved to {output_path}")

class ModelRegistry:
    """MLflow Model Registry for production models"""

    def __init__(self, tracking_uri: str = None):
        """
        Initialize model registry

        Args:
            tracking_uri: MLflow tracking URI
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def register_best_model(self, experiment_name: str, strategy: str, metric: str = "total_return"):
        """
        Register the best performing model for a strategy

        Args:
            experiment_name: MLflow experiment name
            strategy: Strategy name
            metric: Metric to use for selection

        Returns:
            Registered model version info
        """
        try:
            # Find best run
            comparator = ExperimentComparator(experiment_name)
            best_params = comparator.find_best_hyperparameters(strategy, metric)

            if not best_params:
                print(f"No models found for strategy {strategy}")
                return None

            run_id = best_params['run_id']

            # Register model
            model_uri = f"runs:/{run_id}/backtest_model"
            model_version = mlflow.register_model(model_uri, f"{strategy}_production")

            print(f"Registered model version: {model_version.version}")
            return model_version

        except Exception as e:
            print(f"Failed to register model: {e}")
            return None

    def transition_model_stage(self, model_name: str, version: str, stage: str):
        """
        Transition model to different stage

        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage ('Staging', 'Production', 'Archived')
        """
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            print(f"Model {model_name} v{version} transitioned to {stage}")
        except Exception as e:
            print(f"Failed to transition model stage: {e}")

# Convenience functions
def create_backtest_tracker(experiment_name: str = "quant_backtests") -> BacktestTracker:
    """Create backtest tracker"""
    return BacktestTracker(experiment_name)

def create_experiment_comparator(experiment_name: str = "quant_backtests") -> ExperimentComparator:
    """Create experiment comparator"""
    return ExperimentComparator(experiment_name)

def create_model_registry() -> ModelRegistry:
    """Create model registry"""
    return ModelRegistry()

if __name__ == "__main__":
    # Example usage
    tracker = create_backtest_tracker("quant_backtests_demo")
    comparator = create_experiment_comparator("quant_backtests_demo")

    # Log a sample run
    with tracker.start_run("demo_run", tags={"strategy": "momentum", "symbol": "SPY"}):
        tracker.log_backtest_params({
            "window": 20,
            "threshold": 0.05,
            "stop_loss": 0.10
        })

        tracker.log_backtest_metrics({
            "total_return": 0.156,
            "sharpe_ratio": 1.23,
            "max_drawdown": -0.085,
            "win_rate": 0.62
        })

    print("MLflow tracking demo completed!")

    # Generate performance report
    comparator.generate_performance_report("demo_report.html")
