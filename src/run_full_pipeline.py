#!/usr/bin/env python3
"""
Master script to run complete MSAGAT-Net experiments pipeline:
1. Run all experiments (training across datasets, horizons, ablations, and seeds)
2. Consolidate metrics
3. Generate publication figures
"""
import os
import sys
import subprocess
import logging
from datetime import datetime

# =============================================================================
# SETUP
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
LOG_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(LOG_DIR, exist_ok=True)

# Setup logging
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"master_experiments_{ts}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('master_experiments')

# =============================================================================
# PIPELINE STEPS
# =============================================================================

def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and handle errors."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return False
    
    logger.info(f"{'='*60}")
    logger.info(f"STEP: {description}")
    logger.info(f"Script: {script_name}")
    logger.info(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error in {description}: {e}")
        return False


def main():
    """Run the complete experiments pipeline."""
    logger.info("=" * 70)
    logger.info("MSAGAT-Net Complete Experiments Pipeline")
    logger.info("=" * 70)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info("")
    
    start_time = datetime.now()
    
    # Step 1: Run experiments (includes consolidation)
    step1_success = run_script(
        'run_experiments.py',
        'Training all models and consolidating metrics (7 datasets × multiple horizons × 4 ablations × 5 seeds)'
    )
    
    if not step1_success:
        logger.error("Training/consolidation failed. Cannot proceed to visualization.")
        return
    
    # Step 2: Generate publication figures
    step2_success = run_script(
        'generate_visualizations.py',
        'Generating publication-ready figures'
    )
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Step 1 (Training + Consolidation): {'✓ SUCCESS' if step1_success else '✗ FAILED'}")
    logger.info(f"Step 2 (Visualization):            {'✓ SUCCESS' if step2_success else '✗ FAILED'}")
    logger.info("")
    logger.info(f"Total duration: {duration}")
    logger.info(f"Log saved to: {log_file}")
    logger.info("=" * 70)
    
    # Output locations
    if step2_success:
        logger.info("")
        logger.info("Output locations:")
        logger.info(f"  - Models: {BASE_DIR}/save_all/")
        logger.info(f"  - Metrics: {BASE_DIR}/report/results/")
        logger.info(f"  - Figures: {BASE_DIR}/report/figures/paper/")
        logger.info(f"  - Logs: {LOG_DIR}/")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
