"""
run_evaluation.py

Script to run the complete evaluation pipeline for the BPFragmentODRL system.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from evaluation_pipeline import EvaluationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the evaluation pipeline."""
    parser = argparse.ArgumentParser(description='BPFragmentODRL Evaluation Pipeline')
    parser.add_argument('--dataset', type=str, default='datasets/FBPM/FBPM2-ProcessModels',
                        help='Path to the dataset directory')
    parser.add_argument('--output', type=str, default='results',
                        help='Path to the output directory')
    parser.add_argument('--strategy', type=str, default='gateway',
                        choices=['gateway', 'activity', 'connected', 'hierarchical'],
                        help='Fragmentation strategy')
    parser.add_argument('--max_models', type=int, default=10,
                        help='Maximum number of models to process (0 for all)')
    
    args = parser.parse_args()
    
    # Log start time and configuration
    start_time = datetime.now()
    logger.info(f"Starting evaluation at {start_time}")
    logger.info(f"Configuration: dataset={args.dataset}, output={args.output}, "
                f"strategy={args.strategy}, max_models={args.max_models}")
    
    # Ensure the dataset directory exists
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset directory not found: {args.dataset}")
        print(f"\nERROR: Dataset directory not found: {args.dataset}")
        print("\nUSER ACTION REQUIRED")
        print("Please download the BPMN XML files manually from the provided dataset link and place them in the folder:")
        print(f"/{args.dataset}/")
        return
    
    # Create the evaluation pipeline
    pipeline = EvaluationPipeline(args.dataset, args.output, args.strategy)
    
    try:
        # Run the evaluation
        logger.info("Running evaluation pipeline...")
        pipeline.run_evaluation(args.max_models)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        pipeline.generate_visualizations()
        
        # Generate summary report
        logger.info("Generating summary report...")
        report_file = pipeline.generate_summary_report()
        
        # Log completion
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Evaluation completed at {end_time}")
        logger.info(f"Total duration: {duration}")
        
        # Print summary
        print("\n" + "="*80)
        print(f"EVALUATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {os.path.abspath(args.output)}")
        print(f"Summary report: {os.path.abspath(report_file)}")
        print(f"Total duration: {duration}")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        print(f"\nERROR: Evaluation failed: {str(e)}")
        return

if __name__ == "__main__":
    main()
