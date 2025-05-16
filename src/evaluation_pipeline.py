"""
evaluation_pipeline.py

Provides a complete evaluation pipeline for the BPFragmentODRL system.
"""

import os
import sys
import json
import time
import logging
import argparse
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import all modules
from bpmn_parser import BPMNParser
from enhanced_fragmenter import EnhancedFragmenter
# Policy generators are imported conditionally within _process_model
from policy_consistency_checker import PolicyConsistencyChecker
from policy_reconstructor import PolicyReconstructor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """
    EvaluationPipeline integrates all components of the BPFragmentODRL system
    and provides a complete evaluation workflow.
    """
    
    def __init__(self, dataset_path, output_path, fragmentation_strategy='gateway', policy_generator_type='rule_based'):
        """
        Initialize the evaluation pipeline.
        
        :param dataset_path: Path to the dataset directory containing BPMN XML files
        :param output_path: Path to the output directory for results
        :param fragmentation_strategy: Strategy for fragmenting BPMN models (
            "gateway", "activity", "connected", "hierarchical"
        )
        :param policy_generator_type: Type of policy generator to use ("rule_based" or "llm_based")
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.fragmentation_strategy = fragmentation_strategy
        self.policy_generator_type = policy_generator_type # Store the generator type
        
        # Create output directories
        self.parsed_models_dir = os.path.join(output_path, 'parsed_models')
        self.fragments_dir = os.path.join(output_path, 'fragments')
        self.policies_dir = os.path.join(output_path, 'policies')
        self.conflicts_dir = os.path.join(output_path, 'conflicts')
        self.reconstruction_dir = os.path.join(output_path, 'reconstruction')
        self.visualizations_path = os.path.join(output_path, 'visualizations') 
        
        os.makedirs(self.parsed_models_dir, exist_ok=True)
        os.makedirs(self.fragments_dir, exist_ok=True)
        os.makedirs(self.policies_dir, exist_ok=True)
        os.makedirs(self.conflicts_dir, exist_ok=True)
        os.makedirs(self.reconstruction_dir, exist_ok=True)
        os.makedirs(self.visualizations_path, exist_ok=True) 
        
        self.results = {
            'models': [],
            'summary': {
                'total_models': 0,
                'successful_models': 0,
                'failed_models': 0,
                'avg_activities': 0,
                'avg_fragments': 0,
                'avg_policy_generation_time': 0,
                'avg_permissions': 0,
                'avg_prohibitions': 0,
                'avg_obligations': 0,
                'avg_intra_conflicts': 0,
                'avg_inter_conflicts': 0,
                'avg_reconstruction_accuracy': 0,
                'avg_policy_size_kb': 0
            }
        }
    
    def run_evaluation(self, max_models=0):
        """
        Run the complete evaluation pipeline on all BPMN models in the dataset.
        
        :param max_models: Maximum number of models to process (0 for all)
        """
        bpmn_files = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.bpmn') or file.endswith('.xml'):
                    bpmn_files.append(os.path.join(root, file))
        
        if max_models > 0 and len(bpmn_files) > max_models:
            bpmn_files = bpmn_files[:max_models]
        
        logger.info(f"Found {len(bpmn_files)} BPMN models in dataset to process.")
        self.results['summary']['total_models'] = len(bpmn_files)
        
        for bpmn_file in tqdm(bpmn_files, desc="Evaluating models"):
            try:
                model_result = self._process_model(bpmn_file)
                self.results['models'].append(model_result)
                if model_result['status'] == 'success':
                    self.results['summary']['successful_models'] += 1
                else:
                    self.results['summary']['failed_models'] += 1
            except Exception as e:
                logger.error(f"Critical error processing model {bpmn_file}: {str(e)}", exc_info=True)
                self.results['summary']['failed_models'] += 1
                self.results['models'].append({
                    'model_name': os.path.basename(bpmn_file),
                    'status': 'failed',
                    'error': str(e)
                })
        
        self._calculate_summary()
        self._save_results()
    
    def _process_model(self, bpmn_file):
        """
        Process a single BPMN model through the complete pipeline.
        """
        model_name = os.path.basename(bpmn_file)
        logger.debug(f"Processing model: {model_name}")
        
        model_result = {
            'model_name': model_name,
            'status': 'success',
            'file_path': bpmn_file,
            'metrics': {}
        }
        
        start_time = time.time()
        parser = BPMNParser()
        bp_model_data = parser.parse_file(bpmn_file)
        parsing_time = time.time() - start_time
        
        parsed_model_path = os.path.join(self.parsed_models_dir, f"{os.path.splitext(model_name)[0]}.json")
        with open(parsed_model_path, 'w') as f:
            json.dump(bp_model_data, f, indent=2)
        
        model_result['metrics']['activities'] = len(bp_model_data['activities'])
        model_result['metrics']['gateways'] = len(bp_model_data.get('gateways', []))
        model_result['metrics']['flows'] = len(bp_model_data.get('flows', []))
        model_result['metrics']['parsing_time'] = parsing_time
        
        start_time = time.time()
        fragmenter = EnhancedFragmenter(bp_model_data)
        fragments = fragmenter.fragment_process(strategy=self.fragmentation_strategy)
        fragment_dependencies = fragmenter.fragment_dependencies
        fragmentation_time = time.time() - start_time
        
        fragment_dir = os.path.join(self.fragments_dir, os.path.splitext(model_name)[0])
        os.makedirs(fragment_dir, exist_ok=True)
        fragmenter.save_fragments(fragment_dir)
        
        model_result['metrics']['fragments'] = len(fragments)
        model_result['metrics']['fragment_dependencies'] = len(fragment_dependencies)
        model_result['metrics']['fragmentation_time'] = fragmentation_time
        model_result['metrics']['fragmentation_strategy'] = self.fragmentation_strategy
        
        start_time = time.time()
        policy_generator = None
        llm_generator_imported_successfully = False

        if self.policy_generator_type == "llm_based":
            try:
                from enhanced_policy_generator_llm import EnhancedPolicyGeneratorLLM
                policy_generator = EnhancedPolicyGeneratorLLM(model_data=bp_model_data, fragmentation_strategy=self.fragmentation_strategy)
                llm_generator_imported_successfully = True 
                logger.info(f"Using LLM-based policy generator for {model_name}")
            except ImportError as e:
                logger.error(f"Could not import LLM policy generator (EnhancedPolicyGeneratorLLM): {e}. Falling back to rule-based for {model_name}.")
                from enhanced_policy_generator import EnhancedPolicyGenerator
                policy_generator = EnhancedPolicyGenerator(bp_model_data, fragments, fragment_dependencies)
        else: 
            from enhanced_policy_generator import EnhancedPolicyGenerator
            policy_generator = EnhancedPolicyGenerator(bp_model_data, fragments, fragment_dependencies)
            logger.info(f"Using rule-based policy generator for {model_name}")

        if self.policy_generator_type == "llm_based" and llm_generator_imported_successfully:
            activity_policies, dependency_policies = policy_generator.generate_policies(fragments, fragment_dependencies, use_llm=True)
        else:
            activity_policies, dependency_policies = policy_generator.generate_policies(use_templates=True, policy_density=0.7)
        
        policy_generation_time = time.time() - start_time
        
        policy_dir = os.path.join(self.policies_dir, os.path.splitext(model_name)[0])
        os.makedirs(policy_dir, exist_ok=True)
        policy_generator.save_policies(policy_dir)
        
        policy_metrics = policy_generator.get_policy_metrics()
        
        model_result['metrics']['policy_generation_time'] = policy_generation_time
        model_result['metrics']['fragment_activity_policies'] = policy_metrics['fragment_activity_policies']['total_policies']
        model_result['metrics']['fragment_dependency_policies'] = policy_metrics['fragment_dependency_policies']['total_policies']
        model_result['metrics']['permissions'] = policy_metrics['total']['permissions']
        model_result['metrics']['prohibitions'] = policy_metrics['total']['prohibitions']
        model_result['metrics']['obligations'] = policy_metrics['total']['obligations']
        
        policy_size = self._calculate_policy_size(activity_policies, dependency_policies)
        model_result['metrics']['policy_size_kb'] = policy_size
        
        start_time = time.time()
        checker = PolicyConsistencyChecker(activity_policies, dependency_policies, fragments, fragment_dependencies)
        intra_conflicts = checker.check_intra_fragment_consistency()
        inter_conflicts = checker.check_inter_fragment_consistency()
        consistency_checking_time = time.time() - start_time
        
        conflict_dir = os.path.join(self.conflicts_dir, os.path.splitext(model_name)[0])
        os.makedirs(conflict_dir, exist_ok=True)
        checker.save_conflicts(conflict_dir)
        
        conflict_metrics = checker.get_conflict_metrics()
        
        model_result['metrics']['consistency_checking_time'] = consistency_checking_time
        model_result['metrics']['intra_fragment_conflicts'] = conflict_metrics['intra_fragment']['total']
        model_result['metrics']['inter_fragment_conflicts'] = conflict_metrics['inter_fragment']['total']
        model_result['metrics']['total_conflicts'] = conflict_metrics['total_conflicts']
        
        original_bp_policy = self._create_synthetic_bp_policy(bp_model_data, activity_policies)
        
        start_time = time.time()
        reconstructor = PolicyReconstructor(activity_policies, dependency_policies, original_bp_policy, fragments)
        reconstructed_policy = reconstructor.reconstruct_policy()
        reconstruction_time = time.time() - start_time
        
        reconstruction_metrics = reconstructor.get_reconstruction_metrics()
        
        reconstruction_dir_path = os.path.join(self.reconstruction_dir, os.path.splitext(model_name)[0])
        os.makedirs(reconstruction_dir_path, exist_ok=True)
        reconstructor.save_reconstruction(reconstruction_dir_path)
        
        model_result['metrics']['reconstruction_time'] = reconstruction_time
        model_result['metrics']['original_rules'] = reconstruction_metrics['total_original_rules']
        model_result['metrics']['reconstructed_rules'] = reconstruction_metrics['total_reconstructed_rules']
        model_result['metrics']['matched_rules'] = reconstruction_metrics['matched_rules']
        model_result['metrics']['lost_rules'] = reconstruction_metrics['lost_rules']
        model_result['metrics']['new_rules'] = reconstruction_metrics['new_rules']
        model_result['metrics']['reconstruction_accuracy'] = reconstruction_metrics['accuracy']
        
        return model_result

    def _create_synthetic_bp_policy(self, bp_model_data, activity_policies):
        synthetic_policy = {
            "permissions": [],
            "prohibitions": [],
            "obligations": []
        }
        for act_id, policy_types_for_act in activity_policies.items():
            for policy_type, rules_list in policy_types_for_act.items():
                if policy_type in synthetic_policy:
                    synthetic_policy[policy_type].extend(rules_list)
        return synthetic_policy

    def _calculate_policy_size(self, activity_policies, dependency_policies):
        total_size_bytes = 0
        total_size_bytes += sys.getsizeof(json.dumps(activity_policies))
        total_size_bytes += sys.getsizeof(json.dumps(dependency_policies))
        return total_size_bytes / 1024

    def _calculate_summary(self):
        if not self.results['models']:
            return
        
        df = pd.DataFrame([res['metrics'] for res in self.results['models'] if res['status'] == 'success'])
        if df.empty:
            logger.warning("No successful models to calculate summary from.")
            return

        self.results['summary']['avg_activities'] = df['activities'].mean()
        self.results['summary']['avg_fragments'] = df['fragments'].mean()
        self.results['summary']['avg_policy_generation_time'] = df['policy_generation_time'].mean()
        self.results['summary']['avg_permissions'] = df['permissions'].mean()
        self.results['summary']['avg_prohibitions'] = df['prohibitions'].mean()
        self.results['summary']['avg_obligations'] = df['obligations'].mean()
        self.results['summary']['avg_intra_conflicts'] = df['intra_fragment_conflicts'].mean()
        self.results['summary']['avg_inter_conflicts'] = df['inter_fragment_conflicts'].mean()
        self.results['summary']['avg_reconstruction_accuracy'] = df['reconstruction_accuracy'].mean()
        self.results['summary']['avg_policy_size_kb'] = df['policy_size_kb'].mean()

    def _save_results(self):
        results_json_path = os.path.join(self.output_path, 'evaluation_results.json')
        with open(results_json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Detailed results saved to {results_json_path}")
        
        if self.results['models']:
            model_metrics_list = []
            for res in self.results['models']:
                if res['status'] == 'success':
                    metrics = res['metrics'].copy()
                    metrics['model_name'] = res['model_name']
                    metrics['status'] = res['status']
                    model_metrics_list.append(metrics)
                else:
                    model_metrics_list.append({
                        'model_name': res['model_name'], 
                        'status': res['status'], 
                        'error': res.get('error', 'Unknown error')
                    })
            
            if model_metrics_list: 
                df = pd.DataFrame(model_metrics_list)
                results_csv_path = os.path.join(self.output_path, 'results.csv')
                df.to_csv(results_csv_path, index=False)
                logger.info(f"Summary results saved to {results_csv_path}")
            else:
                logger.warning("No data to save to CSV.")
        else:
            logger.warning("No models processed, skipping CSV summary.")

    def generate_visualizations(self):
        if not self.results['models'] or self.results['summary']['successful_models'] == 0:
            logger.warning("No successful model results to visualize.")
            return

        df_list = []
        for res in self.results['models']:
            if res['status'] == 'success':
                metrics = res['metrics'].copy()
                metrics['model_name'] = res['model_name'] 
                df_list.append(metrics)
        
        if not df_list:
            logger.warning("No successful model metrics to create DataFrame for visualization.")
            return
            
        df = pd.DataFrame(df_list)

        os.makedirs(self.visualizations_path, exist_ok=True)

        if 'fragments' in df.columns and 'activities' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='activities', y='fragments', hue='fragmentation_strategy')
            plt.title('Number of Fragments vs. Number of Activities')
            plt.xlabel('Number of Activities')
            plt.ylabel('Number of Fragments')
            plt.savefig(os.path.join(self.visualizations_path, 'fragments_vs_activities.png'))
            plt.close()

        if 'policy_generation_time' in df.columns and 'fragments' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='fragments', y='policy_generation_time', hue='fragmentation_strategy')
            plt.title('Policy Generation Time vs. Number of Fragments')
            plt.xlabel('Number of Fragments')
            plt.ylabel('Policy Generation Time (s)')
            plt.savefig(os.path.join(self.visualizations_path, 'policy_time_vs_fragments.png'))
            plt.close()

        if 'total_conflicts' in df.columns and 'fragments' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='fragments', y='total_conflicts', hue='fragmentation_strategy')
            plt.title('Total Conflicts vs. Number of Fragments')
            plt.xlabel('Number of Fragments')
            plt.ylabel('Total Conflicts')
            plt.savefig(os.path.join(self.visualizations_path, 'conflicts_vs_fragments.png'))
            plt.close()

        if 'policy_size_kb' in df.columns and 'fragments' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='fragments', y='policy_size_kb', hue='fragmentation_strategy')
            plt.title('Policy Size (KB) vs. Number of Fragments')
            plt.xlabel('Number of Fragments')
            plt.ylabel('Policy Size (KB)')
            plt.savefig(os.path.join(self.visualizations_path, 'policy_size_vs_fragments.png'))
            plt.close()

        if 'reconstruction_accuracy' in df.columns and not df['reconstruction_accuracy'].dropna().empty:
            if len(df['reconstruction_accuracy'].dropna().unique()) >= 2:
                try:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(data=df, x='reconstruction_accuracy', kde=True, hue='fragmentation_strategy', multiple="stack")
                    plt.title('Distribution of Policy Reconstruction Accuracy')
                    plt.xlabel('Reconstruction Accuracy')
                    plt.ylabel('Frequency')
                    plt.savefig(os.path.join(self.visualizations_path, 'reconstruction_accuracy_distribution.png'))
                    plt.close()
                except np.linalg.LinAlgError as e:
                    logger.warning(f"Could not generate reconstruction accuracy histogram with KDE due to LinAlgError: {e}. Skipping this plot.")
                except Exception as e:
                    logger.warning(f"An unexpected error occurred while generating reconstruction accuracy histogram: {e}. Skipping this plot.")
            else:
                logger.warning("Skipping reconstruction accuracy histogram as there are fewer than 2 unique data points.")
        else:
            logger.warning("Skipping reconstruction accuracy histogram as 'reconstruction_accuracy' column is missing or empty.")

        avg_metrics = {
            'Permissions': df['permissions'].mean() if 'permissions' in df.columns else 0,
            'Prohibitions': df['prohibitions'].mean() if 'prohibitions' in df.columns else 0,
            'Obligations': df['obligations'].mean() if 'obligations' in df.columns else 0,
            'Intra-Fragment Conflicts': df['intra_fragment_conflicts'].mean() if 'intra_fragment_conflicts' in df.columns else 0,
            'Inter-Fragment Conflicts': df['inter_fragment_conflicts'].mean() if 'inter_fragment_conflicts' in df.columns else 0
        }
        avg_df = pd.DataFrame(list(avg_metrics.items()), columns=['Metric', 'Average Value'])
        
        if not avg_df.empty:
            plt.figure(figsize=(12, 7))
            sns.barplot(data=avg_df, x='Metric', y='Average Value')
            plt.title('Average Policy and Conflict Metrics')
            plt.ylabel('Average Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.visualizations_path, 'average_metrics_barchart.png'))
            plt.close()

        logger.info(f"Visualizations saved to {self.visualizations_path}")

    def generate_summary_report(self):
        report_path = os.path.join(self.output_path, 'summary_report.md')
        with open(report_path, 'w') as f:
            f.write("# Evaluation Summary Report\n\n")
            f.write(f"* Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"* Dataset: {self.dataset_path}\n")
            f.write(f"* Fragmentation Strategy: {self.fragmentation_strategy}\n")
            f.write(f"* Policy Generator Type: {self.policy_generator_type}\n")
            f.write(f"* Output Path: {self.output_path}\n\n")
            
            f.write("## Overall Summary\n")
            f.write(f"- Total Models Processed: {self.results['summary']['total_models']}\n")
            f.write(f"- Successful Models: {self.results['summary']['successful_models']}\n")
            f.write(f"- Failed Models: {self.results['summary']['failed_models']}\n")
            
            if self.results['summary']['successful_models'] > 0:
                f.write("\n## Average Metrics (for successful models)\n")
                for key, value in self.results['summary'].items():
                    if key.startswith('avg_'):
                        f.write(f"- {key.replace('avg_', '').replace('_', ' ').capitalize()}: {value:.2f}\n")
            
            f.write("\n## Failed Models\n")
            if self.results['summary']['failed_models'] > 0:
                for model_res in self.results['models']:
                    if model_res['status'] == 'failed':
                        f.write(f"- **{model_res['model_name']}**: {model_res.get('error', 'Unknown error')}\n")
            else:
                f.write("No models failed during processing.\n")
            
            f.write("\n## Visualizations\n")
            f.write("Visualizations are saved in the `visualizations` subdirectory.\n")

        logger.info(f"Summary report saved to {report_path}")
        return report_path # Return the path to the report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run BPFragmentODRL Evaluation Pipeline")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--output', type=str, default='results', help="Path to the output directory")
    parser.add_argument('--strategy', type=str, default='gateway', 
                        choices=['gateway', 'activity', 'connected', 'hierarchical'], 
                        help="Fragmentation strategy")
    parser.add_argument('--max_models', type=int, default=0, help="Maximum number of models to process (0 for all)")
    parser.add_argument('--policy_generator_type', type=str, default='rule_based', 
                        choices=['rule_based', 'llm_based'], 
                        help="Type of policy generator to use ('rule_based' or 'llm_based')")

    args = parser.parse_args()
    
    logger.info(f"Starting evaluation at {datetime.now()}")
    logger.info(f"Configuration: dataset={args.dataset}, output={args.output}, strategy={args.strategy}, max_models={args.max_models}, policy_generator_type={args.policy_generator_type}")
    
    pipeline = EvaluationPipeline(args.dataset, args.output, args.strategy, args.policy_generator_type)
    report_file = None # Initialize report_file
    start_total_time = time.time()
    try:
        logger.info("Running evaluation pipeline...")
        pipeline.run_evaluation(max_models=args.max_models)
        logger.info("Generating visualizations...")
        pipeline.generate_visualizations()
        logger.info("Generating summary report...")
        report_file = pipeline.generate_summary_report() # Assign returned path
    except Exception as e:
        logger.error(f"ERROR: Evaluation failed: {str(e)}", exc_info=True)
        print(f"ERROR: Evaluation failed: {str(e)}") 
        sys.exit(1)
    finally:
        end_total_time = time.time()
        total_duration = end_total_time - start_total_time
        logger.info(f"Evaluation completed at {datetime.now()}")
        logger.info(f"Total duration: {datetime.utcfromtimestamp(total_duration).strftime('%H:%M:%S.%f')[:-3]}")
        
        print("\n================================================================================")
        print("EVALUATION COMPLETED SUCCESSFULLY" if pipeline.results['summary']['failed_models'] == 0 else "EVALUATION COMPLETED WITH ERRORS")
        print("================================================================================")
        print(f"Results saved to: {os.path.abspath(args.output)}")
        if report_file: # Check if report_file is not None before printing
            print(f"Summary report: {os.path.abspath(report_file)}")
        else:
            print(f"Summary report: Not generated due to an earlier error or empty results.")
        print(f"Total duration: {datetime.utcfromtimestamp(total_duration).strftime('%H:%M:%S.%f')[:-3]}")
        print("================================================================================\n")

