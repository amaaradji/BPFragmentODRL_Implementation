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

# Import all modules
from bpmn_parser import BPMNParser
from enhanced_fragmenter import EnhancedFragmenter
# We will conditionally import the policy generators later
# from enhanced_policy_generator import EnhancedPolicyGenerator
# from enhanced_policy_generator_llm import EnhancedPolicyGenerator as EnhancedPolicyGeneratorLLM
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
        
        os.makedirs(self.parsed_models_dir, exist_ok=True)
        os.makedirs(self.fragments_dir, exist_ok=True)
        os.makedirs(self.policies_dir, exist_ok=True)
        os.makedirs(self.conflicts_dir, exist_ok=True)
        os.makedirs(self.reconstruction_dir, exist_ok=True)
        
        # Initialize results container
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
        
        if max_models > 0 and max_models < len(bpmn_files):
            bpmn_files = bpmn_files[:max_models]
        
        logger.info(f"Found {len(bpmn_files)} BPMN models in dataset")
        self.results['summary']['total_models'] = len(bpmn_files)
        
        for bpmn_file in tqdm(bpmn_files, desc="Evaluating models"):
            try:
                model_result = self._process_model(bpmn_file)
                self.results['models'].append(model_result)
                self.results['summary']['successful_models'] += 1
            except Exception as e:
                logger.error(f"Error processing model {bpmn_file}: {str(e)}", exc_info=True)
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
        bp_model_data = parser.parse_file(bpmn_file) # Renamed to bp_model_data for clarity
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
        
        # Step 3: Generate policies (Conditional Instantiation)
        start_time = time.time()
        policy_generator = None
        use_llm_for_generation = False

        if self.policy_generator_type == "llm_based":
            try:
                from enhanced_policy_generator_llm import EnhancedPolicyGenerator as PolicyGeneratorLLM
                # Pass bp_model_data as model_data and the strategy
                policy_generator = PolicyGeneratorLLM(model_data=bp_model_data, fragmentation_strategy=self.fragmentation_strategy)
                use_llm_for_generation = True # Flag to pass to generate_policies if needed by LLM version
                logger.info(f"Using LLM-based policy generator for {model_name}")
            except ImportError as e:
                logger.error(f"Could not import LLM policy generator: {e}. Falling back to rule-based for {model_name}.")
                from enhanced_policy_generator import EnhancedPolicyGenerator as PolicyGeneratorRuleBased
                policy_generator = PolicyGeneratorRuleBased(bp_model_data, fragments, fragment_dependencies)
        else: # rule_based
            from enhanced_policy_generator import EnhancedPolicyGenerator as PolicyGeneratorRuleBased
            policy_generator = PolicyGeneratorRuleBased(bp_model_data, fragments, fragment_dependencies)
            logger.info(f"Using rule-based policy generator for {model_name}")

        # The generate_policies method signature might differ or take flags.
        # The LLM version I provided has generate_policies(self, fragments_data, use_llm=True)
        # The original rule-based one has generate_policies(self, use_templates=True, policy_density=0.7)
        if self.policy_generator_type == "llm_based" and use_llm_for_generation:
             # Assuming fragments is the fragments_data the LLM generator expects
            activity_policies, dependency_policies = policy_generator.generate_policies(fragments, fragment_dependencies, use_llm=True)
        else:
            # Rule-based generator call (adjust params if your original differs)
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
        
        # Step 4: Check policy consistency
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
        
        # Step 5: Create a synthetic original BP-level policy for comparison
        original_bp_policy = self._create_synthetic_bp_policy(bp_model_data, activity_policies)
        
        # Step 6: Reconstruct the policy
        start_time = time.time()
        reconstructor = PolicyReconstructor(activity_policies, dependency_policies, original_bp_policy, fragments)
        reconstructed_policy = reconstructor.reconstruct_policy()
        reconstruction_time = time.time() - start_time
        
        reconstruction_metrics = reconstructor.get_reconstruction_metrics()
        
        reconstruction_dir_path = os.path.join(self.reconstruction_dir, os.path.splitext(model_name)[0]) # Renamed variable
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
        """
        Create a synthetic original BP-level policy for comparison.
        This is a placeholder and should be adapted based on actual policy structure.
        """
        # For simplicity, let's assume the synthetic policy is a collection of all activity policies
        # In a real scenario, this would be more complex or loaded from an actual source
        synthetic_policy = {
            "permissions": [],
            "prohibitions": [],
            "obligations": []
        }
        for policy_list in activity_policies.values():
            for policy_type, policies in policy_list.items():
                if policy_type in synthetic_policy:
                    synthetic_policy[policy_type].extend(policies)
        return synthetic_policy

    def _calculate_policy_size(self, activity_policies, dependency_policies):
        """
        Calculate the total size of generated policies in KB.
        """
        total_size_bytes = 0
        total_size_bytes += sys.getsizeof(json.dumps(activity_policies))
        total_size_bytes += sys.getsizeof(json.dumps(dependency_policies))
        return total_size_bytes / 1024 # Convert to KB

    def _calculate_summary(self):
        """
        Calculate summary statistics from the evaluation results.
        """
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
        """
        Save the evaluation results to JSON and CSV files.
        """
        results_json_path = os.path.join(self.output_path, 'evaluation_results.json')
        with open(results_json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Detailed results saved to {results_json_path}")
        
        # Create a flat CSV for easier analysis
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
                        'error': res.get('error', '')
                    })
            
            df_results = pd.DataFrame(model_metrics_list)
            results_csv_path = os.path.join(self.output_path, 'results.csv')
            df_results.to_csv(results_csv_path, index=False)
            logger.info(f"Summary results saved to {results_csv_path}")

    def generate_visualizations(self):
        """
        Generate visualizations from the evaluation results.
        Requires matplotlib to be installed.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("Matplotlib or Seaborn not installed. Skipping visualization generation.")
            return

        if not self.results['models'] or self.results['summary']['successful_models'] == 0:
            logger.warning("No successful model results to visualize.")
            return

        df = pd.DataFrame([res['metrics'] for res in self.results['models'] if res['status'] == 'success'])
        if df.empty:
            logger.warning("DataFrame for visualization is empty.")
            return

        viz_dir = os.path.join(self.output_path, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # Plot 1: Activities vs. Rules
        plt.figure(figsize=(10, 6))
        df['total_rules'] = df['permissions'] + df['prohibitions'] + df['obligations']
        sns.scatterplot(data=df, x='activities', y='total_rules', hue='fragmentation_strategy')
        plt.title('Activities vs. Total Generated Rules')
        plt.xlabel('Number of Activities')
        plt.ylabel('Total Generated Rules')
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, 'activities_vs_rules.png'))
        plt.close()

        # Plot 2: Fragments vs. Conflicts
        plt.figure(figsize=(10, 6))
        df['total_conflicts'] = df['intra_fragment_conflicts'] + df['inter_fragment_conflicts']
        sns.scatterplot(data=df, x='fragments', y='total_conflicts', hue='fragmentation_strategy')
        plt.title('Fragments vs. Total Conflicts')
        plt.xlabel('Number of Fragments')
        plt.ylabel('Total Conflicts')
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, 'fragments_vs_conflicts.png'))
        plt.close()

        # Plot 3: Process Size (Activities) vs. Generation Time
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='activities', y='policy_generation_time', hue='fragmentation_strategy')
        plt.title('Process Size (Activities) vs. Policy Generation Time')
        plt.xlabel('Number of Activities')
        plt.ylabel('Policy Generation Time (s)')
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, 'size_vs_generation_time.png'))
        plt.close()

        # Plot 4: Rule Distribution (Bar Plot)
        rule_counts = df[['permissions', 'prohibitions', 'obligations']].sum()
        plt.figure(figsize=(8, 6))
        rule_counts.plot(kind='bar', color=['green', 'red', 'blue'])
        plt.title('Distribution of Generated Rule Types')
        plt.ylabel('Total Count')
        plt.xticks(rotation=0)
        plt.savefig(os.path.join(viz_dir, 'rule_distribution.png'))
        plt.close()        # Plot 5: Reconstruction Accuracy (Histogram)
        if len(df["reconstruction_accuracy"].dropna()) >= 2:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x="reconstruction_accuracy", kde=True, hue="fragmentation_strategy", multiple="stack")
            plt.title("Distribution of Policy Reconstruction Accuracy")
            plt.xlabel("Reconstruction Accuracy")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(os.path.join(viz_dir, "reconstruction_accuracy.png"))
            plt.close()
        else:
            logger.warning("Skipping reconstruction accuracy histogram as there are fewer than 2 data points.")# Plot 6: Conflict Types (Stacked Bar)
        conflict_types = df[['intra_fragment_conflicts', 'inter_fragment_conflicts']].sum()
        plt.figure(figsize=(8, 6))
        conflict_types.plot(kind='bar', stacked=True, color=['orange', 'purple'])
        plt.title('Distribution of Conflict Types')
        plt.ylabel('Total Count')
        plt.xticks(rotation=0)
        plt.savefig(os.path.join(viz_dir, 'conflict_distribution.png'))
        plt.close()

        logger.info(f"Visualizations saved to {viz_dir}")

    def generate_summary_report(self):
        """
        Generate a summary report in Markdown format.
        """
        report_path = os.path.join(self.output_path, 'summary_report.md')
        summary = self.results['summary']

        with open(report_path, 'w') as f:
            f.write("# BPFragmentODRL Evaluation Summary Report\n\n")
            f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Evaluation Configuration\n")
            f.write(f"- Dataset Path: `{self.dataset_path}`\n")
            f.write(f"- Output Path: `{self.output_path}`\n")
            f.write(f"- Fragmentation Strategy: `{self.fragmentation_strategy}`\n")
            f.write(f"- Policy Generator Type: `{self.policy_generator_type}`\n\n")
            
            f.write("## Overall Summary\n")
            f.write(f"- Total Models Processed: {summary['total_models']}\n")
            f.write(f"- Successful Models: {summary['successful_models']}\n")
            f.write(f"- Failed Models: {summary['failed_models']}\n\n")
            
            if summary['successful_models'] > 0:
                f.write("## Average Metrics (for successful models)\n")
                f.write(f"- Average Activities per Model: {summary['avg_activities']:.2f}\n")
                f.write(f"- Average Fragments per Model: {summary['avg_fragments']:.2f}\n")
                f.write(f"- Average Policy Generation Time (s): {summary['avg_policy_generation_time']:.4f}\n")
                f.write(f"- Average Permissions: {summary['avg_permissions']:.2f}\n")
                f.write(f"- Average Prohibitions: {summary['avg_prohibitions']:.2f}\n")
                f.write(f"- Average Obligations: {summary['avg_obligations']:.2f}\n")
                f.write(f"- Average Intra-fragment Conflicts: {summary['avg_intra_conflicts']:.2f}\n")
                f.write(f"- Average Inter-fragment Conflicts: {summary['avg_inter_conflicts']:.2f}\n")
                f.write(f"- Average Policy Reconstruction Accuracy: {summary['avg_reconstruction_accuracy']:.2%}\n")
                f.write(f"- Average Policy Size (KB): {summary['avg_policy_size_kb']:.2f}\n\n")
            
            f.write("## Visualizations\n")
            f.write("Visualizations are saved in the `visualizations` subdirectory.\n")
            f.write("- Activities vs. Rules: `visualizations/activities_vs_rules.png`\n")
            f.write("- Fragments vs. Conflicts: `visualizations/fragments_vs_conflicts.png`\n")
            f.write("- Process Size vs. Generation Time: `visualizations/size_vs_generation_time.png`\n")
            f.write("- Rule Distribution: `visualizations/rule_distribution.png`\n")
            f.write("- Reconstruction Accuracy: `visualizations/reconstruction_accuracy.png`\n")
            f.write("- Conflict Distribution: `visualizations/conflict_distribution.png`\n\n")
            
            f.write("## Detailed Results\n")
            f.write("Detailed model-by-model results are available in `results.csv` and `evaluation_results.json`.\n")

        logger.info(f"Summary report saved to {report_path}")
        return report_path

if __name__ == '__main__':
    # Example usage (for testing the pipeline directly)
    parser = argparse.ArgumentParser(description='BPFragmentODRL Evaluation Pipeline - Direct Test')
    parser.add_argument('--dataset', type=str, default='../datasets/FBPM/FBPM2-ProcessModels', help='Path to the dataset directory')
    parser.add_argument('--output', type=str, default='../results_pipeline_test', help='Path to the output directory')
    parser.add_argument('--strategy', type=str, default='gateway', choices=['gateway', 'activity', 'connected', 'hierarchical'], help='Fragmentation strategy')
    parser.add_argument('--generator', type=str, default='rule_based', choices=['rule_based', 'llm_based'], help='Policy generator type')
    parser.add_argument('--max_models', type=int, default=2, help='Maximum number of models to process')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    pipeline = EvaluationPipeline(args.dataset, args.output, args.strategy, args.generator)
    pipeline.run_evaluation(args.max_models)
    pipeline.generate_visualizations()
    pipeline.generate_summary_report()
    print(f"Pipeline test completed. Results in {args.output}")

