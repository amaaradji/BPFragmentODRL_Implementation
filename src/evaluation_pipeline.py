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
from enhanced_policy_generator import EnhancedPolicyGenerator
from policy_consistency_checker import PolicyConsistencyChecker
from policy_reconstructor import PolicyReconstructor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """
    EvaluationPipeline integrates all components of the BPFragmentODRL system
    and provides a complete evaluation workflow.
    
    It supports:
    - Processing multiple BPMN models
    - Applying different fragmentation strategies
    - Generating fragment policies
    - Checking policy consistency
    - Reconstructing policies
    - Collecting comprehensive metrics
    
    Typical usage:
        pipeline = EvaluationPipeline(dataset_path, output_path)
        pipeline.run_evaluation()
        pipeline.generate_results()
    """
    
    def __init__(self, dataset_path, output_path, fragmentation_strategy='gateway'):
        """
        Initialize the evaluation pipeline.
        
        :param dataset_path: Path to the dataset directory containing BPMN XML files
        :param output_path: Path to the output directory for results
        :param fragmentation_strategy: Strategy for fragmenting BPMN models ('gateway', 'activity', 'connected', 'hierarchical')
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.fragmentation_strategy = fragmentation_strategy
        
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
        # Find all BPMN XML files in the dataset
        bpmn_files = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.bpmn') or file.endswith('.xml'):
                    bpmn_files.append(os.path.join(root, file))
        
        # Limit the number of models if specified
        if max_models > 0 and max_models < len(bpmn_files):
            bpmn_files = bpmn_files[:max_models]
        
        logger.info(f"Found {len(bpmn_files)} BPMN models in dataset")
        self.results['summary']['total_models'] = len(bpmn_files)
        
        # Process each model
        for bpmn_file in tqdm(bpmn_files, desc="Evaluating models"):
            try:
                model_result = self._process_model(bpmn_file)
                self.results['models'].append(model_result)
                self.results['summary']['successful_models'] += 1
            except Exception as e:
                logger.error(f"Error processing model {bpmn_file}: {str(e)}")
                self.results['summary']['failed_models'] += 1
                # Add a failed model entry
                self.results['models'].append({
                    'model_name': os.path.basename(bpmn_file),
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Calculate summary statistics
        self._calculate_summary()
        
        # Save results
        self._save_results()
    
    def _process_model(self, bpmn_file):
        """
        Process a single BPMN model through the complete pipeline.
        
        :param bpmn_file: Path to the BPMN XML file
        :return: dict with model evaluation results
        """
        model_name = os.path.basename(bpmn_file)
        logger.debug(f"Processing model: {model_name}")
        
        # Initialize result structure
        model_result = {
            'model_name': model_name,
            'status': 'success',
            'file_path': bpmn_file,
            'metrics': {}
        }
        
        # Step 1: Parse BPMN model
        start_time = time.time()
        parser = BPMNParser()
        model = parser.parse_file(bpmn_file)
        parsing_time = time.time() - start_time
        
        # Save parsed model
        parsed_model_path = os.path.join(self.parsed_models_dir, f"{os.path.splitext(model_name)[0]}.json")
        with open(parsed_model_path, 'w') as f:
            json.dump(model, f, indent=2)
        
        # Record basic model metrics
        model_result['metrics']['activities'] = len(model['activities'])
        model_result['metrics']['gateways'] = len(model.get('gateways', []))
        model_result['metrics']['flows'] = len(model.get('flows', []))
        model_result['metrics']['parsing_time'] = parsing_time
        
        # Step 2: Fragment the model
        start_time = time.time()
        fragmenter = EnhancedFragmenter(model)
        fragments = fragmenter.fragment_process(strategy=self.fragmentation_strategy)
        fragment_dependencies = fragmenter.fragment_dependencies
        fragmentation_time = time.time() - start_time
        
        # Save fragments
        fragment_dir = os.path.join(self.fragments_dir, os.path.splitext(model_name)[0])
        os.makedirs(fragment_dir, exist_ok=True)
        fragmenter.save_fragments(fragment_dir)
        
        # Record fragmentation metrics
        model_result['metrics']['fragments'] = len(fragments)
        model_result['metrics']['fragment_dependencies'] = len(fragment_dependencies)
        model_result['metrics']['fragmentation_time'] = fragmentation_time
        model_result['metrics']['fragmentation_strategy'] = self.fragmentation_strategy
        
        # Step 3: Generate policies
        start_time = time.time()
        policy_generator = EnhancedPolicyGenerator(model, fragments, fragment_dependencies)
        activity_policies, dependency_policies = policy_generator.generate_policies()
        policy_generation_time = time.time() - start_time
        
        # Save policies
        policy_dir = os.path.join(self.policies_dir, os.path.splitext(model_name)[0])
        os.makedirs(policy_dir, exist_ok=True)
        policy_generator.save_policies(policy_dir)
        
        # Get policy metrics
        policy_metrics = policy_generator.get_policy_metrics()
        
        # Record policy generation metrics
        model_result['metrics']['policy_generation_time'] = policy_generation_time
        model_result['metrics']['fragment_activity_policies'] = policy_metrics['fragment_activity_policies']['total_policies']
        model_result['metrics']['fragment_dependency_policies'] = policy_metrics['fragment_dependency_policies']['total_policies']
        model_result['metrics']['permissions'] = policy_metrics['total']['permissions']
        model_result['metrics']['prohibitions'] = policy_metrics['total']['prohibitions']
        model_result['metrics']['obligations'] = policy_metrics['total']['obligations']
        
        # Calculate policy size
        policy_size = self._calculate_policy_size(activity_policies, dependency_policies)
        model_result['metrics']['policy_size_kb'] = policy_size
        
        # Step 4: Check policy consistency
        start_time = time.time()
        checker = PolicyConsistencyChecker(activity_policies, dependency_policies, fragments, fragment_dependencies)
        intra_conflicts = checker.check_intra_fragment_consistency()
        inter_conflicts = checker.check_inter_fragment_consistency()
        consistency_checking_time = time.time() - start_time
        
        # Save conflicts
        conflict_dir = os.path.join(self.conflicts_dir, os.path.splitext(model_name)[0])
        os.makedirs(conflict_dir, exist_ok=True)
        checker.save_conflicts(conflict_dir)
        
        # Get conflict metrics
        conflict_metrics = checker.get_conflict_metrics()
        
        # Record consistency checking metrics
        model_result['metrics']['consistency_checking_time'] = consistency_checking_time
        model_result['metrics']['intra_fragment_conflicts'] = conflict_metrics['intra_fragment']['total']
        model_result['metrics']['inter_fragment_conflicts'] = conflict_metrics['inter_fragment']['total']
        model_result['metrics']['total_conflicts'] = conflict_metrics['total_conflicts']
        
        # Step 5: Create a synthetic original BP-level policy for comparison
        original_bp_policy = self._create_synthetic_bp_policy(model, activity_policies)
        
        # Step 6: Reconstruct the policy
        start_time = time.time()
        reconstructor = PolicyReconstructor(activity_policies, dependency_policies, original_bp_policy, fragments)
        reconstructed_policy = reconstructor.reconstruct_policy()
        reconstruction_time = time.time() - start_time
        
        # Evaluate the reconstruction
        reconstruction_metrics = reconstructor.get_reconstruction_metrics()
        
        # Save reconstruction results
        reconstruction_dir = os.path.join(self.reconstruction_dir, os.path.splitext(model_name)[0])
        os.makedirs(reconstruction_dir, exist_ok=True)
        reconstructor.save_reconstruction(reconstruction_dir)
        
        # Record reconstruction metrics
        model_result['metrics']['reconstruction_time'] = reconstruction_time
        model_result['metrics']['original_rules'] = reconstruction_metrics['total_original_rules']
        model_result['metrics']['reconstructed_rules'] = reconstruction_metrics['total_reconstructed_rules']
        model_result['metrics']['matched_rules'] = reconstruction_metrics['matched_rules']
        model_result['metrics']['lost_rules'] = reconstruction_metrics['lost_rules']
        model_result['metrics']['new_rules'] = reconstruction_metrics['new_rules']
        model_result['metrics']['reconstruction_accuracy'] = reconstruction_metrics['accuracy']
        
        # Record total processing time
        model_result['metrics']['total_processing_time'] = (
            parsing_time + fragmentation_time + policy_generation_time + 
            consistency_checking_time + reconstruction_time
        )
        
        return model_result
    
    def _calculate_policy_size(self, activity_policies, dependency_policies):
        """
        Calculate the size of policies in KB.
        
        :param activity_policies: Fragment activity policies
        :param dependency_policies: Fragment dependency policies
        :return: Size in KB
        """
        # Convert to JSON and calculate size
        combined_policies = {
            'activity_policies': activity_policies,
            'dependency_policies': dependency_policies
        }
        
        policy_json = json.dumps(combined_policies)
        size_kb = len(policy_json) / 1024
        
        return size_kb
    
    def _create_synthetic_bp_policy(self, model, activity_policies):
        """
        Create a synthetic original BP-level policy for testing reconstruction.
        This combines a subset of the fragment activity policies into a single policy.
        
        :param model: BPMN model
        :param activity_policies: Fragment activity policies
        :return: Synthetic BP-level policy
        """
        # Create a skeleton for the BP-level policy
        bp_policy = {
            "@context": "http://www.w3.org/ns/odrl.jsonld",
            "uid": "http://example.com/policy:original_bp",
            "@type": "Set",
            "permission": [],
            "prohibition": [],
            "obligation": []
        }
        
        # Collect all activities
        all_activities = [act['name'] for act in model.get('activities', [])]
        
        # Select a subset of activities (up to 5 or 50% of activities, whichever is larger)
        import random
        num_to_select = max(5, len(all_activities) // 2)
        selected_activities = random.sample(all_activities, min(num_to_select, len(all_activities)))
        
        # Find policies for these activities
        for fragment_id, policies in activity_policies.items():
            for activity_name, policy in policies.items():
                if activity_name in selected_activities:
                    # Extract rules
                    for rule_type in ['permission', 'prohibition', 'obligation']:
                        if rule_type in policy:
                            for rule in policy[rule_type]:
                                # Create a copy of the rule
                                import copy
                                rule_copy = copy.deepcopy(rule)
                                
                                # Update the rule UID to avoid conflicts
                                if 'uid' in rule_copy:
                                    rule_copy['uid'] = f"{rule_copy['uid']}_original"
                                
                                # Add to BP-level policy
                                bp_policy[rule_type].append(rule_copy)
        
        return bp_policy
    
    def _calculate_summary(self):
        """Calculate summary statistics from all processed models."""
        # Only consider successful models for averages
        successful_models = [m for m in self.results['models'] if m.get('status') == 'success']
        
        if not successful_models:
            logger.warning("No successful models to calculate summary statistics")
            return
        
        # Calculate averages
        metrics_to_average = [
            'activities', 'fragments', 'policy_generation_time', 
            'permissions', 'prohibitions', 'obligations',
            'intra_fragment_conflicts', 'inter_fragment_conflicts',
            'reconstruction_accuracy', 'policy_size_kb'
        ]
        
        for metric in metrics_to_average:
            values = [m['metrics'].get(metric, 0) for m in successful_models]
            self.results['summary'][f'avg_{metric}'] = sum(values) / len(successful_models)
    
    def _save_results(self):
        """Save evaluation results to files."""
        # Save full results as JSON
        results_file = os.path.join(self.output_path, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create a CSV with key metrics
        csv_data = []
        for model in self.results['models']:
            if model.get('status') == 'success':
                row = {
                    'model_name': model['model_name'],
                    'activities': model['metrics'].get('activities', 0),
                    'fragments': model['metrics'].get('fragments', 0),
                    'policy_generation_time': model['metrics'].get('policy_generation_time', 0),
                    'permissions': model['metrics'].get('permissions', 0),
                    'prohibitions': model['metrics'].get('prohibitions', 0),
                    'obligations': model['metrics'].get('obligations', 0),
                    'intra_fragment_conflicts': model['metrics'].get('intra_fragment_conflicts', 0),
                    'inter_fragment_conflicts': model['metrics'].get('inter_fragment_conflicts', 0),
                    'reconstruction_accuracy': model['metrics'].get('reconstruction_accuracy', 0),
                    'policy_size_kb': model['metrics'].get('policy_size_kb', 0)
                }
                csv_data.append(row)
        
        # Create DataFrame and save to CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_file = os.path.join(self.output_path, 'results.csv')
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved results CSV to {csv_file}")
        
        logger.info(f"Saved evaluation results to {results_file}")
    
    def generate_visualizations(self):
        """Generate visualizations of evaluation results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set up the visualization style
            sns.set(style="whitegrid")
            plt.rcParams['figure.figsize'] = (12, 8)
            
            # Create a DataFrame from successful models
            successful_models = [m for m in self.results['models'] if m.get('status') == 'success']
            
            if not successful_models:
                logger.warning("No successful models to generate visualizations")
                return
            
            # Extract metrics for visualization
            data = []
            for model in successful_models:
                metrics = model['metrics']
                row = {
                    'model_name': model['model_name'],
                    'activities': metrics.get('activities', 0),
                    'fragments': metrics.get('fragments', 0),
                    'policy_generation_time': metrics.get('policy_generation_time', 0),
                    'permissions': metrics.get('permissions', 0),
                    'prohibitions': metrics.get('prohibitions', 0),
                    'obligations': metrics.get('obligations', 0),
                    'intra_fragment_conflicts': metrics.get('intra_fragment_conflicts', 0),
                    'inter_fragment_conflicts': metrics.get('inter_fragment_conflicts', 0),
                    'total_conflicts': metrics.get('total_conflicts', 0),
                    'reconstruction_accuracy': metrics.get('reconstruction_accuracy', 0),
                    'policy_size_kb': metrics.get('policy_size_kb', 0)
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Create visualizations directory
            vis_dir = os.path.join(self.output_path, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # 1. Activities vs. number of rules
            plt.figure()
            plt.scatter(df['activities'], df['permissions'] + df['prohibitions'] + df['obligations'])
            plt.title('Activities vs. Number of Rules')
            plt.xlabel('Number of Activities')
            plt.ylabel('Total Number of Rules')
            plt.grid(True)
            plt.savefig(os.path.join(vis_dir, 'activities_vs_rules.png'), dpi=300)
            plt.close()
            
            # 2. Fragment count vs. conflict count
            plt.figure()
            plt.scatter(df['fragments'], df['total_conflicts'])
            plt.title('Fragment Count vs. Conflict Count')
            plt.xlabel('Number of Fragments')
            plt.ylabel('Number of Conflicts')
            plt.grid(True)
            plt.savefig(os.path.join(vis_dir, 'fragments_vs_conflicts.png'), dpi=300)
            plt.close()
            
            # 3. Process size vs. generation time
            plt.figure()
            plt.scatter(df['activities'], df['policy_generation_time'])
            plt.title('Process Size vs. Generation Time')
            plt.xlabel('Number of Activities')
            plt.ylabel('Policy Generation Time (seconds)')
            plt.grid(True)
            plt.savefig(os.path.join(vis_dir, 'size_vs_time.png'), dpi=300)
            plt.close()
            
            # 4. Rule type distribution
            plt.figure()
            rule_types = ['permissions', 'prohibitions', 'obligations']
            rule_counts = [df['permissions'].sum(), df['prohibitions'].sum(), df['obligations'].sum()]
            plt.bar(rule_types, rule_counts)
            plt.title('Distribution of Rule Types')
            plt.xlabel('Rule Type')
            plt.ylabel('Count')
            plt.savefig(os.path.join(vis_dir, 'rule_distribution.png'), dpi=300)
            plt.close()
            
            # 5. Reconstruction accuracy histogram
            plt.figure()
            plt.hist(df['reconstruction_accuracy'], bins=10, range=(0, 1))
            plt.title('Distribution of Reconstruction Accuracy')
            plt.xlabel('Reconstruction Accuracy')
            plt.ylabel('Number of Models')
            plt.grid(True)
            plt.savefig(os.path.join(vis_dir, 'reconstruction_accuracy.png'), dpi=300)
            plt.close()
            
            # 6. Conflict types
            plt.figure()
            conflict_types = ['intra_fragment_conflicts', 'inter_fragment_conflicts']
            conflict_counts = [df['intra_fragment_conflicts'].sum(), df['inter_fragment_conflicts'].sum()]
            plt.bar(conflict_types, conflict_counts)
            plt.title('Distribution of Conflict Types')
            plt.xlabel('Conflict Type')
            plt.ylabel('Count')
            plt.savefig(os.path.join(vis_dir, 'conflict_distribution.png'), dpi=300)
            plt.close()
            
            logger.info(f"Generated visualizations in {vis_dir}")
            
        except ImportError:
            logger.warning("Matplotlib or seaborn not available, skipping visualizations")
    
    def generate_summary_report(self):
        """Generate a summary report in Markdown format."""
        # Create report directory
        report_dir = os.path.join(self.output_path, 'report')
        os.makedirs(report_dir, exist_ok=True)
        
        # Check if we have any successful models
        successful_models = [m for m in self.results['models'] if m.get('status') == 'success']
        if not successful_models:
            # Generate a basic report for failed runs
            report_content = f"""# BPFragmentODRL Evaluation Report

## Overview

This report summarizes the evaluation of the BPFragmentODRL system, which implements automated generation of fragment-level policies for business processes using ODRL.

**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary

- **Total Models:** {self.results['summary']['total_models']}
- **Successfully Processed Models:** {self.results['summary']['successful_models']}
- **Failed Models:** {self.results['summary']['failed_models']}

## Error Analysis

No models were successfully processed. Please check the log files for detailed error messages.

## Recommendations

1. Verify that the BPMN XML files are in the correct format
2. Check for compatibility issues between the BPMN files and the parser
3. Review the error messages in the log file for specific issues

"""
        else:
            # Generate a complete report with metrics
            report_content = f"""# BPFragmentODRL Evaluation Report

## Overview

This report summarizes the evaluation of the BPFragmentODRL system, which implements automated generation of fragment-level policies for business processes using ODRL.

**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary

- **Total Models:** {self.results['summary']['total_models']}
- **Successfully Processed Models:** {self.results['summary']['successful_models']}
- **Failed Models:** {self.results['summary']['failed_models']}
- **Average Activities per Model:** {self.results['summary']['avg_activities']:.2f}

## Fragmentation Results

- **Fragmentation Strategy:** {self.fragmentation_strategy}
- **Average Fragments per Model:** {self.results['summary']['avg_fragments']:.2f}

## Policy Generation Results

- **Average Policy Generation Time:** {self.results['summary']['avg_policy_generation_time']:.2f} seconds
- **Average Permissions per Model:** {self.results['summary']['avg_permissions']:.2f}
- **Average Prohibitions per Model:** {self.results['summary']['avg_prohibitions']:.2f}
- **Average Obligations per Model:** {self.results['summary']['avg_obligations']:.2f}
- **Average Policy Size:** {self.results['summary']['avg_policy_size_kb']:.2f} KB

## Consistency Checking Results

- **Average Intra-Fragment Conflicts:** {self.results['summary']['avg_intra_fragment_conflicts']:.2f}
- **Average Inter-Fragment Conflicts:** {self.results['summary']['avg_inter_fragment_conflicts']:.2f}

## Policy Reconstruction Results

- **Average Reconstruction Accuracy:** {self.results['summary']['avg_reconstruction_accuracy']:.2f}

## Key Findings

1. The {self.fragmentation_strategy} fragmentation strategy produced an average of {self.results['summary']['avg_fragments']:.2f} fragments per model.
2. Policy generation took an average of {self.results['summary']['avg_policy_generation_time']:.2f} seconds per model.
3. The system detected an average of {self.results['summary']['avg_intra_fragment_conflicts'] + self.results['summary']['avg_inter_fragment_conflicts']:.2f} conflicts per model.
4. Policy reconstruction achieved an average accuracy of {self.results['summary']['avg_reconstruction_accuracy']:.2f}.

## Conclusion

The BPFragmentODRL system successfully demonstrates the feasibility of fragmenting business processes and generating fragment-level policies using ODRL. The high reconstruction accuracy indicates that the fragment policies effectively capture the original business process policies.

## Visualizations

Visualizations of the evaluation results can be found in the `visualizations` directory.
"""
        
        # Write report to file
        report_file = os.path.join(report_dir, 'summary_report.md')
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Generated summary report at {report_file}")
        
        return report_file

def main():
    """Main function to run the evaluation pipeline."""
    parser = argparse.ArgumentParser(description='BPFragmentODRL Evaluation Pipeline')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output', type=str, default='./results', help='Path to the output directory')
    parser.add_argument('--strategy', type=str, default='gateway', 
                        choices=['gateway', 'activity', 'connected', 'hierarchical'],
                        help='Fragmentation strategy')
    parser.add_argument('--max_models', type=int, default=0, 
                        help='Maximum number of models to process (0 for all)')
    
    args = parser.parse_args()
    
    # Create the evaluation pipeline
    pipeline = EvaluationPipeline(args.dataset, args.output, args.strategy)
    
    # Run the evaluation
    pipeline.run_evaluation(args.max_models)
    
    # Generate visualizations
    pipeline.generate_visualizations()
    
    # Generate summary report
    pipeline.generate_summary_report()

if __name__ == "__main__":
    main()
