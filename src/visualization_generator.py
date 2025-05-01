"""
visualization_generator.py

Generates visualizations for the BPFragmentODRL evaluation results.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """
    Generates visualizations for the BPFragmentODRL evaluation results.
    
    This class creates various plots to visualize the results of the evaluation,
    including:
    - Activities vs. number of rules
    - Fragment count vs. conflict count
    - Dataset vs. reconstruction accuracy
    - Process size vs. generation time
    """
    
    def __init__(self, results_file, output_dir):
        """
        Initialize the visualization generator.
        
        :param results_file: Path to the evaluation results JSON file
        :param output_dir: Directory to save the generated visualizations
        """
        self.results_file = results_file
        self.output_dir = output_dir
        self.results = None
        self.df = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up the visualization style
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def load_results(self):
        """
        Load the evaluation results from the JSON file.
        
        :return: True if successful, False otherwise
        """
        try:
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
            
            # Extract successful models
            successful_models = [m for m in self.results['models'] if m.get('status') == 'success']
            
            if not successful_models:
                logger.warning("No successful models found in results")
                return False
            
            # Create DataFrame for easier plotting
            data = []
            for model in successful_models:
                metrics = model.get('metrics', {})
                row = {
                    'model_name': model['model_name'],
                    'activities': metrics.get('activities', 0),
                    'fragments': metrics.get('fragments', 0),
                    'policy_generation_time': metrics.get('policy_generation_time', 0),
                    'permissions': metrics.get('permissions', 0),
                    'prohibitions': metrics.get('prohibitions', 0),
                    'obligations': metrics.get('obligations', 0),
                    'total_rules': metrics.get('permissions', 0) + metrics.get('prohibitions', 0) + metrics.get('obligations', 0),
                    'intra_fragment_conflicts': metrics.get('intra_fragment_conflicts', 0),
                    'inter_fragment_conflicts': metrics.get('inter_fragment_conflicts', 0),
                    'total_conflicts': metrics.get('intra_fragment_conflicts', 0) + metrics.get('inter_fragment_conflicts', 0),
                    'reconstruction_accuracy': metrics.get('reconstruction_accuracy', 0),
                    'policy_size_kb': metrics.get('policy_size_kb', 0)
                }
                data.append(row)
            
            self.df = pd.DataFrame(data)
            return True
            
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            return False
    
    def generate_all_visualizations(self):
        """
        Generate all visualizations for the evaluation results.
        
        :return: List of generated visualization file paths
        """
        if self.df is None and not self.load_results():
            logger.error("Failed to load results for visualization")
            return []
        
        generated_files = []
        
        # Generate each visualization
        viz_functions = [
            self.plot_activities_vs_rules,
            self.plot_fragments_vs_conflicts,
            self.plot_size_vs_generation_time,
            self.plot_rule_distribution,
            self.plot_reconstruction_accuracy,
            self.plot_conflict_distribution
        ]
        
        for viz_func in viz_functions:
            try:
                file_path = viz_func()
                if file_path:
                    generated_files.append(file_path)
            except Exception as e:
                logger.error(f"Error generating visualization {viz_func.__name__}: {str(e)}")
        
        return generated_files
    
    def plot_activities_vs_rules(self):
        """
        Plot the number of activities vs. the number of rules.
        
        :return: Path to the generated visualization file
        """
        plt.figure()
        
        # Create scatter plot
        sns.scatterplot(data=self.df, x='activities', y='total_rules', s=100, alpha=0.7)
        
        # Add regression line
        sns.regplot(data=self.df, x='activities', y='total_rules', 
                   scatter=False, ci=None, line_kws={"color": "red"})
        
        # Add labels and title
        plt.title('Activities vs. Number of Rules', fontsize=16)
        plt.xlabel('Number of Activities', fontsize=14)
        plt.ylabel('Total Number of Rules', fontsize=14)
        plt.grid(True)
        
        # Add annotations for each point
        for i, row in self.df.iterrows():
            plt.annotate(row['model_name'], 
                        (row['activities'], row['total_rules']),
                        xytext=(5, 5), textcoords='offset points')
        
        # Save the figure
        output_path = os.path.join(self.output_dir, 'activities_vs_rules.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated activities vs. rules plot: {output_path}")
        return output_path
    
    def plot_fragments_vs_conflicts(self):
        """
        Plot the number of fragments vs. the number of conflicts.
        
        :return: Path to the generated visualization file
        """
        plt.figure()
        
        # Create scatter plot
        sns.scatterplot(data=self.df, x='fragments', y='total_conflicts', s=100, alpha=0.7)
        
        # Add regression line
        sns.regplot(data=self.df, x='fragments', y='total_conflicts', 
                   scatter=False, ci=None, line_kws={"color": "red"})
        
        # Add labels and title
        plt.title('Fragment Count vs. Conflict Count', fontsize=16)
        plt.xlabel('Number of Fragments', fontsize=14)
        plt.ylabel('Number of Conflicts', fontsize=14)
        plt.grid(True)
        
        # Add annotations for each point
        for i, row in self.df.iterrows():
            plt.annotate(row['model_name'], 
                        (row['fragments'], row['total_conflicts']),
                        xytext=(5, 5), textcoords='offset points')
        
        # Save the figure
        output_path = os.path.join(self.output_dir, 'fragments_vs_conflicts.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated fragments vs. conflicts plot: {output_path}")
        return output_path
    
    def plot_size_vs_generation_time(self):
        """
        Plot the process size vs. policy generation time.
        
        :return: Path to the generated visualization file
        """
        plt.figure()
        
        # Create scatter plot
        sns.scatterplot(data=self.df, x='activities', y='policy_generation_time', s=100, alpha=0.7)
        
        # Add regression line
        sns.regplot(data=self.df, x='activities', y='policy_generation_time', 
                   scatter=False, ci=None, line_kws={"color": "red"})
        
        # Add labels and title
        plt.title('Process Size vs. Generation Time', fontsize=16)
        plt.xlabel('Number of Activities', fontsize=14)
        plt.ylabel('Policy Generation Time (seconds)', fontsize=14)
        plt.grid(True)
        
        # Add annotations for each point
        for i, row in self.df.iterrows():
            plt.annotate(row['model_name'], 
                        (row['activities'], row['policy_generation_time']),
                        xytext=(5, 5), textcoords='offset points')
        
        # Save the figure
        output_path = os.path.join(self.output_dir, 'size_vs_time.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated size vs. time plot: {output_path}")
        return output_path
    
    def plot_rule_distribution(self):
        """
        Plot the distribution of rule types.
        
        :return: Path to the generated visualization file
        """
        plt.figure()
        
        # Calculate totals for each rule type
        rule_types = ['permissions', 'prohibitions', 'obligations']
        rule_counts = [self.df['permissions'].sum(), self.df['prohibitions'].sum(), self.df['obligations'].sum()]
        
        # Create bar plot
        bars = plt.bar(rule_types, rule_counts, color=['#3498db', '#e74c3c', '#2ecc71'])
        
        # Add labels and title
        plt.title('Distribution of Rule Types', fontsize=16)
        plt.xlabel('Rule Type', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.grid(axis='y')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Save the figure
        output_path = os.path.join(self.output_dir, 'rule_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated rule distribution plot: {output_path}")
        return output_path
    
    def plot_reconstruction_accuracy(self):
        """
        Plot the distribution of reconstruction accuracy.
        
        :return: Path to the generated visualization file
        """
        plt.figure()
        
        # Create histogram
        sns.histplot(data=self.df, x='reconstruction_accuracy', bins=10, kde=True)
        
        # Add labels and title
        plt.title('Distribution of Reconstruction Accuracy', fontsize=16)
        plt.xlabel('Reconstruction Accuracy', fontsize=14)
        plt.ylabel('Number of Models', fontsize=14)
        plt.grid(True)
        
        # Set x-axis limits
        plt.xlim(0, 1.05)
        
        # Save the figure
        output_path = os.path.join(self.output_dir, 'reconstruction_accuracy.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated reconstruction accuracy plot: {output_path}")
        return output_path
    
    def plot_conflict_distribution(self):
        """
        Plot the distribution of conflict types.
        
        :return: Path to the generated visualization file
        """
        plt.figure()
        
        # Calculate totals for each conflict type
        conflict_types = ['Intra-Fragment', 'Inter-Fragment']
        conflict_counts = [self.df['intra_fragment_conflicts'].sum(), self.df['inter_fragment_conflicts'].sum()]
        
        # Create bar plot
        bars = plt.bar(conflict_types, conflict_counts, color=['#9b59b6', '#f39c12'])
        
        # Add labels and title
        plt.title('Distribution of Conflict Types', fontsize=16)
        plt.xlabel('Conflict Type', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.grid(axis='y')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Save the figure
        output_path = os.path.join(self.output_dir, 'conflict_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated conflict distribution plot: {output_path}")
        return output_path

def main():
    """
    Main function to run the visualization generator.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualizations for BPFragmentODRL evaluation results')
    parser.add_argument('--results', type=str, default='results/evaluation_results.json',
                        help='Path to the evaluation results JSON file')
    parser.add_argument('--output', type=str, default='results/visualizations',
                        help='Directory to save the generated visualizations')
    
    args = parser.parse_args()
    
    # Create visualization generator
    viz_generator = VisualizationGenerator(args.results, args.output)
    
    # Generate all visualizations
    generated_files = viz_generator.generate_all_visualizations()
    
    if generated_files:
        logger.info(f"Generated {len(generated_files)} visualizations in {args.output}")
        for file_path in generated_files:
            logger.info(f"  - {os.path.basename(file_path)}")
    else:
        logger.warning("No visualizations were generated")

if __name__ == "__main__":
    main()
