# BPFragmentODRL Implementation

This repository contains the implementation of a system for automated generation of fragment-level policies for business processes using ODRL (Open Digital Rights Language).

## Overview

The BPFragmentODRL system enables:
- Parsing BPMN XML models
- Fragmenting business processes using various strategies
- Generating ODRL-based policies for fragments and their dependencies, using either **rule-based heuristics or advanced Large Language Models (LLMs)**
- Checking consistency of fragment policies
- Reconstructing business process-level policies
- Evaluating the approach with comprehensive metrics

## Project Structure

```
BPFragmentODRL_Implementation/
├── datasets/                  # BPMN model datasets
│   └── FBPM/                  # FBPM dataset
├── results/                   # Evaluation results
│   ├── policies/              # Generated ODRL policies for each model
│   ├── conflicts/             # Detected policy conflicts for each model
│   ├── visualizations/        # Generated plots
│   └── summary_report.md      # Comprehensive report
├── src/                       # Source code
│   ├── bpmn_parser.py         # BPMN XML parser
│   ├── enhanced_fragmenter.py # Process fragmentation module
│   ├── enhanced_policy_generator.py # Rule-based fragment policy generation
│   ├── enhanced_policy_generator_llm.py # LLM-based fragment policy generation
│   ├── policy_consistency_checker.py # Conflict detection
│   ├── policy_reconstructor.py # Policy reconstruction
│   ├── evaluation_pipeline.py # Evaluation framework
│   └── visualization_generator.py # Results visualization
├── tests/                     # Test modules (if any)
├── run_evaluation.py          # Main script for running evaluation
└── requirements.txt           # Python dependencies
```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/BPFragmentODRL_Implementation.git
    cd BPFragmentODRL_Implementation
    ```

2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    For LLM-based policy generation, you will also need the `openai` library:
    ```bash
    pip install openai
    ```

## Configuration for LLM-Based Policy Generation

If you plan to use the LLM-based policy generator (`--policy_generator_type llm_based`), you must configure the following environment variables for Azure OpenAI:

-   `AZURE_OPENAI_KEY`: Your Azure OpenAI API key.
-   `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI service endpoint (e.g., `https://your-resource-name.openai.azure.com/`).
-   `AZURE_OPENAI_API_VERSION`: The API version for your Azure OpenAI deployment (e.g., `2024-05-01-preview`).
-   `AZURE_OPENAI_DEPLOYMENT_NAME`: The name of your deployed model (e.g., `gpt-4o`, `gpt-35-turbo`).

**Example (PowerShell):**
```powershell
$Env:AZURE_OPENAI_KEY = "YOUR_API_KEY"
$Env:AZURE_OPENAI_ENDPOINT = "https://your-resource-name.openai.azure.com/"
$Env:AZURE_OPENAI_API_VERSION = "2024-05-01-preview"
$Env:AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o"
```

**Example (Bash/Zsh):**
```bash
export AZURE_OPENAI_KEY="YOUR_API_KEY"
export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-05-01-preview"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o"
```

## Usage

### Running the Evaluation

To run the complete evaluation pipeline:

```bash
python run_evaluation.py --dataset path/to/your/datasets --strategy gateway --policy_generator_type rule_based
```
Or, for LLM-based generation:
```bash
python run_evaluation.py --dataset path/to/your/datasets --strategy gateway --policy_generator_type llm_based
```

**Parameters for `run_evaluation.py`:**

-   `--dataset`: Path to the directory containing BPMN XML files (e.g., `datasets/FBPM/FBPM2-ProcessModels`).
-   `--strategy`: Fragmentation strategy. Options include `gateway`, `activity`, `connected`, `hierarchical` (ensure the chosen strategy is fully implemented).
-   `--policy_generator_type`: Type of policy generator to use. 
    -   `rule_based` (default): Uses the heuristic-based generator.
    -   `llm_based`: Uses the LLM-based generator (requires environment variables to be set).
-   `--output`: Output directory for results (default: `results`).
-   `--max_models`: Maximum number of models to process from the dataset. Set to `0` or omit to process all models (default: `0`).

### Generating Visualizations

To generate visualizations from evaluation results (after running the pipeline):

```bash
python src/visualization_generator.py --results results/evaluation_results.json --output results/visualizations
```

## Datasets

The implementation supports the following datasets (ensure they are downloaded and placed in the `datasets` folder or provide the correct path):

1.  **FBPM Dataset**: Process models from "Fundamentals of Business Process Management"
    -   Source: http://fundamentals-of-bpm.org/process-model-collections/

2.  **Zenodo 3758705**: BPM Academic Initiative models
    -   Source: https://zenodo.org/records/3758705

3.  **Zenodo 7012043**: Complete process models
    -   Source: https://zenodo.org/records/7012043

## Key Components

### BPMN Parser

Converts BPMN XML files to an internal JSON format representing activities, gateways, sequence flows, and other relevant elements.

### Enhanced Fragmenter

Splits business processes into smaller, manageable fragments using various strategies such as gateway-based, activity-based, connected components, or hierarchical decomposition.

### Policy Generator

Creates ODRL-based policies for fragment activities (FPa) and fragment dependencies (FPd). Supports permissions, prohibitions, and obligations with constraints.

Two types of policy generators are available:

1.  **Rule-Based (`enhanced_policy_generator.py`)**: Generates policies based on predefined heuristics, activity naming conventions, and structural properties of the fragments.
2.  **LLM-Based (`enhanced_policy_generator_llm.py`)**: Leverages a Large Language Model (via Azure OpenAI) to interpret fragment content and generate contextually relevant ODRL policies. This approach aims for more nuanced and semantically rich policy creation.

### Consistency Checker

Identifies potential conflicts in the generated fragment policies, including:
-   Intra-fragment conflicts (e.g., a permission and a prohibition for the same action on the same activity within a fragment under overlapping conditions).
-   Inter-fragment conflicts (e.g., a dependency policy allowing transition to a fragment where the entry activity is prohibited).

### Policy Reconstructor

Recombines fragment policies into a complete business process-level policy. This component also evaluates the accuracy of the reconstruction process against original or synthesized global policies if available.

## Results

The evaluation results are saved in the specified output directory (default: `results/`):

-   `results/results.csv`: Summary metrics for each processed model.
-   `results/evaluation_results.json`: Detailed JSON output containing all metrics and generated data for each model.
-   `results/policies/<model_name>/`: Directory containing generated `activity_policies.json` and `dependency_policies.json` for each model.
-   `results/conflicts/<model_name>/`: Directory containing detected `intra_fragment_conflicts.json` and `inter_fragment_conflicts.json` for each model.
-   `results/visualizations/`: Various plots visualizing the evaluation metrics.
-   `results/summary_report.md`: A comprehensive Markdown report summarizing the evaluation run.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (assuming one exists).

## Acknowledgments

-   Based on the research paper "BPFragmentODRL: Automated Generation of Fragment-Level Policies for Business Processes" (or relevant source paper).
-   Uses datasets such as the FBPM dataset from "Fundamentals of Business Process Management".

