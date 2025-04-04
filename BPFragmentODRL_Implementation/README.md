# BPFragmentODRL Implementation

This repository contains the implementation of a system for automated generation of fragment-level policies for business processes using ODRL (Open Digital Rights Language).

## Overview

The BPFragmentODRL system enables:
- Parsing BPMN XML models
- Fragmenting business processes using various strategies
- Generating ODRL-based policies for fragments and their dependencies
- Checking consistency of fragment policies
- Reconstructing business process-level policies
- Evaluating the approach with comprehensive metrics

## Project Structure

```
BPFragmentODRL_Implementation/
├── datasets/                  # BPMN model datasets
│   └── FBPM/                  # FBPM dataset
├── results/                   # Evaluation results
│   ├── visualizations/        # Generated plots
│   └── summary_report.md      # Comprehensive report
├── src/                       # Source code
│   ├── bpmn_parser.py         # BPMN XML parser
│   ├── enhanced_fragmenter.py # Process fragmentation module
│   ├── enhanced_policy_generator.py # Fragment policy generation
│   ├── policy_consistency_checker.py # Conflict detection
│   ├── policy_reconstructor.py # Policy reconstruction
│   ├── evaluation_pipeline.py # Evaluation framework
│   └── visualization_generator.py # Results visualization
├── tests/                     # Test modules
├── run_evaluation.py          # Main script for running evaluation
└── requirements.txt           # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BPFragmentODRL_Implementation.git
cd BPFragmentODRL_Implementation
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Evaluation

To run the complete evaluation pipeline:

```bash
python run_evaluation.py --dataset datasets/FBPM/FBPM2-ProcessModels --strategy gateway
```

Parameters:
- `--dataset`: Path to the directory containing BPMN XML files
- `--strategy`: Fragmentation strategy (gateway, activity, connected, hierarchical)
- `--output`: Output directory for results (default: results)
- `--max_models`: Maximum number of models to process (default: all)

### Generating Visualizations

To generate visualizations from evaluation results:

```bash
python src/visualization_generator.py --results results/evaluation_results.json --output results/visualizations
```

## Datasets

The implementation supports the following datasets:

1. **FBPM Dataset**: Process models from "Fundamentals of Business Process Management"
   - Source: http://fundamentals-of-bpm.org/process-model-collections/

2. **Zenodo 3758705**: BPM Academic Initiative models
   - Source: https://zenodo.org/records/3758705

3. **Zenodo 7012043**: Complete process models
   - Source: https://zenodo.org/records/7012043

## Key Components

### BPMN Parser

Converts BPMN XML files to an internal JSON format with activities, gateways, and flows.

### Enhanced Fragmenter

Splits business processes into fragments using various strategies:
- Gateway-based fragmentation
- Activity-based fragmentation
- Connected components fragmentation
- Hierarchical decomposition

### Policy Generator

Creates ODRL-based policies for:
- Fragment activities (FPa)
- Fragment dependencies (FPd)

Supports permissions, prohibitions, and obligations with constraints.

### Consistency Checker

Identifies conflicts in fragment policies:
- Intra-fragment conflicts (within a fragment)
- Inter-fragment conflicts (between fragments)

### Policy Reconstructor

Recombines fragment policies into a complete business process-level policy and evaluates reconstruction accuracy.

## Results

The evaluation results are available in:
- `results/results.csv`: Summary metrics for each model
- `results/evaluation_results.json`: Detailed results
- `results/visualizations/`: Generated plots
- `results/summary_report.md`: Comprehensive report

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the research paper "BPFragmentODRL: Automated Generation of Fragment-Level Policies for Business Processes"
- Uses the FBPM dataset from "Fundamentals of Business Process Management"
