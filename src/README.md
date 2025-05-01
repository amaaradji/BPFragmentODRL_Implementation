# BPFragmentODRL

A Python-based approach for fragmenting BPMN models and generating ODRL policies, featuring:

- **Fragmenter**: Splits BPMN processes into fragments.
- **PolicyGenerator**: Produces ODRL policies for each fragment and inter-fragment dependencies.
- **PolicyChecker**: Ensures no contradictory rules (intra-/inter-fragment).
- **PolicyMetrics**: Gathers metrics (time, complexity, conflict count).

## Requirements
- Python 3.7+
- `matplotlib` (optional, for plotting)

Install via:
pip install -r requirements.txt


## Usage
1. Place your BPMN model JSON in the same folder (e.g., `bpmn_model.json`).
2. Run `python main.py`.
3. Check console output and generated plots.

## Folder Structure
BPFragmentODRL/ 
├─ fragmenter.py 

├─ policy_generator.py 

├─ policy_checker.py 

├─ policy_metrics.py 

├─ main.py 

├─ dataset/ # sample BPMN files 

└─ README.md

## Contributing
- Fork this repo, then create a Pull Request (PR) for your changes.



