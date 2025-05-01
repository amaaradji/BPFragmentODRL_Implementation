import os
import json
import policy_metrics
import matplotlib.pyplot as plt

from fragmenter import Fragmenter
from policy_generator import PolicyGenerator
from policy_checker import PolicyChecker

def run_experiment_on_model(model_path, global_policy_path=None):
    """
    Runs the entire pipeline on a single BPMN model (and optional global policy).
    Returns a dictionary of metrics: {
       "model_name": ...,
       "num_activities": ...,
       "num_flows": ...,
       "generation_time": ...,
       "total_rules": ...,
       "conflicts": ...
    }
    """
    with open(model_path, "r") as f:
        bp_model = json.load(f)

    # Count activities/flows
    num_activities = len(bp_model["activities"])
    num_flows = len(bp_model["flows"])

    if global_policy_path and os.path.exists(global_policy_path):
        with open(global_policy_path, "r") as pf:
            global_policy = json.load(pf)
    else:
        global_policy = None

    # 1) Fragment
    fragmenter = Fragmenter(bp_model)
    fragments = fragmenter.fragment_process()

    # 2) Policy Generation (measure time)
    (policy_data, gen_time) = policy_metrics.measure_generation_time(
        generator_func = lambda: PolicyGenerator(bp_model, global_policy).generate_policies()
    )
    (activity_policies, dependency_policies) = policy_data

    # 3) Consistency Check
    checker = PolicyChecker(activity_policies, dependency_policies)
    conflict_count = checker.check_consistency()

    # 4) Count rules
    total_rules = policy_metrics.count_all_rules(activity_policies, dependency_policies)

    # Build metrics dict
    return {
        "model_name": os.path.basename(model_path),
        "num_activities": num_activities,
        "num_flows": num_flows,
        "generation_time": gen_time,
        "total_rules": total_rules,
        "conflicts": conflict_count
    }

def main():
    dataset_dir = "dataset"
    bpmn_files = ["model_small.json", "model_medium.json", "model_large.json"]
    global_policy_file = "global_policy.json"  # optional

    results = []
    for bf in bpmn_files:
        full_path = os.path.join(dataset_dir, bf)
        metrics = run_experiment_on_model(full_path, os.path.join(dataset_dir, global_policy_file))
        results.append(metrics)

    # Print results to console
    print("=== RESULTS ===")
    for r in results:
        print(r)

    # Example: Plot # of activities vs total_rules in a line chart
    x_activities = [r["num_activities"] for r in results]
    y_rules = [r["total_rules"] for r in results]

    plt.plot(x_activities, y_rules, marker="o")
    plt.xlabel("Number of Activities")
    plt.ylabel("Total Policy Rules Generated")
    plt.title("Rules vs. Activities (Synthetic BPMN Models)")
    plt.xticks(x_activities)
    plt.grid(True)
    plt.show()

    # Another example: bar chart for conflicts
    model_names = [r["model_name"] for r in results]
    conflicts = [r["conflicts"] for r in results]

    plt.bar(model_names, conflicts)
    plt.xlabel("Model")
    plt.ylabel("Conflict Count")
    plt.title("Conflict Count by Model")
    plt.show()

if __name__ == "__main__":
    main()
