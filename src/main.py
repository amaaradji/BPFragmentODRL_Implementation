import json
from fragmenter import Fragmenter
from policy_generator import PolicyGenerator
from policy_checker import PolicyChecker
import policy_metrics

def main():
    # Load BPMN model
    with open("bpmn_model.json", "r") as f:
        bp_model = json.load(f)
    # Optionally load a high-level ODRL policy
    with open("process_policy.json", "r") as f:
        bp_policy = json.load(f)

    # Fragment the model
    fragmenter = Fragmenter(bp_model)
    fragments = fragmenter.fragment_process()

    # Generate policies, measuring time
    (result, gen_time) = policy_metrics.measure_generation_time(
        generator_func = lambda: PolicyGenerator(bp_model, bp_policy).generate_policies()
    )
    (activity_policies, dependency_policies) = result
    print(f"Policy generation took {gen_time:.4f} seconds.")

    # Check consistency
    checker = PolicyChecker(activity_policies, dependency_policies)
    conflict_count = checker.check_consistency()

    # Count rules
    total_rules = policy_metrics.count_all_rules(activity_policies, dependency_policies)

    # Print metrics
    print(f"Total ODRL rules: {total_rules}")
    print(f"Number of conflicts: {conflict_count}")

if __name__ == "__main__":
    main()
