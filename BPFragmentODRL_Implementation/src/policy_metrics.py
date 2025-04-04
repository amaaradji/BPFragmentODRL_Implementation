"""
policy_metrics.py

Utilities to measure time, count rules, etc., for evaluating policy generation.
"""

import time

def measure_generation_time(generator_func, *args, **kwargs):
    """
    Measures the time taken by a policy generation (or any) function.

    :param generator_func: function to call
    :param args, kwargs: arguments to pass to that function
    :return: (result_of_func, elapsed_time_in_seconds)
    """
    start = time.time()
    result = generator_func(*args, **kwargs)
    end = time.time()
    elapsed = end - start
    return result, elapsed

def count_odrl_rules(policy_dict):
    """
    Counts how many top-level ODRL rules (permission, prohibition, obligation)
    are in a single policy dict.

    :param policy_dict: ODRL policy as a dictionary
    :return: integer
    """
    count = 0
    for rule_type in ("permission", "prohibition", "obligation"):
        if rule_type in policy_dict:
            # If it's a list, add its length
            count += len(policy_dict[rule_type])
    return count

def count_all_rules(activity_policies, dependency_policies):
    """
    Aggregates rule counts across all activity and dependency policies.
    :return: (int) total number of rules
    """
    total = 0
    for _, p in activity_policies.items():
        total += count_odrl_rules(p)
    for _, p_list in dependency_policies.items():
        for p in p_list:
            total += count_odrl_rules(p)
    return total
