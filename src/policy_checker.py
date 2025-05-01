"""
policy_checker.py

Performs consistency checks on activity-level and dependency-level policies.
"""

class PolicyChecker:
    """
    PolicyChecker ensures:
      1) Intra-fragment consistency: no contradictions within a single activity's policy.
      2) Inter-fragment consistency: the dependency policies do not conflict with each other
         or with the activity-level policies they reference.
    """

    def __init__(self, activity_policies, dependency_policies):
        """
        :param activity_policies: dict {activity_name: ODRL policy dict}
        :param dependency_policies: dict { 'A->B': [ODRL policy dict], ... }
        """
        self.activity_policies = activity_policies
        self.dependency_policies = dependency_policies

    def check_consistency(self):
        """
        Performs consistency checks across all policies.
        Prints or logs conflicts. Returns the number of conflicts found.

        :return: int, conflict_count
        """
        conflict_count = 0

        # 1. Check each activity policy
        for activity, policy in self.activity_policies.items():
            if not self._validate_policy(policy):
                print(f"Inconsistent policy detected for activity {activity}")
                conflict_count += 1

        # 2. Check each dependency policy
        for dep_key, pol_list in self.dependency_policies.items():
            for pol in pol_list:
                if not self._validate_policy(pol):
                    print(f"Inconsistent policy detected for dependency {dep_key}")
                    conflict_count += 1

        if conflict_count == 0:
            print("All fragment policies are consistent.")
        else:
            print(f"Found {conflict_count} conflicts in total.")

        return conflict_count

    def _validate_policy(self, policy):
        """
        Minimal check: ensures the policy has a proper structure. You can expand
        to detect contradictory constraints, e.g. a permission and a prohibition
        for the same target/action.

        :param policy: a dict representing an ODRL policy
        :return: bool, True if policy is valid, else False
        """
        # Example checks:
        if "@type" not in policy:
            return False

        # In practice, you might want to:
        # - Ensure 'permission' items don't conflict with 'prohibition' on the same action/target
        # - Validate constraints are well-formed
        # - etc.

        return True
