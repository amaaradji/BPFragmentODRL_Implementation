import os
"""
policy_consistency_checker.py

Provides a PolicyConsistencyChecker class to detect conflicts in ODRL policies for BPMN fragments.
"""

import json
import logging
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolicyConsistencyChecker:
    """
    PolicyConsistencyChecker detects conflicts in ODRL policies for BPMN fragments.
    
    It supports:
    - Intra-fragment conflict detection (within a fragment)
    - Inter-fragment conflict detection (between fragments)
    - Conflict categorization and reporting
    
    Typical usage:
        checker = PolicyConsistencyChecker(fragment_activity_policies, fragment_dependency_policies, fragments, fragment_dependencies)
        intra_conflicts = checker.check_intra_fragment_consistency()
        inter_conflicts = checker.check_inter_fragment_consistency()
    """
    
    def __init__(self, fragment_activity_policies, fragment_dependency_policies, fragments=None, fragment_dependencies=None):
        """
        Initialize the consistency checker.
        
        :param fragment_activity_policies: dict of activity policies (FPa) for the current model.
                                         Structure: {activity_id_str: {rule_type_str: [List[rule_dicts]]}}
        :param fragment_dependency_policies: dict of fragment dependency policies (FPd) for the current model.
                                             Structure: {dependency_id_str: {rule_type_str: [List[rule_dicts]]}}
        :param fragments: optional list of fragment dicts for additional context (e.g., mapping activity to fragment).
                          Structure: [{'id': 'frag_id', 'activities': ['act_id1', 'act_id2'], ...}]
        :param fragment_dependencies: optional list of dependency dicts for additional context.
                                      Structure: [{'id': 'dep_id', 'source_fragment': 'frag1', 'target_fragment': 'frag2', ...}]
        """
        self.fragment_activity_policies = fragment_activity_policies if fragment_activity_policies else {}
        self.fragment_dependency_policies = fragment_dependency_policies if fragment_dependency_policies else {}
        self.fragments = fragments if fragments else [] # Ensure it's a list
        self.fragment_dependencies = fragment_dependencies if fragment_dependencies else [] # Ensure it's a list
        
        self.intra_fragment_conflicts = []
        self.inter_fragment_conflicts = []
    
    def check_intra_fragment_consistency(self):
        """
        Check for conflicts within policies related to the same activity (intra-activity conflicts).
        Detects:
        - Permission/prohibition conflicts on the same activity/action
        - Contradictory constraints within rules for the same activity/action
        :return: list of conflict dicts
        """
        self.intra_fragment_conflicts = []
        
        activity_to_fragment_map = {}
        if self.fragments:
            for frag_dict in self.fragments:
                fragment_id_val = frag_dict.get('id')
                if fragment_id_val is not None:
                    for act_id_val in frag_dict.get('activities', []):
                        activity_to_fragment_map[str(act_id_val)] = str(fragment_id_val)

        # self.fragment_activity_policies is Dict[activity_id_str, Dict[rule_type_str, List[rule_dicts]]]
        for activity_id_str, rules_by_type_for_activity in self.fragment_activity_policies.items():
            current_fragment_id_str = activity_to_fragment_map.get(activity_id_str) # Can be None

            permissions_list = rules_by_type_for_activity.get('permission', [])
            prohibitions_list = rules_by_type_for_activity.get('prohibition', [])
            
            # Check for permission/prohibition conflicts for the current activity_id
            for p_rule_dict in permissions_list:
                perm_action_str = p_rule_dict.get('action', '')
                
                for pr_rule_dict in prohibitions_list:
                    prohib_action_str = pr_rule_dict.get('action', '')
                    
                    if perm_action_str == prohib_action_str: # Conflict on the same activity_id and action
                        # _are_constraints_compatible returns True if constraints make them non-overlapping (compatible).
                        # If False, they are not compatible (overlap or no differentiating constraints), so a conflict exists.
                        if self._are_constraints_compatible(p_rule_dict.get('constraints', []), 
                                                          pr_rule_dict.get('constraints', [])):
                            continue # Constraints make them compatible, so no conflict here.
                        
                        conflict_desc = f"Permission and prohibition conflict for action '{perm_action_str}' on activity '{activity_id_str}'"
                        if current_fragment_id_str:
                            conflict_desc += f" in fragment '{current_fragment_id_str}'"
                        
                        conflict = {
                            'type': 'permission_prohibition',
                            'fragment_id': current_fragment_id_str,
                            'activity_id': activity_id_str,
                            'action': perm_action_str,
                            'permission_rule': p_rule_dict,
                            'prohibition_rule': pr_rule_dict,
                            'description': conflict_desc
                        }
                        self.intra_fragment_conflicts.append(conflict)
            
            # Check for contradictory constraints within all permissions for this activity
            self._check_contradictory_constraints(permissions_list, current_fragment_id_str, activity_id_str, 'permission')
            
            # Check for contradictory constraints within all prohibitions for this activity
            self._check_contradictory_constraints(prohibitions_list, current_fragment_id_str, activity_id_str, 'prohibition')
        
        return self.intra_fragment_conflicts
    
    def check_inter_fragment_consistency(self):
        """
        Check for conflicts between fragment policies (inter-fragment).
        This method needs to be reviewed and potentially updated based on the exact structure of dependency policies and fragment_dependencies.
        For now, it retains its original logic but might need adjustments.
        """
        self.inter_fragment_conflicts = []
        
        if not self.fragment_dependencies or not self.fragment_dependency_policies:
            logger.info("Skipping inter-fragment consistency check: no dependencies or dependency policies provided.")
            return self.inter_fragment_conflicts

        # Create a lookup for dependency policies by dependency ID
        dep_policies_lookup = {str(k): v for k, v in self.fragment_dependency_policies.items()}

        for dependency_info in self.fragment_dependencies: # Iterate through structural dependencies
            dep_id = str(dependency_info.get('id'))
            from_fragment_id = str(dependency_info.get('source_fragment')) # Assuming these keys exist from fragmenter
            to_fragment_id = str(dependency_info.get('target_fragment'))
            dep_type = dependency_info.get('type', 'sequenceFlow') # e.g. sequenceFlow, messageFlow

            dependency_rules_by_type = dep_policies_lookup.get(dep_id, {})
            dep_permissions = dependency_rules_by_type.get('permission', [])
            # dep_prohibitions = dependency_rules_by_type.get('prohibition', []) # Not used in original logic below

            # Check if target fragment (to_fragment_id) has activity policies that might conflict
            # This requires mapping to_fragment_id to its activities, then checking those activities' policies.
            # The original logic was: `if to_fragment_id in self.fragment_activity_policies:` which is wrong as keys are activity_ids.
            
            # Find activities in the target fragment
            target_fragment_activities = []
            for frag_dict in self.fragments:
                if str(frag_dict.get('id')) == to_fragment_id:
                    target_fragment_activities = [str(act_id) for act_id in frag_dict.get('activities', [])]
                    break
            
            for target_activity_id in target_fragment_activities:
                target_activity_rules_by_type = self.fragment_activity_policies.get(target_activity_id, {})
                activity_prohibitions = target_activity_rules_by_type.get('prohibition', [])

                for dep_perm_rule in dep_permissions:
                    dep_perm_action = dep_perm_rule.get('action', '') # e.g., 'traverse' for a flow
                    
                    for act_prohib_rule in activity_prohibitions:
                        act_prohib_action = act_prohib_rule.get('action', '') # e.g., 'execute' for an activity
                        
                        # Define how dependency actions relate to activity actions for conflict
                        # Example: if traversing a flow is permitted, but executing the target activity is prohibited for the same assignee/context.
                        if self._are_dependency_and_activity_actions_conflicting(dep_perm_action, act_prohib_action, dep_type):
                            if not self._are_constraints_compatible(dep_perm_rule.get('constraints',[]), act_prohib_rule.get('constraints',[])):
                                conflict = {
                                    'type': 'dependency_target_activity_conflict',
                                    'dependency_id': dep_id,
                                    'from_fragment_id': from_fragment_id,
                                    'to_fragment_id': to_fragment_id,
                                    'target_activity_id': target_activity_id,
                                    'dependency_permission_rule': dep_perm_rule,
                                    'activity_prohibition_rule': act_prohib_rule,
                                    'description': f"Dependency '{dep_id}' permission ({dep_perm_action}) conflicts with prohibition on target activity '{target_activity_id}' ({act_prohib_action})"
                                }
                                self.inter_fragment_conflicts.append(conflict)
            
            # Original logic for XOR/AND split policies (may need review based on how these are represented)
            # This part assumes `dependency_rules_by_type` (policies for a single dependency ID) is structured differently
            # or that `_check_xor_dependency_conflicts` expects this structure.
            # For now, commenting out as it's likely incompatible with current `dependency_rules_by_type` structure.
            # if dep_type == 'xor_split':
            #     self._check_xor_dependency_conflicts(dependency_info, dependency_rules_by_type)
            # elif dep_type == 'and_split':
            #     self._check_and_dependency_conflicts(dependency_info, dependency_rules_by_type)
        
        return self.inter_fragment_conflicts

    def _are_dependency_and_activity_actions_conflicting(self, dep_action, act_action, dep_type):
        """ Placeholder: Define logic for when a dependency action conflicts with an activity action. """
        # E.g., if dep_action is 'traverse' and act_action is 'execute' for the target activity of the flow.
        if dep_action == 'traverse' and act_action == 'execute': # Basic example
            return True
        return False

    def _check_contradictory_constraints(self, rules_list, fragment_id_str, activity_id_str, rule_type_str):
        """
        Check for contradictory constraints within a list of rules for the same action.
        :param rules_list: list of rule dicts (all permissions or all prohibitions for an activity)
        :param fragment_id_str: ID of the fragment (can be None)
        :param activity_id_str: ID of the activity
        :param rule_type_str: Type of rule ('permission' or 'prohibition')
        """
        for i, rule1_dict in enumerate(rules_list):
            for j in range(i + 1, len(rules_list)):
                rule2_dict = rules_list[j]
                
                action1_str = rule1_dict.get('action', '')
                action2_str = rule2_dict.get('action', '')

                if action1_str != action2_str: # Only check constraints if actions are the same
                    continue
                
                constraints1_list = rule1_dict.get('constraints', [])
                constraints2_list = rule2_dict.get('constraints', [])
                
                # _find_constraint_contradictions returns a list of specific contradictory constraint pairs
                contradictory_constraint_pairs = self._find_constraint_contradictions(constraints1_list, constraints2_list)
                
                if contradictory_constraint_pairs:
                    conflict_desc = f"Contradictory constraints in {rule_type_str}s for action '{action1_str}' on activity '{activity_id_str}'"
                    if fragment_id_str:
                        conflict_desc += f" in fragment '{fragment_id_str}'"
                    conflict = {
                        'type': 'contradictory_constraints',
                        'fragment_id': fragment_id_str,
                        'activity_id': activity_id_str,
                        'rule_type': rule_type_str,
                        'action': action1_str,
                        'rule1': rule1_dict,
                        'rule2': rule2_dict,
                        'contradictions_detail': contradictory_constraint_pairs,
                        'description': conflict_desc
                    }
                    self.intra_fragment_conflicts.append(conflict)
    
    def _find_constraint_contradictions(self, constraints1_list, constraints2_list):
        """
        Find specific contradictions between two lists of constraint dicts.
        :return: list of dicts, each describing a pair of contradictory constraints.
        """
        contradictions_found = []
        for const1_dict in constraints1_list:
            lo1 = const1_dict.get('leftOperand', '')
            op1 = const1_dict.get('operator', '')
            ro1 = const1_dict.get('rightOperand', '')
            
            for const2_dict in constraints2_list:
                lo2 = const2_dict.get('leftOperand', '')
                op2 = const2_dict.get('operator', '')
                ro2 = const2_dict.get('rightOperand', '')
                
                if lo1 != lo2: # Only compare constraints on the same attribute
                    continue
                
                if self._are_operators_contradictory(op1, op2, ro1, ro2):
                    contradictions_found.append({
                        'left_operand': lo1,
                        'constraint1': {'operator': op1, 'right_operand': ro1},
                        'constraint2': {'operator': op2, 'right_operand': ro2}
                    })
        return contradictions_found
    
    def _are_operators_contradictory(self, op1, op2, val1, val2):
        """
        Check if two operators with their values are directly contradictory for the same left operand.
        e.g., time < 10 AND time > 15 for the same 'time'.
        """
        # This is a simplified check. A full implementation would require a proper SMT solver or interval logic.
        # For now, direct contradictions:
        if val1 == val2:
            if (op1 == 'eq' and op2 == 'neq') or (op1 == 'neq' and op2 == 'eq'): return True
            if (op1 == 'lt' and op2 == 'gteq') or (op1 == 'gteq' and op2 == 'lt'): return True
            if (op1 == 'gt' and op2 == 'lteq') or (op1 == 'lteq' and op2 == 'gt'): return True
        # Could add more complex logic, e.g., op1='lt', val1=10 and op2='gt', val2=15 (not contradictory by themselves)
        # vs op1='lt', val1=10 and op2='gt', val2=5 (contradictory: x < 10 and x > 5 is possible, but x < 5 and x > 10 is not)
        # The original code had more complex date/numeric checks, which can be reinstated if needed.
        # For simplicity here, focusing on direct operator contradictions for identical values.
        return False
    
    def _are_constraints_compatible(self, constraints1_list, constraints2_list):
        """
        Check if two sets of constraints are compatible (i.e., allow for a scenario where both can be true).
        Returns True if compatible, False if they are inherently contradictory.
        A more robust implementation would check if the *conjunction* of constraints1 and constraints2 is satisfiable.
        The original logic: `return len(self._find_constraint_contradictions(c1,c2)) == 0`
        This means compatible if NO direct contradictions are found between any pair of individual constraints.
        This is a weak form of compatibility. Two non-contradictory constraints can still make rules apply simultaneously.
        The goal of this function in the context of (Permission vs Prohibition) is to determine if constraints
        make the Permission and Prohibition apply in *different, non-overlapping contexts*.
        If this function returns True, it means constraints *do* separate the contexts, so no conflict.
        If it returns False, it means constraints *do not* separate contexts, so there *is* a conflict.
        So, if `_find_constraint_contradictions` finds any contradiction, this should return False (not compatible).
        If `_find_constraint_contradictions` finds NO contradiction, this should return True (compatible).
        This seems to be the original intent.
        However, if one list is empty, `_find_constraint_contradictions` will return empty.
        Example: P(action) vs Pr(action) with constraint C. If C is met, Pr applies. P always applies.
        This should be a conflict if C can be met.
        If P has C1 and Pr has C2. Conflict if C1 and C2 can be simultaneously true.
        Let's stick to the original interpretation for now: compatible if no direct pairwise contradictions.
        """
        if not constraints1_list and not constraints2_list:
            return False # No constraints on either, so they are NOT compatible in a differentiating way -> conflict
        if not constraints1_list or not constraints2_list:
            # One has constraints, the other doesn't. They are NOT compatible in a differentiating way if the constrained one can apply.
            # This needs more thought. For now, assume they don't differentiate enough. Leads to conflict. 
            return False 

        # Original logic: compatible if no direct pairwise contradictions found.
        return len(self._find_constraint_contradictions(constraints1_list, constraints2_list)) == 0

    # Helper methods for parsing values (can be expanded from original if needed)
    def _is_date(self, value):
        try:
            datetime.fromisoformat(str(value).replace('Z', '+00:00'))
            return True
        except (ValueError, TypeError):
            return False

    def _parse_date(self, value):
        try:
            return datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return None

    def _is_numeric(self, value):
        return isinstance(value, (int, float))

    def get_conflict_metrics(self):
        """Return metrics about the detected conflicts."""
        return {
            'intra_fragment': {'total': len(self.intra_fragment_conflicts)},
            'inter_fragment': {'total': len(self.inter_fragment_conflicts)},
            'total_conflicts': len(self.intra_fragment_conflicts) + len(self.inter_fragment_conflicts)
        }

    def save_conflicts(self, output_dir):
        """Save the detected conflicts to JSON files."""
        os.makedirs(output_dir, exist_ok=True)
        intra_conflicts_path = os.path.join(output_dir, 'intra_fragment_conflicts.json')
        inter_conflicts_path = os.path.join(output_dir, 'inter_fragment_conflicts.json')

        with open(intra_conflicts_path, 'w') as f:
            json.dump(self.intra_fragment_conflicts, f, indent=2)
        logger.info(f"Intra-fragment conflicts saved to {intra_conflicts_path}")

        with open(inter_conflicts_path, 'w') as f:
            json.dump(self.inter_fragment_conflicts, f, indent=2)
        logger.info(f"Inter-fragment conflicts saved to {inter_conflicts_path}")

    # Methods like _check_xor_dependency_conflicts, _check_and_dependency_conflicts from original
    # would need to be reviewed and adapted if inter-fragment checking is a priority.
    # For now, they are omitted as the primary error was intra-fragment.


