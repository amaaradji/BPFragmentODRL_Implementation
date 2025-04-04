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
        checker = PolicyConsistencyChecker(fragment_activity_policies, fragment_dependency_policies)
        intra_conflicts = checker.check_intra_fragment_consistency()
        inter_conflicts = checker.check_inter_fragment_consistency()
    """
    
    def __init__(self, fragment_activity_policies, fragment_dependency_policies, fragments=None, fragment_dependencies=None):
        """
        Initialize the consistency checker.
        
        :param fragment_activity_policies: dict of fragment activity policies (FPa)
        :param fragment_dependency_policies: dict of fragment dependency policies (FPd)
        :param fragments: optional list of fragment dicts for additional context
        :param fragment_dependencies: optional list of dependency dicts for additional context
        """
        self.fragment_activity_policies = fragment_activity_policies
        self.fragment_dependency_policies = fragment_dependency_policies
        self.fragments = fragments
        self.fragment_dependencies = fragment_dependencies
        
        # Initialize conflict containers
        self.intra_fragment_conflicts = []
        self.inter_fragment_conflicts = []
    
    def check_intra_fragment_consistency(self):
        """
        Check for conflicts within fragment policies (intra-fragment).
        
        Detects:
        - Permission/prohibition conflicts on the same activity/action
        - Contradictory constraints
        
        :return: list of conflict dicts
        """
        self.intra_fragment_conflicts = []
        
        # Check each fragment's activity policies
        for fragment_id, policies in self.fragment_activity_policies.items():
            # Check each activity's policy
            for activity_name, policy in policies.items():
                # Extract all rules
                permissions = policy.get('permission', [])
                prohibitions = policy.get('prohibition', [])
                
                # Check for permission/prohibition conflicts
                for permission in permissions:
                    perm_action = permission.get('action', '')
                    perm_target = permission.get('target', '')
                    
                    for prohibition in prohibitions:
                        prohib_action = prohibition.get('action', '')
                        prohib_target = prohibition.get('target', '')
                        
                        # If action and target match, we have a potential conflict
                        if perm_action == prohib_action and perm_target == prohib_target:
                            # Check if constraints make them compatible
                            if self._are_constraints_compatible(permission.get('constraint', []), 
                                                              prohibition.get('constraint', [])):
                                continue
                            
                            # Record the conflict
                            conflict = {
                                'type': 'permission_prohibition',
                                'fragment_id': fragment_id,
                                'activity': activity_name,
                                'action': perm_action,
                                'permission': permission,
                                'prohibition': prohibition,
                                'description': f"Permission and prohibition conflict for action '{perm_action}' on activity '{activity_name}'"
                            }
                            self.intra_fragment_conflicts.append(conflict)
                
                # Check for contradictory constraints within permissions
                self._check_contradictory_constraints(permissions, fragment_id, activity_name, 'permission')
                
                # Check for contradictory constraints within prohibitions
                self._check_contradictory_constraints(prohibitions, fragment_id, activity_name, 'prohibition')
        
        return self.intra_fragment_conflicts
    
    def check_inter_fragment_consistency(self):
        """
        Check for conflicts between fragment policies (inter-fragment).
        
        Detects:
        - Dependencies that violate policies of the target fragment
        - Unresolvable sequence/message flow policies
        
        :return: list of conflict dicts
        """
        self.inter_fragment_conflicts = []
        
        # Check each fragment dependency policy
        for dependency_key, policies in self.fragment_dependency_policies.items():
            # Parse the dependency key to get source and target fragments
            from_fragment, to_fragment = dependency_key.split('->')
            from_fragment_id = int(from_fragment)
            to_fragment_id = int(to_fragment)
            
            # Check if the target fragment has activity policies
            if to_fragment_id in self.fragment_activity_policies:
                target_fragment_policies = self.fragment_activity_policies[to_fragment_id]
                
                # Check each dependency policy
                for dependency_policy in policies:
                    # Extract permissions and prohibitions
                    permissions = dependency_policy.get('permission', [])
                    prohibitions = dependency_policy.get('prohibition', [])
                    
                    # Check if dependency permissions conflict with target fragment prohibitions
                    for permission in permissions:
                        perm_action = permission.get('action', '')
                        perm_target = permission.get('target', '')
                        
                        # Extract target fragment name from the target URI
                        target_fragment_match = re.search(r'fragment_(\d+)', perm_target)
                        if not target_fragment_match or int(target_fragment_match.group(1)) != to_fragment_id:
                            continue
                        
                        # Check against each activity in the target fragment
                        for activity_name, activity_policy in target_fragment_policies.items():
                            activity_prohibitions = activity_policy.get('prohibition', [])
                            
                            for prohibition in activity_prohibitions:
                                prohib_action = prohibition.get('action', '')
                                
                                # Check for conflicting actions
                                if self._are_actions_conflicting(perm_action, prohib_action):
                                    # Record the conflict
                                    conflict = {
                                        'type': 'dependency_violation',
                                        'from_fragment': from_fragment_id,
                                        'to_fragment': to_fragment_id,
                                        'activity': activity_name,
                                        'dependency_action': perm_action,
                                        'activity_action': prohib_action,
                                        'dependency_permission': permission,
                                        'activity_prohibition': prohibition,
                                        'description': f"Dependency permission '{perm_action}' conflicts with activity prohibition '{prohib_action}' in target fragment"
                                    }
                                    self.inter_fragment_conflicts.append(conflict)
            
            # Check for unresolvable sequence flow policies
            if self.fragment_dependencies:
                for dependency in self.fragment_dependencies:
                    if dependency.get('from_fragment') == from_fragment_id and dependency.get('to_fragment') == to_fragment_id:
                        dependency_type = dependency.get('type', 'sequence')
                        
                        # Check for conflicts in XOR dependencies
                        if dependency_type == 'xor_split':
                            self._check_xor_dependency_conflicts(dependency, policies)
                        
                        # Check for conflicts in AND dependencies
                        elif dependency_type == 'and_split':
                            self._check_and_dependency_conflicts(dependency, policies)
        
        return self.inter_fragment_conflicts
    
    def _check_contradictory_constraints(self, rules, fragment_id, activity_name, rule_type):
        """
        Check for contradictory constraints within a set of rules.
        
        :param rules: list of rules (permissions or prohibitions)
        :param fragment_id: ID of the fragment
        :param activity_name: Name of the activity
        :param rule_type: Type of rule ('permission' or 'prohibition')
        """
        for i, rule1 in enumerate(rules):
            for j in range(i+1, len(rules)):
                rule2 = rules[j]
                
                # Skip if actions are different
                if rule1.get('action', '') != rule2.get('action', ''):
                    continue
                
                # Check constraints for contradictions
                constraints1 = rule1.get('constraint', [])
                constraints2 = rule2.get('constraint', [])
                
                contradictions = self._find_constraint_contradictions(constraints1, constraints2)
                
                if contradictions:
                    # Record the conflict
                    conflict = {
                        'type': 'contradictory_constraints',
                        'fragment_id': fragment_id,
                        'activity': activity_name,
                        'rule_type': rule_type,
                        'action': rule1.get('action', ''),
                        'rule1': rule1,
                        'rule2': rule2,
                        'contradictions': contradictions,
                        'description': f"Contradictory constraints in {rule_type}s for action '{rule1.get('action', '')}' on activity '{activity_name}'"
                    }
                    self.intra_fragment_conflicts.append(conflict)
    
    def _find_constraint_contradictions(self, constraints1, constraints2):
        """
        Find contradictions between two sets of constraints.
        
        :param constraints1: First set of constraints
        :param constraints2: Second set of constraints
        :return: list of contradiction dicts
        """
        contradictions = []
        
        for constraint1 in constraints1:
            left_operand1 = constraint1.get('leftOperand', '')
            operator1 = constraint1.get('operator', '')
            right_operand1 = constraint1.get('rightOperand', '')
            
            for constraint2 in constraints2:
                left_operand2 = constraint2.get('leftOperand', '')
                operator2 = constraint2.get('operator', '')
                right_operand2 = constraint2.get('rightOperand', '')
                
                # Skip if different left operands
                if left_operand1 != left_operand2:
                    continue
                
                # Check for contradictions based on operator and right operand
                if self._are_operators_contradictory(operator1, operator2, right_operand1, right_operand2):
                    contradiction = {
                        'left_operand': left_operand1,
                        'constraint1': {
                            'operator': operator1,
                            'right_operand': right_operand1
                        },
                        'constraint2': {
                            'operator': operator2,
                            'right_operand': right_operand2
                        }
                    }
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _are_operators_contradictory(self, op1, op2, val1, val2):
        """
        Check if two operators with their values are contradictory.
        
        :param op1: First operator
        :param op2: Second operator
        :param val1: First value
        :param val2: Second value
        :return: True if contradictory, False otherwise
        """
        # Handle date comparisons
        if self._is_date(val1) and self._is_date(val2):
            date1 = self._parse_date(val1)
            date2 = self._parse_date(val2)
            
            if date1 and date2:
                # Check for date-based contradictions
                if op1 == 'lt' and op2 == 'gt' and date1 <= date2:
                    return True
                if op1 == 'gt' and op2 == 'lt' and date1 >= date2:
                    return True
                if op1 == 'lteq' and op2 == 'gt' and date1 < date2:
                    return True
                if op1 == 'gteq' and op2 == 'lt' and date1 > date2:
                    return True
                if op1 == 'eq' and op2 == 'neq' and date1 == date2:
                    return True
        
        # Handle numeric comparisons
        if self._is_numeric(val1) and self._is_numeric(val2):
            num1 = float(val1)
            num2 = float(val2)
            
            # Check for numeric contradictions
            if op1 == 'lt' and op2 == 'gt' and num1 <= num2:
                return True
            if op1 == 'gt' and op2 == 'lt' and num1 >= num2:
                return True
            if op1 == 'lteq' and op2 == 'gt' and num1 < num2:
                return True
            if op1 == 'gteq' and op2 == 'lt' and num1 > num2:
                return True
            if op1 == 'eq' and op2 == 'neq' and num1 == num2:
                return True
        
        # Handle string comparisons
        if op1 == 'eq' and op2 == 'eq' and val1 != val2:
            return True
        if op1 == 'eq' and op2 == 'neq' and val1 == val2:
            return True
        
        # Handle direct contradictions
        contradictory_pairs = [
            ('eq', 'neq'),
            ('lt', 'gteq'),
            ('gt', 'lteq')
        ]
        
        if (op1, op2) in contradictory_pairs and val1 == val2:
            return True
        
        return False
    
    def _are_constraints_compatible(self, constraints1, constraints2):
        """
        Check if two sets of constraints are compatible (non-contradictory).
        
        :param constraints1: First set of constraints
        :param constraints2: Second set of constraints
        :return: True if compatible, False if contradictory
        """
        # If either set is empty, they're compatible
        if not constraints1 or not constraints2:
            return False
        
        # Check each constraint pair for contradictions
        contradictions = self._find_constraint_contradictions(constraints1, constraints2)
        
        # If no contradictions found, constraints are compatible
        return len(contradictions) == 0
    
    def _are_actions_conflicting(self, action1, action2):
        """
        Check if two actions are conflicting.
        
        :param action1: First action
        :param action2: Second action
        :return: True if conflicting, False otherwise
        """
        # Direct conflicts
        if action1 == action2:
            return True
        
        # Known conflicting action pairs
        conflicting_pairs = [
            ('execute', 'skip'),
            ('enable', 'disable'),
            ('select', 'skip'),
            ('execute_parallel', 'skip'),
            ('read', 'hide'),
            ('modify', 'lock'),
            ('approve', 'reject')
        ]
        
        if (action1, action2) in conflicting_pairs or (action2, action1) in conflicting_pairs:
            return True
        
        return False
    
    def _check_xor_dependency_conflicts(self, dependency, policies):
        """
        Check for conflicts in XOR dependencies.
        
        :param dependency: Dependency dict
        :param policies: List of policies for this dependency
        """
        from_fragment = dependency.get('from_fragment')
        to_fragment = dependency.get('to_fragment')
        gateway = dependency.get('gateway')
        
        # Check for conflicting conditions in XOR policies
        conditions = set()
        
        for policy in policies:
            permissions = policy.get('permission', [])
            
            for permission in permissions:
                if permission.get('action') == 'select':
                    constraints = permission.get('constraint', [])
                    
                    for constraint in constraints:
                        if constraint.get('leftOperand') == 'condition':
                            condition = constraint.get('rightOperand', '')
                            
                            if condition in conditions:
                                # Record the conflict
                                conflict = {
                                    'type': 'xor_condition_conflict',
                                    'from_fragment': from_fragment,
                                    'to_fragment': to_fragment,
                                    'gateway': gateway,
                                    'condition': condition,
                                    'policy': policy,
                                    'description': f"Duplicate condition '{condition}' in XOR gateway policies"
                                }
                                self.inter_fragment_conflicts.append(conflict)
                            else:
                                conditions.add(condition)
    
    def _check_and_dependency_conflicts(self, dependency, policies):
        """
        Check for conflicts in AND dependencies.
        
        :param dependency: Dependency dict
        :param policies: List of policies for this dependency
        """
        from_fragment = dependency.get('from_fragment')
        to_fragment = dependency.get('to_fragment')
        gateway = dependency.get('gateway')
        
        # Check for skip permissions in AND policies (which should not exist)
        for policy in policies:
            permissions = policy.get('permission', [])
            
            for permission in permissions:
                if permission.get('action') == 'skip':
                    # Record the conflict
                    conflict = {
                        'type': 'and_skip_conflict',
                        'from_fragment': from_fragment,
                        'to_fragment': to_fragment,
                        'gateway': gateway,
                        'policy': policy,
                        'description': f"Skip permission in AND gateway policy, which violates parallel execution requirement"
                    }
                    self.inter_fragment_conflicts.append(conflict)
    
    def _is_date(self, value):
        """Check if a value is a date string."""
        if not isinstance(value, str):
            return False
        
        # Check common date formats
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}'   # DD-MM-YYYY
        ]
        
        return any(re.fullmatch(pattern, value) for pattern in date_patterns)
    
    def _parse_date(self, date_str):
        """Parse a date string into a datetime object."""
        try:
            # Try common formats
            formats = [
                '%Y-%m-%d',  # YYYY-MM-DD
                '%m/%d/%Y',  # MM/DD/YYYY
                '%d-%m-%Y'   # DD-MM-YYYY
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            return None
        except:
            return None
    
    def _is_numeric(self, value):
        """Check if a value is numeric."""
        if isinstance(value, (int, float)):
            return True
        
        if isinstance(value, str):
            try:
                float(value)
                return True
            except ValueError:
                return False
        
        return False
    
    def get_conflict_metrics(self):
        """
        Calculate metrics for detected conflicts.
        
        :return: dict with conflict metrics
        """
        metrics = {
            'intra_fragment': {
                'total': len(self.intra_fragment_conflicts),
                'by_type': {}
            },
            'inter_fragment': {
                'total': len(self.inter_fragment_conflicts),
                'by_type': {}
            },
            'total_conflicts': len(self.intra_fragment_conflicts) + len(self.inter_fragment_conflicts)
        }
        
        # Count intra-fragment conflicts by type
        for conflict in self.intra_fragment_conflicts:
            conflict_type = conflict.get('type', 'unknown')
            metrics['intra_fragment']['by_type'][conflict_type] = metrics['intra_fragment']['by_type'].get(conflict_type, 0) + 1
        
        # Count inter-fragment conflicts by type
        for conflict in self.inter_fragment_conflicts:
            conflict_type = conflict.get('type', 'unknown')
            metrics['inter_fragment']['by_type'][conflict_type] = metrics['inter_fragment']['by_type'].get(conflict_type, 0) + 1
        
        return metrics
    
    def save_conflicts(self, output_dir):
        """
        Save detected conflicts to JSON files.
        
        :param output_dir: Directory to save the conflict files
        :return: dict with paths to saved files
        """
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save intra-fragment conflicts
        if self.intra_fragment_conflicts:
            intra_file = os.path.join(output_dir, "intra_fragment_conflicts.json")
            with open(intra_file, 'w') as f:
                json.dump(self.intra_fragment_conflicts, f, indent=2)
            saved_files['intra_fragment'] = intra_file
        
        # Save inter-fragment conflicts
        if self.inter_fragment_conflicts:
            inter_file = os.path.join(output_dir, "inter_fragment_conflicts.json")
            with open(inter_file, 'w') as f:
                json.dump(self.inter_fragment_conflicts, f, indent=2)
            saved_files['inter_fragment'] = inter_file
        
        # Save metrics
        metrics = self.get_conflict_metrics()
        metrics_file = os.path.join(output_dir, "conflict_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        saved_files['metrics'] = metrics_file
        
        return saved_files
