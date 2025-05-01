"""
policy_reconstructor.py

Provides a PolicyReconstructor class to recombine fragment policies into a complete policy set.
"""

import json
import logging
import copy
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolicyReconstructor:
    """
    PolicyReconstructor recombines fragment policies into a complete policy set.
    
    It supports:
    - Recombining Fragment Activity Policies (FPa)
    - Recombining Fragment Dependency Policies (FPd)
    - Comparing reconstructed policies to original BP-level policies
    - Reporting on reconstruction accuracy
    
    Typical usage:
        reconstructor = PolicyReconstructor(fragment_activity_policies, fragment_dependency_policies, original_bp_policy)
        reconstructed_policy = reconstructor.reconstruct_policy()
        accuracy = reconstructor.evaluate_reconstruction()
    """
    
    def __init__(self, fragment_activity_policies, fragment_dependency_policies, original_bp_policy=None, fragments=None):
        """
        Initialize the policy reconstructor.
        
        :param fragment_activity_policies: dict of fragment activity policies (FPa)
        :param fragment_dependency_policies: dict of fragment dependency policies (FPd)
        :param original_bp_policy: optional original BP-level policy for comparison
        :param fragments: optional list of fragment dicts for additional context
        """
        self.fragment_activity_policies = fragment_activity_policies
        self.fragment_dependency_policies = fragment_dependency_policies
        self.original_bp_policy = original_bp_policy
        self.fragments = fragments
        
        # Initialize containers
        self.reconstructed_policy = None
        self.reconstruction_report = None
    
    def reconstruct_policy(self):
        """
        Recombine fragment policies into a complete policy set.
        
        :return: dict representing the reconstructed ODRL policy
        """
        # Create a skeleton for the reconstructed policy
        self.reconstructed_policy = {
            "@context": "http://www.w3.org/ns/odrl.jsonld",
            "uid": "http://example.com/policy:reconstructed",
            "@type": "Set",
            "permission": [],
            "prohibition": [],
            "obligation": []
        }
        
        # Process fragment activity policies (FPa)
        self._process_fragment_activity_policies()
        
        # Process fragment dependency policies (FPd)
        self._process_fragment_dependency_policies()
        
        # Remove duplicates
        self._remove_duplicate_rules()
        
        return self.reconstructed_policy
    
    def _process_fragment_activity_policies(self):
        """Process and merge fragment activity policies (FPa)."""
        # Track processed activities to avoid duplicates
        processed_activities = set()
        
        # Process each fragment's activity policies
        for fragment_id, policies in self.fragment_activity_policies.items():
            # Process each activity's policy
            for activity_name, policy in policies.items():
                # Skip if already processed
                if activity_name in processed_activities:
                    continue
                
                processed_activities.add(activity_name)
                
                # Extract rules
                for rule_type in ['permission', 'prohibition', 'obligation']:
                    if rule_type in policy:
                        for rule in policy[rule_type]:
                            # Create a copy of the rule
                            rule_copy = copy.deepcopy(rule)
                            
                            # Update the rule UID to avoid conflicts
                            if 'uid' in rule_copy:
                                rule_copy['uid'] = f"{rule_copy['uid']}_reconstructed"
                            
                            # Add to reconstructed policy
                            self.reconstructed_policy[rule_type].append(rule_copy)
    
    def _process_fragment_dependency_policies(self):
        """Process and merge fragment dependency policies (FPd)."""
        # Process each dependency's policies
        for dependency_key, policies in self.fragment_dependency_policies.items():
            # Parse the dependency key
            from_fragment, to_fragment = dependency_key.split('->')
            
            # Process each policy
            for policy in policies:
                # Extract rules
                for rule_type in ['permission', 'prohibition', 'obligation']:
                    if rule_type in policy:
                        for rule in policy[rule_type]:
                            # Create a copy of the rule
                            rule_copy = copy.deepcopy(rule)
                            
                            # Update the rule UID to avoid conflicts
                            if 'uid' in rule_copy:
                                rule_copy['uid'] = f"{rule_copy['uid']}_reconstructed"
                            
                            # Transform fragment-specific targets to BP-level targets
                            if 'target' in rule_copy:
                                rule_copy['target'] = self._transform_target(rule_copy['target'])
                            
                            # Add to reconstructed policy
                            self.reconstructed_policy[rule_type].append(rule_copy)
    
    def _transform_target(self, target):
        """
        Transform fragment-specific targets to BP-level targets.
        
        :param target: Original target URI
        :return: Transformed target URI
        """
        # Check if this is a fragment-specific target
        import re
        fragment_match = re.search(r'fragment_(\d+)', target)
        
        if fragment_match:
            # Extract the fragment ID
            fragment_id = int(fragment_match.group(1))
            
            # If we have fragment information, use it to get the actual activity name
            if self.fragments and fragment_id < len(self.fragments):
                fragment = self.fragments[fragment_id]
                
                # If there's only one activity in the fragment, use it as the target
                if len(fragment.get('activities', [])) == 1:
                    activity_name = fragment['activities'][0]
                    return f"http://example.com/asset:{activity_name}"
            
            # If we can't determine the specific activity, use a generic BP-level target
            return f"http://example.com/asset:bp_level_activity"
        
        # If not a fragment-specific target, return as is
        return target
    
    def _remove_duplicate_rules(self):
        """Remove duplicate rules from the reconstructed policy."""
        for rule_type in ['permission', 'prohibition', 'obligation']:
            # Use a set to track unique rule signatures
            unique_rules = set()
            filtered_rules = []
            
            for rule in self.reconstructed_policy[rule_type]:
                # Create a signature for the rule (excluding uid)
                rule_copy = copy.deepcopy(rule)
                if 'uid' in rule_copy:
                    del rule_copy['uid']
                
                # Convert to a hashable representation
                rule_signature = json.dumps(rule_copy, sort_keys=True)
                
                # Add to filtered list if not a duplicate
                if rule_signature not in unique_rules:
                    unique_rules.add(rule_signature)
                    filtered_rules.append(rule)
            
            # Update the policy with filtered rules
            self.reconstructed_policy[rule_type] = filtered_rules
    
    def evaluate_reconstruction(self):
        """
        Evaluate the accuracy of the policy reconstruction by comparing to the original BP-level policy.
        
        :return: dict with reconstruction evaluation metrics
        """
        # Ensure we have a reconstructed policy
        if not self.reconstructed_policy:
            self.reconstruct_policy()
        
        # Initialize the report
        self.reconstruction_report = {
            'total_rules': {
                'original': 0,
                'reconstructed': 0
            },
            'rule_types': {
                'permission': {'original': 0, 'reconstructed': 0, 'matched': 0},
                'prohibition': {'original': 0, 'reconstructed': 0, 'matched': 0},
                'obligation': {'original': 0, 'reconstructed': 0, 'matched': 0}
            },
            'matched_rules': [],
            'lost_rules': [],
            'new_rules': [],
            'accuracy': 0.0
        }
        
        # If no original policy, we can only report on the reconstructed policy
        if not self.original_bp_policy:
            for rule_type in ['permission', 'prohibition', 'obligation']:
                if rule_type in self.reconstructed_policy:
                    count = len(self.reconstructed_policy[rule_type])
                    self.reconstruction_report['rule_types'][rule_type]['reconstructed'] = count
                    self.reconstruction_report['total_rules']['reconstructed'] += count
            
            # Without an original policy, we can't calculate accuracy
            self.reconstruction_report['accuracy'] = None
            return self.reconstruction_report
        
        # Count rules in original policy
        for rule_type in ['permission', 'prohibition', 'obligation']:
            if rule_type in self.original_bp_policy:
                count = len(self.original_bp_policy[rule_type])
                self.reconstruction_report['rule_types'][rule_type]['original'] = count
                self.reconstruction_report['total_rules']['original'] += count
        
        # Count rules in reconstructed policy
        for rule_type in ['permission', 'prohibition', 'obligation']:
            if rule_type in self.reconstructed_policy:
                count = len(self.reconstructed_policy[rule_type])
                self.reconstruction_report['rule_types'][rule_type]['reconstructed'] = count
                self.reconstruction_report['total_rules']['reconstructed'] += count
        
        # Compare rules
        for rule_type in ['permission', 'prohibition', 'obligation']:
            # Skip if not in both policies
            if rule_type not in self.original_bp_policy or rule_type not in self.reconstructed_policy:
                continue
            
            # Track which original rules have been matched
            matched_original = set()
            
            # For each reconstructed rule, try to find a match in the original
            for recon_rule in self.reconstructed_policy[rule_type]:
                matched = False
                
                for i, orig_rule in enumerate(self.original_bp_policy[rule_type]):
                    if i in matched_original:
                        continue
                    
                    if self._rules_match(orig_rule, recon_rule):
                        # Record the match
                        self.reconstruction_report['matched_rules'].append({
                            'type': rule_type,
                            'original': orig_rule,
                            'reconstructed': recon_rule
                        })
                        
                        matched_original.add(i)
                        self.reconstruction_report['rule_types'][rule_type]['matched'] += 1
                        matched = True
                        break
                
                if not matched:
                    # This is a new rule not in the original
                    self.reconstruction_report['new_rules'].append({
                        'type': rule_type,
                        'rule': recon_rule
                    })
            
            # Find original rules that weren't matched (lost rules)
            for i, orig_rule in enumerate(self.original_bp_policy[rule_type]):
                if i not in matched_original:
                    self.reconstruction_report['lost_rules'].append({
                        'type': rule_type,
                        'rule': orig_rule
                    })
        
        # Calculate overall accuracy
        total_matched = sum(stats['matched'] for stats in self.reconstruction_report['rule_types'].values())
        total_original = self.reconstruction_report['total_rules']['original']
        
        if total_original > 0:
            self.reconstruction_report['accuracy'] = total_matched / total_original
        else:
            self.reconstruction_report['accuracy'] = 0.0
        
        return self.reconstruction_report
    
    def _rules_match(self, rule1, rule2):
        """
        Check if two rules match (ignoring UIDs).
        
        :param rule1: First rule
        :param rule2: Second rule
        :return: True if rules match, False otherwise
        """
        # Create copies without UIDs
        rule1_copy = copy.deepcopy(rule1)
        rule2_copy = copy.deepcopy(rule2)
        
        if 'uid' in rule1_copy:
            del rule1_copy['uid']
        if 'uid' in rule2_copy:
            del rule2_copy['uid']
        
        # Check if action and target match
        if rule1_copy.get('action', '') != rule2_copy.get('action', ''):
            return False
        
        # For targets, we need to handle fragment-specific targets
        target1 = rule1_copy.get('target', '')
        target2 = rule2_copy.get('target', '')
        
        if not self._targets_match(target1, target2):
            return False
        
        # Check constraints
        constraints1 = rule1_copy.get('constraint', [])
        constraints2 = rule2_copy.get('constraint', [])
        
        if not self._constraints_match(constraints1, constraints2):
            return False
        
        # If we get here, the rules match
        return True
    
    def _targets_match(self, target1, target2):
        """
        Check if two targets match, considering fragment-specific targets.
        
        :param target1: First target
        :param target2: Second target
        :return: True if targets match, False otherwise
        """
        # If exactly the same, they match
        if target1 == target2:
            return True
        
        # Check if one is a fragment-specific target
        import re
        fragment_match1 = re.search(r'fragment_(\d+)', target1)
        fragment_match2 = re.search(r'fragment_(\d+)', target2)
        
        # If both are fragment-specific or neither is, they don't match
        if bool(fragment_match1) == bool(fragment_match2):
            return False
        
        # Extract the activity name from the non-fragment target
        activity_match = re.search(r'asset:([^/]+)', target1 if not fragment_match1 else target2)
        if not activity_match:
            return False
        
        activity_name = activity_match.group(1)
        
        # Extract the fragment ID from the fragment-specific target
        fragment_id = int(fragment_match1.group(1) if fragment_match1 else fragment_match2.group(1))
        
        # Check if the activity is in the fragment
        if self.fragments and fragment_id < len(self.fragments):
            fragment = self.fragments[fragment_id]
            return activity_name in fragment.get('activities', [])
        
        return False
    
    def _constraints_match(self, constraints1, constraints2):
        """
        Check if two sets of constraints match.
        
        :param constraints1: First set of constraints
        :param constraints2: Second set of constraints
        :return: True if constraints match, False otherwise
        """
        # If both empty, they match
        if not constraints1 and not constraints2:
            return True
        
        # If only one is empty, they don't match
        if bool(constraints1) != bool(constraints2):
            return False
        
        # If different lengths, they don't match
        if len(constraints1) != len(constraints2):
            return False
        
        # Create copies for sorting
        sorted_constraints1 = sorted(constraints1, key=lambda c: (c.get('leftOperand', ''), c.get('operator', '')))
        sorted_constraints2 = sorted(constraints2, key=lambda c: (c.get('leftOperand', ''), c.get('operator', '')))
        
        # Compare each constraint
        for c1, c2 in zip(sorted_constraints1, sorted_constraints2):
            if c1.get('leftOperand', '') != c2.get('leftOperand', ''):
                return False
            if c1.get('operator', '') != c2.get('operator', ''):
                return False
            
            # For rightOperand, we need to handle fragment-specific values
            right1 = c1.get('rightOperand', '')
            right2 = c2.get('rightOperand', '')
            
            # Check for fragment-specific values
            import re
            fragment_match1 = re.search(r'fragment_(\d+)', str(right1))
            fragment_match2 = re.search(r'fragment_(\d+)', str(right2))
            
            # If both are fragment-specific or neither is, they should match exactly
            if bool(fragment_match1) == bool(fragment_match2):
                if right1 != right2:
                    return False
            # Otherwise, they're referring to different levels and don't match
            else:
                return False
        
        # If we get here, all constraints match
        return True
    
    def save_reconstruction(self, output_dir):
        """
        Save the reconstructed policy and evaluation report to JSON files.
        
        :param output_dir: Directory to save the files
        :return: dict with paths to saved files
        """
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save reconstructed policy
        if self.reconstructed_policy:
            policy_file = os.path.join(output_dir, "reconstructed_policy.json")
            with open(policy_file, 'w') as f:
                json.dump(self.reconstructed_policy, f, indent=2)
            saved_files['policy'] = policy_file
        
        # Save evaluation report
        if self.reconstruction_report:
            report_file = os.path.join(output_dir, "reconstruction_report.json")
            with open(report_file, 'w') as f:
                json.dump(self.reconstruction_report, f, indent=2)
            saved_files['report'] = report_file
        
        return saved_files
    
    def get_reconstruction_metrics(self):
        """
        Get metrics for the policy reconstruction.
        
        :return: dict with reconstruction metrics
        """
        # Ensure we have an evaluation report
        if not self.reconstruction_report:
            self.evaluate_reconstruction()
        
        metrics = {
            'total_original_rules': self.reconstruction_report['total_rules']['original'],
            'total_reconstructed_rules': self.reconstruction_report['total_rules']['reconstructed'],
            'matched_rules': sum(stats['matched'] for stats in self.reconstruction_report['rule_types'].values()),
            'lost_rules': len(self.reconstruction_report['lost_rules']),
            'new_rules': len(self.reconstruction_report['new_rules']),
            'accuracy': self.reconstruction_report['accuracy'],
            'rule_types': {
                'permission': {
                    'original': self.reconstruction_report['rule_types']['permission']['original'],
                    'reconstructed': self.reconstruction_report['rule_types']['permission']['reconstructed'],
                    'matched': self.reconstruction_report['rule_types']['permission']['matched']
                },
                'prohibition': {
                    'original': self.reconstruction_report['rule_types']['prohibition']['original'],
                    'reconstructed': self.reconstruction_report['rule_types']['prohibition']['reconstructed'],
                    'matched': self.reconstruction_report['rule_types']['prohibition']['matched']
                },
                'obligation': {
                    'original': self.reconstruction_report['rule_types']['obligation']['original'],
                    'reconstructed': self.reconstruction_report['rule_types']['obligation']['reconstructed'],
                    'matched': self.reconstruction_report['rule_types']['obligation']['matched']
                }
            }
        }
        
        return metrics
