"""
enhanced_policy_generator.py

Provides an EnhancedPolicyGenerator class to generate ODRL-based policies for BPMN fragments.
This extends the original policy_generator.py with additional functionality.
"""

import json
import logging
import random
import string
import uuid
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedPolicyGenerator:
    """
    EnhancedPolicyGenerator creates:
    - Fragment Activity Policies (FPa): Policies for activities within fragments
    - Fragment Dependency Policies (FPd): Policies for dependencies between fragments
    
    It supports:
    - Policy generation from existing BP-level policies
    - Synthetic policy generation based on activity names and fragment structure
    - Multiple policy types (permissions, prohibitions, obligations)
    - Constraint generation
    - Policy templates
    
    Typical usage:
        generator = EnhancedPolicyGenerator(bp_model, fragments, fragment_dependencies)
        activity_policies, dependency_policies = generator.generate_policies()
    """
    
    def __init__(self, bp_model, fragments, fragment_dependencies, bp_policy=None):
        """
        Initialize the policy generator.
        
        :param bp_model: dict representing the BPMN model
        :param fragments: list of fragment dicts from the fragmenter
        :param fragment_dependencies: list of dependency dicts from the fragmenter
        :param bp_policy: optional dict representing an original ODRL policy
        """
        self.bp_model = bp_model
        self.fragments = fragments
        self.fragment_dependencies = fragment_dependencies
        self.bp_policy = bp_policy
        
        # Create lookup dictionaries for faster access
        self.activity_map = {act['name']: act for act in bp_model.get('activities', [])}
        self.gateway_map = {gw['name']: gw for gw in bp_model.get('gateways', [])}
        
        # Initialize policy containers
        self.fragment_activity_policies = {}  # FPa
        self.fragment_dependency_policies = {}  # FPd
        
        # Load policy templates
        self.policy_templates = self._load_policy_templates()
    
    def _load_policy_templates(self):
        """
        Load or define policy templates for synthetic policy generation.
        
        :return: dict of policy templates
        """
        # Define basic templates for different activity types
        templates = {
            'default': {
                'permission': [
                    {
                        'action': 'execute',
                        'constraint_templates': [
                            {'leftOperand': 'dateTime', 'operator': 'gt', 'rightOperand': 'CURRENT_DATE'},
                            {'leftOperand': 'dateTime', 'operator': 'lt', 'rightOperand': 'FUTURE_DATE'}
                        ]
                    },
                    {
                        'action': 'read',
                        'constraint_templates': [
                            {'leftOperand': 'role', 'operator': 'eq', 'rightOperand': 'ROLE'}
                        ]
                    }
                ],
                'prohibition': [
                    {
                        'action': 'modify',
                        'constraint_templates': [
                            {'leftOperand': 'role', 'operator': 'neq', 'rightOperand': 'ADMIN_ROLE'}
                        ]
                    }
                ],
                'obligation': [
                    {
                        'action': 'log',
                        'constraint_templates': [
                            {'leftOperand': 'event', 'operator': 'eq', 'rightOperand': 'EXECUTION_COMPLETE'}
                        ]
                    }
                ]
            },
            'approval': {
                'permission': [
                    {
                        'action': 'approve',
                        'constraint_templates': [
                            {'leftOperand': 'role', 'operator': 'eq', 'rightOperand': 'MANAGER_ROLE'}
                        ]
                    }
                ],
                'prohibition': [
                    {
                        'action': 'approve',
                        'constraint_templates': [
                            {'leftOperand': 'role', 'operator': 'eq', 'rightOperand': 'REQUESTER_ROLE'}
                        ]
                    }
                ],
                'obligation': [
                    {
                        'action': 'notify',
                        'constraint_templates': [
                            {'leftOperand': 'event', 'operator': 'eq', 'rightOperand': 'APPROVAL_COMPLETE'}
                        ]
                    }
                ]
            },
            'payment': {
                'permission': [
                    {
                        'action': 'pay',
                        'constraint_templates': [
                            {'leftOperand': 'amount', 'operator': 'lteq', 'rightOperand': 'PAYMENT_LIMIT'}
                        ]
                    }
                ],
                'prohibition': [
                    {
                        'action': 'pay',
                        'constraint_templates': [
                            {'leftOperand': 'amount', 'operator': 'gt', 'rightOperand': 'PAYMENT_LIMIT'}
                        ]
                    }
                ],
                'obligation': [
                    {
                        'action': 'record',
                        'constraint_templates': [
                            {'leftOperand': 'event', 'operator': 'eq', 'rightOperand': 'PAYMENT_COMPLETE'}
                        ]
                    }
                ]
            },
            'verification': {
                'permission': [
                    {
                        'action': 'verify',
                        'constraint_templates': [
                            {'leftOperand': 'role', 'operator': 'eq', 'rightOperand': 'AUDITOR_ROLE'}
                        ]
                    }
                ],
                'prohibition': [
                    {
                        'action': 'modify',
                        'constraint_templates': [
                            {'leftOperand': 'status', 'operator': 'eq', 'rightOperand': 'VERIFIED'}
                        ]
                    }
                ],
                'obligation': [
                    {
                        'action': 'timestamp',
                        'constraint_templates': [
                            {'leftOperand': 'event', 'operator': 'eq', 'rightOperand': 'VERIFICATION_COMPLETE'}
                        ]
                    }
                ]
            }
        }
        
        # Add templates for dependency types
        templates['sequence'] = {
            'permission': [
                {
                    'action': 'enable',
                    'constraint_templates': [
                        {'leftOperand': 'event', 'operator': 'eq', 'rightOperand': 'PREDECESSOR_COMPLETE'}
                    ]
                }
            ],
            'prohibition': [
                {
                    'action': 'execute',
                    'constraint_templates': [
                        {'leftOperand': 'predecessor_status', 'operator': 'neq', 'rightOperand': 'COMPLETE'}
                    ]
                }
            ],
            'obligation': [
                {
                    'action': 'notify',
                    'constraint_templates': [
                        {'leftOperand': 'event', 'operator': 'eq', 'rightOperand': 'PREDECESSOR_COMPLETE'}
                    ]
                }
            ]
        }
        
        templates['xor_split'] = {
            'permission': [
                {
                    'action': 'select',
                    'constraint_templates': [
                        {'leftOperand': 'condition', 'operator': 'eq', 'rightOperand': 'CONDITION_MET'}
                    ]
                }
            ],
            'prohibition': [
                {
                    'action': 'select_multiple',
                    'constraint_templates': []
                }
            ],
            'obligation': [
                {
                    'action': 'evaluate',
                    'constraint_templates': [
                        {'leftOperand': 'event', 'operator': 'eq', 'rightOperand': 'GATEWAY_REACHED'}
                    ]
                }
            ]
        }
        
        templates['and_split'] = {
            'permission': [
                {
                    'action': 'execute_parallel',
                    'constraint_templates': []
                }
            ],
            'prohibition': [
                {
                    'action': 'skip',
                    'constraint_templates': []
                }
            ],
            'obligation': [
                {
                    'action': 'start_all',
                    'constraint_templates': [
                        {'leftOperand': 'event', 'operator': 'eq', 'rightOperand': 'GATEWAY_REACHED'}
                    ]
                }
            ]
        }
        
        return templates
    
    def generate_policies(self, use_templates=True, policy_density=0.7):
        """
        Generate fragment activity policies (FPa) and fragment dependency policies (FPd).
        
        :param use_templates: Whether to use templates for synthetic policy generation
        :param policy_density: Probability (0-1) of generating a policy for each activity/dependency
        :return: (fragment_activity_policies, fragment_dependency_policies)
        """
        # Generate fragment activity policies (FPa)
        self._generate_fragment_activity_policies(use_templates, policy_density)
        
        # Generate fragment dependency policies (FPd)
        self._generate_fragment_dependency_policies(use_templates, policy_density)
        
        return self.fragment_activity_policies, self.fragment_dependency_policies
    
    def _generate_fragment_activity_policies(self, use_templates, policy_density):
        """
        Generate policies for activities within fragments (FPa).
        
        :param use_templates: Whether to use templates for synthetic policy generation
        :param policy_density: Probability (0-1) of generating a policy for each activity
        """
        # Process each fragment
        for fragment in self.fragments:
            fragment_id = fragment['id']
            self.fragment_activity_policies[fragment_id] = {}
            
            # Process each activity in the fragment
            for activity_name in fragment['activities']:
                # Decide whether to generate a policy for this activity
                if random.random() > policy_density:
                    continue
                
                # Try to extract from BP-level policy if available
                activity_policy = self._extract_activity_policy(activity_name)
                
                # If no policy found and templates are enabled, generate synthetic policy
                if not activity_policy and use_templates:
                    activity_policy = self._generate_synthetic_activity_policy(activity_name)
                
                # Store the policy if we have one
                if activity_policy:
                    self.fragment_activity_policies[fragment_id][activity_name] = activity_policy
    
    def _generate_fragment_dependency_policies(self, use_templates, policy_density):
        """
        Generate policies for dependencies between fragments (FPd).
        
        :param use_templates: Whether to use templates for synthetic policy generation
        :param policy_density: Probability (0-1) of generating a policy for each dependency
        """
        # Process each fragment dependency
        for dependency in self.fragment_dependencies:
            # Decide whether to generate a policy for this dependency
            if random.random() > policy_density:
                continue
            
            from_fragment = dependency['from_fragment']
            to_fragment = dependency['to_fragment']
            dependency_type = dependency.get('type', 'sequence')
            gateway = dependency.get('gateway', None)
            
            # Generate a key for this dependency
            dependency_key = f"{from_fragment}->{to_fragment}"
            
            # Initialize if not exists
            if dependency_key not in self.fragment_dependency_policies:
                self.fragment_dependency_policies[dependency_key] = []
            
            # Generate policy based on dependency type
            if dependency_type == 'sequence':
                policy = self._create_sequence_dependency_policy(from_fragment, to_fragment)
            elif dependency_type == 'xor_split':
                policy = self._create_xor_dependency_policy(from_fragment, to_fragment, gateway)
            elif dependency_type == 'and_split':
                policy = self._create_and_dependency_policy(from_fragment, to_fragment, gateway)
            else:
                # Default to sequence for unknown types
                policy = self._create_sequence_dependency_policy(from_fragment, to_fragment)
            
            # Add the policy to the list
            if policy:
                self.fragment_dependency_policies[dependency_key].append(policy)
    
    def _extract_activity_policy(self, activity_name):
        """
        Extract policy for an activity from the BP-level policy if available.
        
        :param activity_name: Name of the activity
        :return: ODRL policy dict or None if not found
        """
        if not self.bp_policy:
            return None
        
        # A minimal ODRL skeleton
        policy_skel = {
            "@context": "http://www.w3.org/ns/odrl.jsonld",
            "uid": f"http://example.com/policy:{activity_name}",
            "@type": self.bp_policy.get("@type", "Set"),
        }
        
        # Look for rules in the BP-level policy that target this activity
        for rule_type in ["permission", "prohibition", "obligation"]:
            if rule_type in self.bp_policy:
                for rule in self.bp_policy[rule_type]:
                    if "target" in rule and activity_name in rule["target"]:
                        # Add this rule to policy_skel
                        policy_skel.setdefault(rule_type, []).append(rule)
        
        # Return the policy if we found any relevant rules
        if any(rt in policy_skel for rt in ["permission", "prohibition", "obligation"]):
            return policy_skel
        else:
            return None
    
    def _generate_synthetic_activity_policy(self, activity_name):
        """
        Generate a synthetic policy for an activity based on its name and templates.
        
        :param activity_name: Name of the activity
        :return: ODRL policy dict
        """
        # Determine activity type from name
        activity_type = 'default'
        name_lower = activity_name.lower()
        
        if any(term in name_lower for term in ['approve', 'review', 'confirm']):
            activity_type = 'approval'
        elif any(term in name_lower for term in ['pay', 'payment', 'invoice']):
            activity_type = 'payment'
        elif any(term in name_lower for term in ['verify', 'check', 'validate']):
            activity_type = 'verification'
        
        # Get the template for this activity type
        template = self.policy_templates.get(activity_type, self.policy_templates['default'])
        
        # Create policy skeleton
        policy = {
            "@context": "http://www.w3.org/ns/odrl.jsonld",
            "uid": f"http://example.com/policy:{activity_name}_{self._generate_id()}",
            "@type": "Set"
        }
        
        # Add rules based on templates
        for rule_type in ['permission', 'prohibition', 'obligation']:
            if rule_type in template:
                policy[rule_type] = []
                
                # Randomly select 1-2 rules from the template
                num_rules = random.randint(1, min(2, len(template[rule_type])))
                selected_rules = random.sample(template[rule_type], num_rules)
                
                for rule_template in selected_rules:
                    rule = {
                        "uid": f"http://example.com/rule:{activity_name}_{rule_type}_{self._generate_id()}",
                        "target": f"http://example.com/asset:{activity_name}",
                        "action": rule_template['action']
                    }
                    
                    # Add constraints if any
                    if 'constraint_templates' in rule_template and rule_template['constraint_templates']:
                        # Randomly select 0-2 constraints
                        num_constraints = random.randint(0, min(2, len(rule_template['constraint_templates'])))
                        if num_constraints > 0:
                            selected_constraints = random.sample(rule_template['constraint_templates'], num_constraints)
                            rule['constraint'] = []
                            
                            for constraint_template in selected_constraints:
                                constraint = constraint_template.copy()
                                
                                # Replace placeholders with realistic values
                                if constraint['rightOperand'] == 'CURRENT_DATE':
                                    constraint['rightOperand'] = datetime.now().strftime('%Y-%m-%d')
                                elif constraint['rightOperand'] == 'FUTURE_DATE':
                                    future_date = datetime.now() + timedelta(days=random.randint(30, 365))
                                    constraint['rightOperand'] = future_date.strftime('%Y-%m-%d')
                                elif constraint['rightOperand'] == 'ROLE':
                                    constraint['rightOperand'] = random.choice(['user', 'manager', 'admin'])
                                elif constraint['rightOperand'] == 'ADMIN_ROLE':
                                    constraint['rightOperand'] = 'admin'
                                elif constraint['rightOperand'] == 'MANAGER_ROLE':
                                    constraint['rightOperand'] = 'manager'
                                elif constraint['rightOperand'] == 'REQUESTER_ROLE':
                                    constraint['rightOperand'] = 'requester'
                                elif constraint['rightOperand'] == 'AUDITOR_ROLE':
                                    constraint['rightOperand'] = 'auditor'
                                elif constraint['rightOperand'] == 'PAYMENT_LIMIT':
                                    constraint['rightOperand'] = str(random.randint(1000, 10000))
                                elif constraint['rightOperand'] == 'EXECUTION_COMPLETE':
                                    constraint['rightOperand'] = 'execution_complete'
                                elif constraint['rightOperand'] == 'APPROVAL_COMPLETE':
                                    constraint['rightOperand'] = 'approval_complete'
                                elif constraint['rightOperand'] == 'PAYMENT_COMPLETE':
                                    constraint['rightOperand'] = 'payment_complete'
                                elif constraint['rightOperand'] == 'VERIFICATION_COMPLETE':
                                    constraint['rightOperand'] = 'verification_complete'
                                elif constraint['rightOperand'] == 'PREDECESSOR_COMPLETE':
                                    constraint['rightOperand'] = 'predecessor_complete'
                                elif constraint['rightOperand'] == 'COMPLETE':
                                    constraint['rightOperand'] = 'complete'
                                elif constraint['rightOperand'] == 'CONDITION_MET':
                                    constraint['rightOperand'] = 'condition_met'
                                elif constraint['rightOperand'] == 'GATEWAY_REACHED':
                                    constraint['rightOperand'] = 'gateway_reached'
                                elif constraint['rightOperand'] == 'VERIFIED':
                                    constraint['rightOperand'] = 'verified'
                                
                                rule['constraint'].append(constraint)
                    
                    policy[rule_type].append(rule)
        
        return policy
    
    def _create_sequence_dependency_policy(self, from_fragment, to_fragment):
        """
        Create a policy for a sequence dependency between fragments.
        
        :param from_fragment: ID of the source fragment
        :param to_fragment: ID of the target fragment
        :return: ODRL policy dict
        """
        policy_id = self._generate_id()
        
        # Create a policy that enables the target fragment when the source fragment completes
        policy = {
            "@context": "http://www.w3.org/ns/odrl.jsonld",
            "uid": f"http://example.com/policy:seq_{from_fragment}_{to_fragment}_{policy_id}",
            "@type": "Agreement",
            "permission": [{
                "uid": f"http://example.com/rule:seq_{from_fragment}_{to_fragment}_{policy_id}",
                "target": f"http://example.com/asset:fragment_{to_fragment}",
                "action": "enable",
                "constraint": [{
                    "leftOperand": "event",
                    "operator": "eq",
                    "rightOperand": f"fragment_{from_fragment}_complete"
                }]
            }],
            "prohibition": [{
                "uid": f"http://example.com/rule:seq_prohibit_{from_fragment}_{to_fragment}_{policy_id}",
                "target": f"http://example.com/asset:fragment_{to_fragment}",
                "action": "execute",
                "constraint": [{
                    "leftOperand": "fragment_status",
                    "operator": "neq",
                    "rightOperand": f"fragment_{from_fragment}_complete"
                }]
            }]
        }
        
        # Randomly add an obligation (50% chance)
        if random.random() > 0.5:
            policy["obligation"] = [{
                "uid": f"http://example.com/rule:seq_obligation_{from_fragment}_{to_fragment}_{policy_id}",
                "target": f"http://example.com/asset:fragment_{from_fragment}",
                "action": "notify",
                "constraint": [{
                    "leftOperand": "event",
                    "operator": "eq",
                    "rightOperand": "completion"
                }]
            }]
        
        return policy
    
    def _create_xor_dependency_policy(self, from_fragment, to_fragment, gateway):
        """
        Create a policy for an XOR gateway dependency between fragments.
        
        :param from_fragment: ID of the source fragment
        :param to_fragment: ID of the target fragment
        :param gateway: Name of the gateway
        :return: ODRL policy dict
        """
        policy_id = self._generate_id()
        gateway_str = gateway if gateway else f"xor_gateway_{from_fragment}_{to_fragment}"
        
        # Create a policy that allows selecting this path based on a condition
        policy = {
            "@context": "http://www.w3.org/ns/odrl.jsonld",
            "uid": f"http://example.com/policy:xor_{from_fragment}_{to_fragment}_{policy_id}",
            "@type": "Agreement",
            "permission": [{
                "uid": f"http://example.com/rule:xor_{from_fragment}_{to_fragment}_{policy_id}",
                "target": f"http://example.com/asset:fragment_{to_fragment}",
                "action": "select",
                "constraint": [{
                    "leftOperand": "condition",
                    "operator": "eq",
                    "rightOperand": f"condition_{gateway_str}_met"
                }]
            }],
            "prohibition": [{
                "uid": f"http://example.com/rule:xor_prohibit_{from_fragment}_{to_fragment}_{policy_id}",
                "target": f"http://example.com/asset:fragment_{to_fragment}",
                "action": "select",
                "constraint": [{
                    "leftOperand": "condition",
                    "operator": "neq",
                    "rightOperand": f"condition_{gateway_str}_met"
                }]
            }]
        }
        
        # Add an obligation to evaluate the condition
        policy["obligation"] = [{
            "uid": f"http://example.com/rule:xor_obligation_{from_fragment}_{to_fragment}_{policy_id}",
            "target": f"http://example.com/asset:{gateway_str}",
            "action": "evaluate",
            "constraint": [{
                "leftOperand": "event",
                "operator": "eq",
                "rightOperand": "gateway_reached"
            }]
        }]
        
        return policy
    
    def _create_and_dependency_policy(self, from_fragment, to_fragment, gateway):
        """
        Create a policy for an AND gateway dependency between fragments.
        
        :param from_fragment: ID of the source fragment
        :param to_fragment: ID of the target fragment
        :param gateway: Name of the gateway
        :return: ODRL policy dict
        """
        policy_id = self._generate_id()
        gateway_str = gateway if gateway else f"and_gateway_{from_fragment}_{to_fragment}"
        
        # Create a policy that requires executing all parallel paths
        policy = {
            "@context": "http://www.w3.org/ns/odrl.jsonld",
            "uid": f"http://example.com/policy:and_{from_fragment}_{to_fragment}_{policy_id}",
            "@type": "Agreement",
            "permission": [{
                "uid": f"http://example.com/rule:and_{from_fragment}_{to_fragment}_{policy_id}",
                "target": f"http://example.com/asset:fragment_{to_fragment}",
                "action": "execute_parallel"
            }],
            "prohibition": [{
                "uid": f"http://example.com/rule:and_prohibit_{from_fragment}_{to_fragment}_{policy_id}",
                "target": f"http://example.com/asset:fragment_{to_fragment}",
                "action": "skip"
            }]
        }
        
        # Add an obligation to start all parallel paths
        policy["obligation"] = [{
            "uid": f"http://example.com/rule:and_obligation_{from_fragment}_{to_fragment}_{policy_id}",
            "target": f"http://example.com/asset:{gateway_str}",
            "action": "start_all",
            "constraint": [{
                "leftOperand": "event",
                "operator": "eq",
                "rightOperand": "gateway_reached"
            }]
        }]
        
        return policy
    
    def _generate_id(self):
        """Generate a short unique ID for policy/rule identifiers."""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    
    def save_policies(self, output_dir):
        """
        Save generated policies to JSON files.
        
        :param output_dir: Directory to save the policy files
        :return: dict with paths to saved files
        """
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {
            'fragment_activity_policies': {},
            'fragment_dependency_policies': {}
        }
        
        # Save fragment activity policies (FPa)
        for fragment_id, policies in self.fragment_activity_policies.items():
            if policies:
                filename = os.path.join(output_dir, f"fragment_{fragment_id}_activity_policies.json")
                with open(filename, 'w') as f:
                    json.dump(policies, f, indent=2)
                saved_files['fragment_activity_policies'][fragment_id] = filename
        
        # Save fragment dependency policies (FPd)
        for dependency_key, policies in self.fragment_dependency_policies.items():
            if policies:
                from_fragment, to_fragment = dependency_key.split('->')
                filename = os.path.join(output_dir, f"dependency_{from_fragment}_{to_fragment}_policies.json")
                with open(filename, 'w') as f:
                    json.dump(policies, f, indent=2)
                saved_files['fragment_dependency_policies'][dependency_key] = filename
        
        # Save a summary file
        summary = {
            'total_fragment_activity_policies': sum(len(policies) for policies in self.fragment_activity_policies.values()),
            'total_fragment_dependency_policies': sum(len(policies) for policies in self.fragment_dependency_policies.values()),
            'fragments_with_policies': len(self.fragment_activity_policies),
            'dependencies_with_policies': len(self.fragment_dependency_policies)
        }
        
        summary_file = os.path.join(output_dir, "policy_generation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        saved_files['summary'] = summary_file
        
        return saved_files
    
    def get_policy_metrics(self):
        """
        Calculate metrics for the generated policies.
        
        :return: dict with policy metrics
        """
        metrics = {
            'fragment_activity_policies': {
                'total_fragments': len(self.fragment_activity_policies),
                'total_policies': sum(len(policies) for policies in self.fragment_activity_policies.values()),
                'permissions': 0,
                'prohibitions': 0,
                'obligations': 0,
                'constraints': 0
            },
            'fragment_dependency_policies': {
                'total_dependencies': len(self.fragment_dependency_policies),
                'total_policies': sum(len(policies) for policies in self.fragment_dependency_policies.values()),
                'permissions': 0,
                'prohibitions': 0,
                'obligations': 0,
                'constraints': 0
            }
        }
        
        # Count rule types in fragment activity policies
        for fragment_id, policies in self.fragment_activity_policies.items():
            for activity, policy in policies.items():
                metrics['fragment_activity_policies']['permissions'] += len(policy.get('permission', []))
                metrics['fragment_activity_policies']['prohibitions'] += len(policy.get('prohibition', []))
                metrics['fragment_activity_policies']['obligations'] += len(policy.get('obligation', []))
                
                # Count constraints
                for rule_type in ['permission', 'prohibition', 'obligation']:
                    for rule in policy.get(rule_type, []):
                        metrics['fragment_activity_policies']['constraints'] += len(rule.get('constraint', []))
        
        # Count rule types in fragment dependency policies
        for dependency_key, policies in self.fragment_dependency_policies.items():
            for policy in policies:
                metrics['fragment_dependency_policies']['permissions'] += len(policy.get('permission', []))
                metrics['fragment_dependency_policies']['prohibitions'] += len(policy.get('prohibition', []))
                metrics['fragment_dependency_policies']['obligations'] += len(policy.get('obligation', []))
                
                # Count constraints
                for rule_type in ['permission', 'prohibition', 'obligation']:
                    for rule in policy.get(rule_type, []):
                        metrics['fragment_dependency_policies']['constraints'] += len(rule.get('constraint', []))
        
        # Calculate totals
        metrics['total'] = {
            'fragments_and_dependencies': metrics['fragment_activity_policies']['total_fragments'] + 
                                         metrics['fragment_dependency_policies']['total_dependencies'],
            'policies': metrics['fragment_activity_policies']['total_policies'] + 
                       metrics['fragment_dependency_policies']['total_policies'],
            'permissions': metrics['fragment_activity_policies']['permissions'] + 
                          metrics['fragment_dependency_policies']['permissions'],
            'prohibitions': metrics['fragment_activity_policies']['prohibitions'] + 
                           metrics['fragment_dependency_policies']['prohibitions'],
            'obligations': metrics['fragment_activity_policies']['obligations'] + 
                          metrics['fragment_dependency_policies']['obligations'],
            'constraints': metrics['fragment_activity_policies']['constraints'] + 
                          metrics['fragment_dependency_policies']['constraints']
        }
        
        return metrics
