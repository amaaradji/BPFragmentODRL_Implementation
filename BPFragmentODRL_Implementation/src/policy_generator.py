"""
policy_generator.py

Generates ODRL-based policies for each activity (fragment policy)
and for each dependency (flow) in the BPMN model.
"""

class PolicyGenerator:
    """
    PolicyGenerator creates:
      - Activity-level policies (FPa) from a global or default BP policy
      - Dependency-level policies (FPd) for sequence flows, message flows, etc.
    """

    def __init__(self, bp_model, bp_policy=None):
        """
        :param bp_model: dict representing the BPMN model (activities, flows, etc.)
        :param bp_policy: optional dict representing an original ODRL policy that
                          might contain rules for each activity (target = activity).
        """
        self.bp_model = bp_model
        # This might contain top-level 'permission'/'prohibition'/'obligation' referencing activities
        self.bp_policy = bp_policy

    def generate_policies(self):
        """
        Main entry point to generate:
          - activity_policies: a dict { activity_name: <ODRL policy dict> }
          - dependency_policies: a dict { 'A->B': [<ODRL policy dict>, ...], ... }

        :return: (activity_policies, dependency_policies)
        """
        activity_policies = {}
        dependency_policies = {}

        # 1. Activity-level policies (look up the activity in bp_policy if provided)
        for activity in self.bp_model['activities']:
            a_name = activity['name']
            # Attempt to match an ODRL rule in bp_policy whose "target" = "http://example.com/asset/<a_name>"
            # This is just a demonstration; adapt how you store/find them.
            act_policy_dict = self._extract_activity_policy(a_name)
            if act_policy_dict:
                activity_policies[a_name] = act_policy_dict
            else:
                # If no policy found in global bp_policy, create an empty 'shell' or skip
                activity_policies[a_name] = {
                    "@context": "http://www.w3.org/ns/odrl.jsonld",
                    "uid": f"http://example.com/policy:{a_name}",
                    "@type": "Set"
                }

        # 2. Dependency-level policies (for flows like sequence, message, etc.)
        for flow in self.bp_model['flows']:
            from_act = flow['from']
            to_act = flow['to']
            flow_type = flow.get('type', 'sequence')

            # We'll store each flow's policies in a list
            key = f"{from_act}->{to_act}"
            if key not in dependency_policies:
                dependency_policies[key] = []

            # Create sequence flow policy if type=sequence
            if flow_type == 'sequence':
                seq_pol = self._create_sequence_flow_policy(from_act, to_act)
                if seq_pol:
                    dependency_policies[key].append(seq_pol)

            # Create message flow policy if type=message
            if flow_type == 'message':
                msg_pol = self._create_message_flow_policy(from_act, to_act)
                if msg_pol:
                    dependency_policies[key].append(msg_pol)

            # ... similarly for conditional flows or other types if needed

        return activity_policies, dependency_policies

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------

    def _extract_activity_policy(self, activity_name):
        """
        Finds all relevant rules in bp_policy that target 'activity_name'.
        For demonstration, we only match if the rule's target ends with <activity_name>.
        Then we build an ODRL policy with those rules.
        """
        if not self.bp_policy:
            return None

        # A minimal ODRL skeleton:
        policy_skel = {
            "@context": "http://www.w3.org/ns/odrl.jsonld",
            "uid": f"http://example.com/policy:{activity_name}",
            "@type": self.bp_policy.get("@type", "Agreement"),
        }

        # We will look for "permission", "prohibition", "obligation" in the bp_policy
        # that mention the activity by name in "target".
        for rule_type in ["permission", "prohibition", "obligation"]:
            if rule_type in self.bp_policy:
                for rule in self.bp_policy[rule_type]:
                    if "target" in rule and activity_name in rule["target"]:
                        # Add this rule to policy_skel
                        policy_skel.setdefault(rule_type, []).append(rule)

        # If we found any relevant rules, return them; else return None
        # Actually we can return the skeleton even if empty
        if any(rt in policy_skel for rt in ["permission","prohibition","obligation"]):
            return policy_skel
        else:
            return None

    def _create_sequence_flow_policy(self, from_act, to_act):
        """
        Creates an ODRL snippet that says:
         - If 'from_act' is done, then enable 'to_act' (or next policy).
        """
        # Minimal example:
        seq_policy = {
            "@context": "http://www.w3.org/ns/odrl.jsonld",
            "uid": f"http://example.com/policy:{from_act}_{to_act}_sequence",
            "@type": "Agreement",
            "permission": [{
                "uid": f"http://example.com/rules/RuleId_{from_act}_{to_act}_sequence",
                "target": f"http://example.com/rules/{from_act}Rule",  # from_act as a symbolic rule
                "action": "enable",
                "duty": [
                    {
                        "action": "nextpolicy",
                        "uid": f"http://example.com/policy:{to_act}"  # Points to to_act policy
                    }
                ],
                "constraint": [
                    {
                        "leftOperand": "event",
                        "operator": "gt",
                        "rightOperand": {"@id": "odrl:policyUsage"}
                    }
                ]
            }]
        }
        return seq_policy

    def _create_message_flow_policy(self, from_act, to_act):
        """
        Creates an ODRL snippet for a message flow:
         - Permission to 'transfer' a message from from_act to to_act
         - Duty to enable the to_act's rule on receipt
        """
        msg_policy = {
            "@context": "http://www.w3.org/ns/odrl.jsonld",
            "uid": f"http://example.com/policy:{from_act}_{to_act}_message",
            "@type": "Agreement",
            "permission": [{
                "uid": f"http://example.com/rules/RuleId_{from_act}_{to_act}_message",
                "target": f"http://example.com/messages/{from_act}_to_{to_act}_message",
                "assignee": f"http://example.com/rules/{from_act}Rule",
                "action": [
                    {
                        "rdf:value": {"@id": "odrl:transfer"},
                        "refinement": [
                            {
                                "leftOperand": "recipient",
                                "operator": "eq",
                                "rightOperand": f"http://example.com/rules/{to_act}Rule"
                            }
                        ]
                    }
                ],
                "duty": [
                    {
                        "action": "enable",
                        "target": f"http://example.com/rules/{to_act}Rule"
                    }
                ],
                "constraint": [
                    {
                        "leftOperand": "event",
                        "operator": "gt",
                        "rightOperand": {"@id": "odrl:policyUsage"}
                    }
                ]
            }]
        }
        return msg_policy
