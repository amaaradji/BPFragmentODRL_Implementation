"""
enhanced_policy_generator_llm.py

This module provides the EnhancedPolicyGenerator class, which is responsible for
generating ODRL-based policies for business process fragments. This version
integrates with Azure OpenAI (GPT-4o) to generate policies using an LLM.
"""

import os
import json
import logging
from openai import AzureOpenAI
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedPolicyGeneratorLLM:
    """
    Generates ODRL-based policies for BPMN fragments, with an option to use an LLM.
    """
    def __init__(self, model_data=None, fragmentation_strategy=None):
        """
        Initializes the EnhancedPolicyGenerator.
        Args:
            model_data (dict, optional): Parsed BPMN model data. Defaults to None.
            fragmentation_strategy (str, optional): The fragmentation strategy used. Defaults to None.
        """
        self.model_data = model_data
        self.fragmentation_strategy = fragmentation_strategy
        self.llm_client = self._initialize_llm_client()
        self.policy_uid_counter = 0

    def _initialize_llm_client(self):
        """
        Initializes and returns the AzureOpenAI client.
        Reads the API key from the AZURE_OPENAI_KEY environment variable.
        """
        try:
            subscription_key = os.getenv("AZURE_OPENAI_KEY")
            if not subscription_key:
                logger.warning("AZURE_OPENAI_KEY environment variable not set. LLM policy generation will be disabled.")
                return None

            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://o3miniapi.openai.azure.com/") 
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
            self.llm_model_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
            
            client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=subscription_key,
            )
            logger.info(f"AzureOpenAI client initialized successfully for endpoint {endpoint} and deployment {self.llm_model_deployment}.")
            return client
        except Exception as e:
            logger.error(f"Error initializing AzureOpenAI client: {e}")
            return None

    def _get_next_policy_uid(self):
        self.policy_uid_counter += 1
        return f"policy_{self.policy_uid_counter:04d}"

    def _construct_llm_prompt(self, fragment_id, fragment_data):
        """
        Constructs a detailed prompt for the LLM to generate policies for a given fragment.
        """
        activities_info = []
        for act_ref in fragment_data.get('activities', []):
            activity_name = "Unnamed Activity"
            activity_id_to_find = act_ref if isinstance(act_ref, str) else act_ref.get('id')
            
            if self.model_data and self.model_data.get('activities'):
                for model_act in self.model_data['activities']:
                    if model_act.get('id') == activity_id_to_find:
                        activity_name = model_act.get('name', activity_name)
                        break
            activities_info.append(f"- Activity: '{activity_name}' (ID: {activity_id_to_find})")
        activities_str = "\n".join(activities_info) if activities_info else "No activities in this fragment."

        gateways_info = [gw.get('id', 'N/A') if isinstance(gw, dict) else gw for gw in fragment_data.get('gateways', [])]
        gateways_str = ", ".join(gateways_info) if gateways_info else "No gateways."

        entry_points_str = ", ".join(fragment_data.get('entry_points', [])) if fragment_data.get('entry_points', []) else "None"
        exit_points_str = ", ".join(fragment_data.get('exit_points', [])) if fragment_data.get('exit_points', []) else "None"
        
        process_name = self.model_data.get("name", "this business process") if self.model_data else "this business process"

        prompt = f"""
        You are an expert in ODRL and business process management. Your task is to generate ODRL-compliant policies for a fragment of a business process named '{process_name}'.

        Fragment ID: {fragment_id}
        Entry Points: {entry_points_str}
        Exit Points: {exit_points_str}
        Activities within this fragment:
        {activities_str}
        Gateways within this fragment: {gateways_str}

        Please generate ODRL policies (Permissions, Prohibitions, Obligations) for the activities in this fragment. 
        Consider typical business roles like 'Employee', 'Manager', 'Auditor', 'SystemUser', 'Customer'.
        For each policy, specify:
        1.  `target_activity_id`: The ID of the activity the policy applies to (must be one of the activity IDs listed above).
        2.  `rule_type`: Must be one of 'permission', 'prohibition', or 'obligation'.
        3.  `action`: The action being governed (e.g., 'execute', 'view_data', 'modify_data', 'approve', 'submit_form').
        4.  `assigner`: The role or entity that assigns/issues the policy (e.g., 'SystemPolicy', 'ProcessOwner').
        5.  `assignee`: The role or entity to whom the policy applies (e.g., 'Employee', 'Manager', 'AuthenticatedUser').
        6.  `constraints`: A list of ODRL constraint objects. Each constraint should be an object with `left_operand` (e.g., 'purpose', 'event', 'time_interval', 'role_hierarchy'), `operator` (e.g., 'eq', 'lte', 'during'), and `right_operand` (the value of the constraint).
            Example constraint: {{"left_operand": "purpose", "operator": "eq", "right_operand": "perform_review"}}
            If no specific constraints, provide an empty list [].

        Output a single JSON object with a root key named "policies". The value of "policies" must be a list of policy rule objects. Each object in the list represents one rule (a permission, prohibition, or obligation).
        Example of the JSON output structure:
        {{
            "policies": [
                {{
                    "target_activity_id": "Activity_123",
                    "rule_type": "permission",
                    "action": "execute",
                    "assigner": "SystemPolicy",
                    "assignee": "Manager",
                    "constraints": [
                        {{ "left_operand": "time_interval", "operator": "during", "right_operand": "working_hours" }}
                    ]
                }},
                {{
                    "target_activity_id": "Activity_456",
                    "rule_type": "prohibition",
                    "action": "view_data",
                    "assigner": "SystemPolicy",
                    "assignee": "ExternalUser",
                    "constraints": []
                }}
            ]
        }}

        Ensure the output is a valid JSON object adhering to this structure. Generate at least one meaningful policy per activity if appropriate. If no policies are applicable for an activity, do not generate any for it.
        Focus on creating meaningful and common business policies.
        """
        return prompt

    def _parse_llm_response_to_odrl(self, llm_json_policies_list, fragment_id):
        """
        Parses the LLM's JSON list of policy objects and converts it into the system's ODRL policy format.
        Returns a dictionary of activity policies for the current fragment.
        Structure: {target_activity_id: {rule_type: [policy_rule_content_dicts]}}
        """
        fragment_activity_policies = defaultdict(lambda: {"permission": [], "prohibition": [], "obligation": []})

        if not isinstance(llm_json_policies_list, list):
            logger.error(f"LLM policy data for fragment {fragment_id} is not a list as expected: {type(llm_json_policies_list)}. Data: {llm_json_policies_list}")
            return {} # Return empty dict for this fragment's activity policies

        for pol_data in llm_json_policies_list:
            if not isinstance(pol_data, dict):
                logger.warning(f"Skipping invalid policy data item (not a dict) in fragment {fragment_id}: {pol_data}")
                continue
            
            target_activity_id = pol_data.get("target_activity_id")
            rule_type = pol_data.get("rule_type")
            action = pol_data.get("action")
            # assigner is optional in prompt, provide default if missing
            assigner = pol_data.get("assigner", "SystemPolicy") 
            assignee = pol_data.get("assignee")
            constraints_data = pol_data.get("constraints", [])

            if not all([target_activity_id, rule_type, action, assignee]):
                logger.warning(f"Skipping policy in fragment {fragment_id} due to missing core fields (target_activity_id, rule_type, action, or assignee): {pol_data}")
                continue
            
            if rule_type not in ["permission", "prohibition", "obligation"]:
                logger.warning(f"Skipping policy in fragment {fragment_id} due to invalid rule_type '{rule_type}': {pol_data}")
                continue

            odrl_constraints = []
            if isinstance(constraints_data, list):
                for const_data in constraints_data:
                    if isinstance(const_data, dict) and all(k in const_data for k in ["left_operand", "operator", "right_operand"]):
                        odrl_constraints.append({
                            "leftOperand": const_data["left_operand"],
                            "operator": const_data["operator"],
                            "rightOperand": const_data["right_operand"]
                        })
                    else:
                        logger.warning(f"Skipping invalid constraint data in fragment {fragment_id}: {const_data}")
            
            policy_rule_content = {
                "uid": self._get_next_policy_uid(), # Add UID to each rule content for uniqueness
                "action": action,
                "assigner": assigner,
                "assignee": assignee,
                "constraints": odrl_constraints
            }
            
            fragment_activity_policies[target_activity_id][rule_type].append(policy_rule_content)
        
        parsed_count = sum(len(rules) for rt in fragment_activity_policies.values() for rules in rt.values())
        logger.info(f"Successfully parsed {parsed_count} activity policies from LLM response for fragment {fragment_id}.")
        return dict(fragment_activity_policies) # Convert defaultdict to dict for this fragment

    def generate_policies_with_llm(self, fragment_id, fragment_data):
        """
        Generates policies for a single fragment using the LLM.
        Returns a dictionary of activity_policies for this fragment.
        """
        if not self.llm_client:
            logger.warning(f"LLM client not initialized. Falling back to rule-based for fragment {fragment_id}.")
            # Fallback returns two dicts, we only need activity_policies here for this fragment
            activity_pols, _ = self._generate_rule_based_policies_for_fragment(fragment_id, fragment_data)
            return activity_pols

        prompt = self._construct_llm_prompt(fragment_id, fragment_data)
        logger.info(f"Sending prompt to LLM for fragment {fragment_id}...")

        try:
            response = self.llm_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert assistant skilled in generating ODRL policies for business process fragments. Ensure your output is a valid JSON object with a root key 'policies' containing a list of policy rule objects, as per the user's instructions.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.llm_model_deployment, 
                temperature=0.3, 
                max_tokens=2048, 
                response_format={"type": "json_object"} 
            )
            llm_response_content = response.choices[0].message.content
            logger.debug(f"Received LLM response for fragment {fragment_id}: {llm_response_content}")

            llm_json_policies_list = []
            try:
                parsed_response = json.loads(llm_response_content)
                if isinstance(parsed_response, dict):
                    if "policies" in parsed_response and isinstance(parsed_response["policies"], list):
                        llm_json_policies_list = parsed_response["policies"]
                        logger.info(f"Successfully extracted 'policies' list from LLM response for fragment {fragment_id}.")
                    else:
                        logger.error(f"LLM JSON response for fragment {fragment_id} is a dict but does not contain a 'policies' list or it's not a list. Keys: {list(parsed_response.keys())}. Response: {llm_response_content}")
                        # Fallback for this fragment
                        activity_pols, _ = self._generate_rule_based_policies_for_fragment(fragment_id, fragment_data)
                        return activity_pols
                else:
                    logger.error(f"LLM JSON response for fragment {fragment_id} is not a dict as expected by response_format='json_object'. Type: {type(parsed_response)}. Response: {llm_response_content}")
                    activity_pols, _ = self._generate_rule_based_policies_for_fragment(fragment_id, fragment_data)
                    return activity_pols

            except json.JSONDecodeError as e:
                logger.error(f"Error decoding LLM JSON response for fragment {fragment_id}: {e}. Response: {llm_response_content}")
                activity_pols, _ = self._generate_rule_based_policies_for_fragment(fragment_id, fragment_data)
                return activity_pols
            
            return self._parse_llm_response_to_odrl(llm_json_policies_list, fragment_id)

        except Exception as e:
            logger.error(f"Error during LLM API call for fragment {fragment_id}: {e}", exc_info=True)
            activity_pols, _ = self._generate_rule_based_policies_for_fragment(fragment_id, fragment_data)
            return activity_pols

    def _generate_rule_based_policies_for_fragment(self, fragment_id, fragment_data):
        """
        Generates basic rule-based policies for a single fragment.
        Returns two dictionaries: activity_policies and dependency_policies (empty for this func).
        """
        logger.info(f"Using rule-based policy generation for fragment {fragment_id}.")
        activity_policies = defaultdict(lambda: {"permission": [], "prohibition": [], "obligation": []})
        
        activities_in_fragment = fragment_data.get('activities', [])
        for act_ref in activities_in_fragment:
            activity_id = act_ref if isinstance(act_ref, str) else act_ref.get('id')
            if not activity_id:
                logger.warning(f"Skipping activity with no ID in fragment {fragment_id}: {act_ref}")
                continue

            activity_policies[activity_id]["permission"].append({
                "uid": self._get_next_policy_uid(),
                "action": "execute",
                "assigner": "SystemPolicy",
                "assignee": "Employee", # Default role
                "constraints": []
            })
        return dict(activity_policies), {} 

    def _generate_rule_based_dependency_policies(self, fragment_dependencies_list):
        """
        Generates rule-based dependency policies (FPd).
        `fragment_dependencies_list` is a list of dependency dicts.
        Returns a dictionary of dependency_policies.
        """
        logger.info(f"Generating rule-based dependency policies for {len(fragment_dependencies_list)} dependencies.")
        dependency_policies = defaultdict(lambda: {"permission": [], "prohibition": [], "obligation": []})
        for dep in fragment_dependencies_list:
            dep_id = dep.get('id') 
            if not dep_id: # Generate an ID if missing, though dependencies should have IDs from fragmenter
                dep_id = f"dep_{dep.get('source', 'unknown')}_{dep.get('target', 'unknown')}_{self._get_next_policy_uid()}"
                logger.warning(f"Dependency missing ID, generated: {dep_id}. Original: {dep}")

            if dep.get('type') == 'sequenceFlow': # Example policy for sequence flows
                dependency_policies[dep_id]["permission"].append({
                    "uid": self._get_next_policy_uid(),
                    "action": "traverse", 
                    "assigner": "SystemWorkflow",
                    "assignee": "SystemUser",
                    "constraints": []
                })
        return dict(dependency_policies)

    def generate_policies(self, fragments_data_list, fragment_dependencies_list, use_llm=True):
        """
        Generates policies for all fragments, using LLM or rule-based approach.
        `fragments_data_list` is a list of fragment dictionaries.
        `fragment_dependencies_list` is a list of dependency dictionaries.
        Returns two dictionaries: one for all activity policies (FPa) and one for all dependency policies (FPd).
        Structure: {target_id: {rule_type: [policy_rule_content_dicts]}}
        """
        self.all_activity_policies = defaultdict(lambda: {"permission": [], "prohibition": [], "obligation": []})
        self.all_dependency_policies = defaultdict(lambda: {"permission": [], "prohibition": [], "obligation": []})
        self.policy_uid_counter = 0 

        if not isinstance(fragments_data_list, list):
            logger.error(f"fragments_data_list is not a list: {type(fragments_data_list)}. Cannot generate policies.")
            return {}, {}

        for fragment_info in fragments_data_list:
            if not isinstance(fragment_info, dict):
                logger.warning(f"Skipping fragment_info as it's not a dict: {fragment_info}")
                continue
            
            fragment_id = fragment_info.get('id')
            # Allow integer 0 as a valid ID, but not None or empty string
            if fragment_id is None or (isinstance(fragment_id, str) and not fragment_id):
                logger.warning(f"Skipping fragment_info due to missing or invalid 'id': {fragment_info}")
                continue
            fragment_id = str(fragment_id) # Ensure fragment_id is a string for consistency as dict key
            
            logger.info(f"Processing fragment: {fragment_id}")
            current_frag_activity_policies = {} 

            if use_llm and self.llm_client:
                logger.info(f"Attempting LLM-based policy generation for fragment {fragment_id}.")
                current_frag_activity_policies = self.generate_policies_with_llm(fragment_id, fragment_info)
            
            # If LLM generation was skipped, failed, or returned empty, fallback to rule-based for this fragment's activities
            if not current_frag_activity_policies:
                logger.info(f"LLM generation yielded no policies for fragment {fragment_id}, or was skipped. Using rule-based for this fragment's activities.")
                # _generate_rule_based_policies_for_fragment returns (activity_pols, {}), we need activity_pols
                current_frag_activity_policies, _ = self._generate_rule_based_policies_for_fragment(fragment_id, fragment_info)
            
            # Merge current fragment's activity policies into the main collection
            for act_id, pol_types in current_frag_activity_policies.items():
                for pol_type, rules in pol_types.items():
                    self.all_activity_policies[str(act_id)][pol_type].extend(rules) # Ensure act_id is string

        # Generate dependency policies (currently only rule-based for all dependencies)
        self.all_dependency_policies = self._generate_rule_based_dependency_policies(fragment_dependencies_list)
            
        total_activity_rules = sum(len(rules) for act_pols in self.all_activity_policies.values() for pol_type, rules in act_pols.items())
        total_dependency_rules = sum(len(rules) for dep_pols in self.all_dependency_policies.values() for pol_type, rules in dep_pols.items())
        logger.info(f"Total activity policies generated: {total_activity_rules}")
        logger.info(f"Total dependency policies generated: {total_dependency_rules}")
        
        return dict(self.all_activity_policies), dict(self.all_dependency_policies)

    def get_policy_metrics(self):
        """Returns metrics about the generated policies. Call after generate_policies."""
        metrics = {
            "fragment_activity_policies": {"total_policies": 0, "permissions": 0, "prohibitions": 0, "obligations": 0},
            "fragment_dependency_policies": {"total_policies": 0, "permissions": 0, "prohibitions": 0, "obligations": 0},
            "total": {"total_policies": 0, "permissions": 0, "prohibitions": 0, "obligations": 0}
        }
        
        if hasattr(self, 'all_activity_policies'):
            for act_id, pol_types in self.all_activity_policies.items():
                for pol_type, rules_list in pol_types.items():
                    count = len(rules_list)
                    metrics["fragment_activity_policies"][pol_type + "s"] += count
                    metrics["fragment_activity_policies"]["total_policies"] += count
                    metrics["total"][pol_type + "s"] += count
                    metrics["total"]["total_policies"] += count
        
        if hasattr(self, 'all_dependency_policies'):
            for dep_id, pol_types in self.all_dependency_policies.items():
                for pol_type, rules_list in pol_types.items():
                    count = len(rules_list)
                    metrics["fragment_dependency_policies"][pol_type + "s"] += count
                    metrics["fragment_dependency_policies"]["total_policies"] += count
                    metrics["total"][pol_type + "s"] += count
                    metrics["total"]["total_policies"] += count
        return metrics

    def save_policies(self, output_dir):
        """Saves the generated policies to the specified directory. Call after generate_policies."""
        os.makedirs(output_dir, exist_ok=True)
        activity_policies_path = os.path.join(output_dir, "activity_policies.json")
        dependency_policies_path = os.path.join(output_dir, "dependency_policies.json")

        if hasattr(self, 'all_activity_policies') and self.all_activity_policies:
            with open(activity_policies_path, "w") as f:
                json.dump(self.all_activity_policies, f, indent=2)
            logger.info(f"Activity policies saved to {activity_policies_path}")
        else:
            logger.info("No activity policies to save.")

        if hasattr(self, 'all_dependency_policies') and self.all_dependency_policies:
            with open(dependency_policies_path, "w") as f:
                json.dump(self.all_dependency_policies, f, indent=2)
            logger.info(f"Dependency policies saved to {dependency_policies_path}")
        else:
            logger.info("No dependency policies to save.")


if __name__ == '__main__':
    # Simplified mock data for testing
    mock_model_data = {
        "name": "Test Order Process", "id": "process_123",
        "activities": [
            {"id": "act_A", "name": "Review Order"},
            {"id": "act_B", "name": "Approve Order"}
        ]
    }
    mock_fragments_data_list = [
        {"id": "frag_1", "activities": ["act_A"], "entry_points": [], "exit_points": []},
        {"id": "frag_2", "activities": ["act_B"], "entry_points": [], "exit_points": []}
    ]
    mock_fragment_dependencies_list = [
        {"id": "dep_1", "source": "act_A", "target": "act_B", "type": "sequenceFlow"}
    ]

    print("--- EnhancedPolicyGenerator LLM Test ---")
    if not os.getenv("AZURE_OPENAI_KEY"):
        print("AZURE_OPENAI_KEY not set. Skipping LLM-specific tests.")
    else:
        llm_policy_gen = EnhancedPolicyGenerator(model_data=mock_model_data, fragmentation_strategy='gateway')
        print("\n--- Testing LLM-based Policy Generation ---")
        act_pols, dep_pols = llm_policy_gen.generate_policies(mock_fragments_data_list, mock_fragment_dependencies_list, use_llm=True)
        print("LLM Activity Policies:", json.dumps(act_pols, indent=2))
        print("LLM Dependency Policies:", json.dumps(dep_pols, indent=2))
        metrics = llm_policy_gen.get_policy_metrics()
        print("LLM Metrics:", json.dumps(metrics, indent=2))
        llm_policy_gen.save_policies("./test_llm_policies_output")

    print("\n--- Testing Rule-based Policy Generation (Explicit) ---")
    rule_based_gen = EnhancedPolicyGenerator(model_data=mock_model_data, fragmentation_strategy='gateway')
    # Disable LLM client for explicit rule-based test
    rule_based_gen.llm_client = None 
    act_pols_rb, dep_pols_rb = rule_based_gen.generate_policies(mock_fragments_data_list, mock_fragment_dependencies_list, use_llm=False) # Explicitly use_llm=False
    print("Rule-based Activity Policies:", json.dumps(act_pols_rb, indent=2))
    print("Rule-based Dependency Policies:", json.dumps(dep_pols_rb, indent=2))
    metrics_rb = rule_based_gen.get_policy_metrics()
    print("Rule-based Metrics:", json.dumps(metrics_rb, indent=2))
    rule_based_gen.save_policies("./test_rule_based_policies_output")

