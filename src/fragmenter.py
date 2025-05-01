"""
fragmenter.py

Provides a Fragmenter class to split a BPMN model into fragments.
"""

class Fragmenter:
    """
    Fragmenter is responsible for splitting a BPMN process model into fragments.
    A 'fragment' is a collection of related activities (and potentially gateways)
    that logically belong together, often separated by BPMN gateways.

    Typical usage:
        fragmenter = Fragmenter(bp_model)
        fragments = fragmenter.fragment_process()
    """

    def __init__(self, bp_model):
        """
        :param bp_model: dict representing a BPMN model with:
            - activities: list of {name, type, ...}
            - gateways: list of {name, type, ...}
            - flows: list of {from, to, type, gateway, ...}
        """
        self.bp_model = bp_model

    def fragment_process(self):
        """
        Splits the BPMN model into fragments based on the presence of gateways
        (especially XOR) and returns a list of fragment dictionaries. Each fragment
        may have keys like 'activities', 'gateways', etc.

        :return: list of fragment dicts, e.g.
            [
              {
                'activities': ['A','B'],
                'gateways': [...],
                ...
              },
              ...
            ]
        """
        fragments = []
        activity_flows = {activity['name']: [] for activity in self.bp_model['activities']}

        # Build adjacency for each activity (i.e., from -> to)
        for flow in self.bp_model['flows']:
            from_act = flow['from']
            to_act   = flow['to']
            if from_act not in activity_flows:
                activity_flows[from_act] = []
            activity_flows[from_act].append(to_act)

        visited = set()

        def traverse(activity_name, current_fragment):
            """
            Depth-first expansion of connected activities, generating fragments
            based on gateway type. An XOR gateway triggers new fragments.
            """
            if activity_name in visited:
                return
            visited.add(activity_name)
            current_fragment['activities'].append(activity_name)

            # Recurse for flows from activity_name
            if activity_name in activity_flows:
                for next_activity in activity_flows[activity_name]:
                    # Check if there's a gateway on the flow
                    gw = self.get_gateway_for_flow(activity_name, next_activity)
                    if gw:
                        # Example: if it's XOR, start a new fragment for that branch
                        if gw['type'].upper() == 'XOR':
                            new_frag = {'activities': [], 'gateways': [gw]}
                            fragments.append(new_frag)
                            traverse(next_activity, new_frag)
                        else:
                            # 'AND' or 'OR' or custom => remain in same fragment
                            if gw not in current_fragment.get('gateways', []):
                                current_fragment.setdefault('gateways', []).append(gw)
                            traverse(next_activity, current_fragment)
                    else:
                        # No gateway => continue in same fragment
                        traverse(next_activity, current_fragment)

        # Identify start activities (just an example: could rely on 'start': True)
        start_activities = [
            act['name'] for act in self.bp_model['activities'] if act.get('start', False)
        ]
        if not start_activities:
            # If no explicit start, pick the first or do your own logic
            start_activities = [self.bp_model['activities'][0]['name']]

        # Start fragmenting from each start activity
        for start_act in start_activities:
            frag = {'activities': [], 'gateways': []}
            fragments.append(frag)
            traverse(start_act, frag)

        # For any not visited
        for act in self.bp_model['activities']:
            if act['name'] not in visited:
                frag = {'activities': [], 'gateways': []}
                fragments.append(frag)
                traverse(act['name'], frag)

        return fragments

    def get_gateway_for_flow(self, from_activity, to_activity):
        """
        Looks for a gateway on the flow from 'from_activity' to 'to_activity'.
        :return: gateway dict if found, else None
        """
        for flow in self.bp_model['flows']:
            if flow['from'] == from_activity and flow['to'] == to_activity:
                if 'gateway' in flow and flow['gateway']:
                    gw_name = flow['gateway']
                    for g in self.bp_model['gateways']:
                        if g['name'] == gw_name:
                            return g
        return None
