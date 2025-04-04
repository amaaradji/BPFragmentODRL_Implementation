"""
enhanced_fragmenter.py

Provides an enhanced Fragmenter class to split a BPMN model into fragments.
This extends the original fragmenter.py with additional functionality.
"""

import logging
import networkx as nx
import os
import json
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedFragmenter:
    """
    EnhancedFragmenter is responsible for splitting a BPMN process model into fragments.
    A 'fragment' is a collection of related activities (and potentially gateways)
    that logically belong together, often separated by BPMN gateways.
    
    This class extends the original Fragmenter with additional functionality:
    - Multiple fragmentation strategies
    - Fragment dependency tracking
    - Fragment visualization
    - Fragment metrics
    
    Typical usage:
        fragmenter = EnhancedFragmenter(bp_model)
        fragments = fragmenter.fragment_process(strategy='gateway')
    """

    def __init__(self, bp_model):
        """
        Initialize the fragmenter with a BPMN model.
        
        :param bp_model: dict representing a BPMN model with:
            - activities: list of {name, type, ...}
            - gateways: list of {name, type, ...}
            - flows: list of {from, to, type, gateway, ...}
        """
        self.bp_model = bp_model
        self.fragments = []
        self.fragment_dependencies = []
        
        # Create lookup dictionaries for faster access
        self.activity_map = {act['name']: act for act in bp_model.get('activities', [])}
        self.gateway_map = {gw['name']: gw for gw in bp_model.get('gateways', [])}
        
        # Build adjacency lists for activities and gateways
        self.adjacency = self._build_adjacency()
        
        # Create a graph representation for advanced algorithms
        self.graph = self._build_graph()

    def _build_adjacency(self):
        """
        Build adjacency lists for all nodes (activities and gateways).
        
        :return: dict mapping node names to lists of successor node names
        """
        adjacency = defaultdict(list)
        
        for flow in self.bp_model.get('flows', []):
            from_node = flow['from']
            to_node = flow['to']
            adjacency[from_node].append(to_node)
            
        return adjacency
    
    def _build_graph(self):
        """
        Build a NetworkX directed graph representation of the BPMN model.
        
        :return: NetworkX DiGraph
        """
        G = nx.DiGraph()
        
        # Add activities as nodes
        for act in self.bp_model.get('activities', []):
            node_attrs = act.copy()
            node_attrs['node_type'] = 'activity'  # Use node_type instead of type to avoid conflict
            G.add_node(act['name'], **node_attrs)
            
        # Add gateways as nodes
        for gw in self.bp_model.get('gateways', []):
            node_attrs = gw.copy()
            node_attrs['node_type'] = 'gateway'  # Use node_type instead of type to avoid conflict
            G.add_node(gw['name'], **node_attrs)
            
        # Add flows as edges
        for flow in self.bp_model.get('flows', []):
            G.add_edge(flow['from'], flow['to'], **flow)
            
        return G
    
    def fragment_process(self, strategy='gateway'):
        """
        Split the BPMN model into fragments based on the specified strategy.
        
        :param strategy: Fragmentation strategy to use:
                        'gateway' - Split at XOR gateways (default)
                        'activity' - Each activity is its own fragment
                        'connected' - Connected components
                        'hierarchical' - Hierarchical decomposition
        :return: list of fragment dicts
        """
        if strategy == 'gateway':
            return self._fragment_by_gateways()
        elif strategy == 'activity':
            return self._fragment_by_activities()
        elif strategy == 'connected':
            return self._fragment_by_connected_components()
        elif strategy == 'hierarchical':
            return self._fragment_by_hierarchy()
        else:
            logger.warning(f"Unknown fragmentation strategy: {strategy}. Using 'gateway' strategy.")
            return self._fragment_by_gateways()
    
    def _fragment_by_gateways(self):
        """
        Split the BPMN model into fragments based on the presence of gateways
        (especially XOR) and returns a list of fragment dictionaries.
        
        This is similar to the original fragmenter.py implementation but with
        enhanced tracking of fragment dependencies.
        
        :return: list of fragment dicts
        """
        self.fragments = []
        activity_flows = self.adjacency
        visited = set()
        
        def traverse(activity_name, current_fragment):
            """
            Depth-first expansion of connected activities, generating fragments
            based on gateway type. An XOR gateway triggers new fragments.
            """
            if activity_name in visited:
                return
            visited.add(activity_name)
            
            # Add to current fragment's activities if it's an activity
            if activity_name in self.activity_map:
                current_fragment['activities'].append(activity_name)
            
            # Recurse for flows from activity_name
            for next_node in activity_flows.get(activity_name, []):
                # Check if there's a gateway on the flow
                gw = self._get_gateway_for_flow(activity_name, next_node)
                
                if gw:
                    # If it's XOR, start a new fragment for that branch
                    if gw['type'].upper() == 'XOR':
                        # Record dependency between fragments
                        new_frag_id = len(self.fragments) + 1
                        current_frag_id = self.fragments.index(current_fragment) if current_fragment in self.fragments else -1
                        
                        if current_frag_id >= 0:
                            self.fragment_dependencies.append({
                                'from_fragment': current_frag_id,
                                'to_fragment': new_frag_id,
                                'type': 'xor_split',
                                'gateway': gw['name']
                            })
                        
                        # Create new fragment
                        new_frag = {
                            'id': new_frag_id,
                            'activities': [],
                            'gateways': [gw['name']],
                            'entry_points': [next_node]
                        }
                        self.fragments.append(new_frag)
                        traverse(next_node, new_frag)
                    else:
                        # 'AND' or 'OR' or custom => remain in same fragment
                        if gw['name'] not in current_fragment.get('gateways', []):
                            current_fragment.setdefault('gateways', []).append(gw['name'])
                        traverse(next_node, current_fragment)
                else:
                    # No gateway => continue in same fragment
                    traverse(next_node, current_fragment)
        
        # Identify start activities
        start_activities = [
            act['name'] for act in self.bp_model.get('activities', []) 
            if act.get('start', False) or 'Start' in act.get('name', '')
        ]
        
        if not start_activities:
            # If no explicit start, find nodes with no incoming edges
            start_activities = [node for node, in_degree in self.graph.in_degree() 
                               if in_degree == 0 and node in self.activity_map]
            
            # If still no start activities, pick the first activity
            if not start_activities and self.bp_model.get('activities', []):
                start_activities = [self.bp_model['activities'][0]['name']]
        
        # Start fragmenting from each start activity
        for start_act in start_activities:
            frag = {
                'id': len(self.fragments),
                'activities': [],
                'gateways': [],
                'entry_points': [start_act]
            }
            self.fragments.append(frag)
            traverse(start_act, frag)
        
        # For any not visited activities
        for act in self.bp_model.get('activities', []):
            if act['name'] not in visited:
                frag = {
                    'id': len(self.fragments),
                    'activities': [],
                    'gateways': [],
                    'entry_points': [act['name']]
                }
                self.fragments.append(frag)
                traverse(act['name'], frag)
        
        # Calculate exit points for each fragment
        self._calculate_exit_points()
        
        return self.fragments
    
    def _fragment_by_activities(self):
        """
        Create a separate fragment for each activity.
        This is the most fine-grained fragmentation approach.
        
        :return: list of fragment dicts
        """
        self.fragments = []
        
        for i, act in enumerate(self.bp_model.get('activities', [])):
            frag = {
                'id': i,
                'activities': [act['name']],
                'gateways': [],
                'entry_points': [act['name']],
                'exit_points': []
            }
            
            # Find outgoing flows from this activity
            for flow in self.bp_model.get('flows', []):
                if flow['from'] == act['name']:
                    frag['exit_points'].append(flow['to'])
                    
                    # Record dependency to other fragments
                    if flow['to'] in self.activity_map:
                        # Find the fragment that will contain this activity
                        to_frag_id = i + 1 if self.bp_model['activities'].index(self.activity_map[flow['to']]) > i else i - 1
                        
                        self.fragment_dependencies.append({
                            'from_fragment': i,
                            'to_fragment': to_frag_id,
                            'type': 'sequence',
                            'gateway': None
                        })
            
            self.fragments.append(frag)
        
        return self.fragments
    
    def _fragment_by_connected_components(self):
        """
        Split the BPMN model into fragments based on connected components.
        
        :return: list of fragment dicts
        """
        self.fragments = []
        
        # Find connected components in the undirected version of the graph
        undirected_graph = self.graph.to_undirected()
        components = list(nx.connected_components(undirected_graph))
        
        for i, component in enumerate(components):
            activities = [node for node in component if node in self.activity_map]
            gateways = [node for node in component if node in self.gateway_map]
            
            frag = {
                'id': i,
                'activities': activities,
                'gateways': gateways,
                'entry_points': [],
                'exit_points': []
            }
            
            # Find entry and exit points
            for node in component:
                # Entry points are nodes with incoming edges from outside the component
                for pred in self.graph.predecessors(node):
                    if pred not in component:
                        frag['entry_points'].append(node)
                        
                        # Record dependency from other fragments
                        for j, other_frag in enumerate(self.fragments):
                            if pred in other_frag['activities'] or pred in other_frag['gateways']:
                                self.fragment_dependencies.append({
                                    'from_fragment': j,
                                    'to_fragment': i,
                                    'type': 'sequence',
                                    'gateway': None
                                })
                
                # Exit points are nodes with outgoing edges to outside the component
                for succ in self.graph.successors(node):
                    if succ not in component:
                        frag['exit_points'].append(node)
            
            self.fragments.append(frag)
        
        return self.fragments
    
    def _fragment_by_hierarchy(self):
        """
        Split the BPMN model into fragments based on hierarchical decomposition.
        This uses the RPST (Refined Process Structure Tree) concept.
        
        :return: list of fragment dicts
        """
        self.fragments = []
        
        # Simplified implementation - in a real RPST we would identify:
        # 1. Trivial fragments (single edges)
        # 2. Polygons (sequences)
        # 3. Bonds (parallel gateways)
        # 4. Rigids (everything else)
        
        # For this implementation, we'll use a simplified approach:
        # 1. Identify sequences between split/join gateways
        # 2. Identify parallel regions (between AND split/join)
        # 3. Identify conditional regions (between XOR split/join)
        
        # First, identify gateway pairs (split/join)
        gateway_pairs = self._identify_gateway_pairs()
        
        # Create fragments for each gateway pair
        fragment_id = 0
        for pair in gateway_pairs:
            split_gw, join_gw = pair
            
            # Find all nodes between these gateways
            subgraph_nodes = self._find_nodes_between(split_gw, join_gw)
            
            activities = [node for node in subgraph_nodes if node in self.activity_map]
            gateways = [node for node in subgraph_nodes if node in self.gateway_map]
            
            # Add the split and join gateways
            if split_gw not in gateways:
                gateways.append(split_gw)
            if join_gw not in gateways:
                gateways.append(join_gw)
            
            frag = {
                'id': fragment_id,
                'activities': activities,
                'gateways': gateways,
                'entry_points': [split_gw],
                'exit_points': [join_gw],
                'type': self.gateway_map[split_gw]['type'] if split_gw in self.gateway_map else 'SEQUENCE'
            }
            
            self.fragments.append(frag)
            fragment_id += 1
        
        # If we haven't covered all activities, create additional fragments
        covered_activities = set()
        for frag in self.fragments:
            covered_activities.update(frag['activities'])
        
        remaining_activities = [act['name'] for act in self.bp_model.get('activities', []) 
                               if act['name'] not in covered_activities]
        
        if remaining_activities:
            # Create a fragment for the remaining activities
            frag = {
                'id': fragment_id,
                'activities': remaining_activities,
                'gateways': [],
                'entry_points': [],
                'exit_points': []
            }
            
            # Find entry and exit points
            for act in remaining_activities:
                # Check if this activity has predecessors outside the fragment
                has_external_pred = False
                for pred in self.graph.predecessors(act):
                    if pred not in remaining_activities and pred not in frag['gateways']:
                        has_external_pred = True
                        frag['entry_points'].append(act)
                        break
                
                # If no external predecessors but no predecessors at all, it's an entry point
                if not has_external_pred and self.graph.in_degree(act) == 0:
                    frag['entry_points'].append(act)
                
                # Check if this activity has successors outside the fragment
                for succ in self.graph.successors(act):
                    if succ not in remaining_activities and succ not in frag['gateways']:
                        frag['exit_points'].append(act)
                        break
            
            self.fragments.append(frag)
        
        # Calculate fragment dependencies
        self._calculate_fragment_dependencies()
        
        return self.fragments
    
    def _identify_gateway_pairs(self):
        """
        Identify matching split/join gateway pairs in the BPMN model.
        
        :return: list of (split_gateway, join_gateway) tuples
        """
        pairs = []
        split_gateways = {}
        
        # Identify split gateways (more than one outgoing edge)
        for node, out_degree in self.graph.out_degree():
            if node in self.gateway_map and out_degree > 1:
                gw_type = self.gateway_map[node]['type']
                split_gateways.setdefault(gw_type, []).append(node)
        
        # For each split gateway, try to find a matching join gateway
        for gw_type, splits in split_gateways.items():
            for split_gw in splits:
                # Find potential join gateways of the same type
                join_candidates = [node for node, in_degree in self.graph.in_degree() 
                                  if node in self.gateway_map 
                                  and self.gateway_map[node]['type'] == gw_type
                                  and in_degree > 1]
                
                # Find the "closest" join gateway
                for join_gw in join_candidates:
                    # Check if there's a path from split to join
                    if nx.has_path(self.graph, split_gw, join_gw):
                        # Check if this is a proper split/join pair
                        # (all paths from split should eventually reach join)
                        is_proper_pair = True
                        for succ in self.graph.successors(split_gw):
                            if not nx.has_path(self.graph, succ, join_gw):
                                is_proper_pair = False
                                break
                        
                        if is_proper_pair:
                            pairs.append((split_gw, join_gw))
                            break
        
        return pairs
    
    def _find_nodes_between(self, start_node, end_node):
        """
        Find all nodes on paths between start_node and end_node.
        
        :param start_node: Starting node
        :param end_node: Ending node
        :return: set of node names
        """
        nodes = set()
        
        def dfs(current, visited):
            if current == end_node:
                return True
            
            visited.add(current)
            
            for succ in self.graph.successors(current):
                if succ not in visited:
                    if dfs(succ, visited.copy()):
                        nodes.add(succ)
                        return True
            
            return False
        
        dfs(start_node, set())
        return nodes
    
    def _calculate_exit_points(self):
        """
        Calculate exit points for each fragment.
        Exit points are activities or gateways that have flows to nodes outside the fragment.
        """
        for frag in self.fragments:
            frag['exit_points'] = []
            
            # Check each activity and gateway in the fragment
            for node in frag['activities'] + frag.get('gateways', []):
                # Look for successors outside this fragment
                for succ in self.adjacency.get(node, []):
                    is_external = True
                    
                    # Check if successor is in this fragment
                    if succ in frag['activities'] or succ in frag.get('gateways', []):
                        is_external = False
                    
                    if is_external and node not in frag['exit_points']:
                        frag['exit_points'].append(node)
    
    def _calculate_fragment_dependencies(self):
        """
        Calculate dependencies between fragments based on flows between them.
        """
        self.fragment_dependencies = []
        
        # Create a mapping from nodes to fragments
        node_to_fragment = {}
        for i, frag in enumerate(self.fragments):
            for node in frag['activities'] + frag.get('gateways', []):
                node_to_fragment[node] = i
        
        # Check each flow for inter-fragment dependencies
        for flow in self.bp_model.get('flows', []):
            from_node = flow['from']
            to_node = flow['to']
            
            if from_node in node_to_fragment and to_node in node_to_fragment:
                from_frag = node_to_fragment[from_node]
                to_frag = node_to_fragment[to_node]
                
                if from_frag != to_frag:
                    # This is an inter-fragment flow
                    dependency = {
                        'from_fragment': from_frag,
                        'to_fragment': to_frag,
                        'type': flow.get('type', 'sequence'),
                        'gateway': flow.get('gateway', None)
                    }
                    
                    # Check if this dependency already exists
                    if not any(d['from_fragment'] == from_frag and 
                              d['to_fragment'] == to_frag for d in self.fragment_dependencies):
                        self.fragment_dependencies.append(dependency)
    
    def _get_gateway_for_flow(self, from_node, to_node):
        """
        Check if there's a gateway on the flow between from_node and to_node.
        
        :param from_node: Source node name
        :param to_node: Target node name
        :return: gateway dict if found, else None
        """
        for flow in self.bp_model.get('flows', []):
            if flow['from'] == from_node and flow['to'] == to_node:
                if 'gateway' in flow and flow['gateway']:
                    gw_name = flow['gateway']
                    if gw_name in self.gateway_map:
                        return self.gateway_map[gw_name]
        return None
    
    def get_fragment_metrics(self):
        """
        Calculate metrics for each fragment.
        
        :return: dict with fragment metrics
        """
        metrics = {
            'total_fragments': len(self.fragments),
            'total_dependencies': len(self.fragment_dependencies),
            'fragments': []
        }
        
        for frag in self.fragments:
            frag_metrics = {
                'id': frag['id'],
                'activity_count': len(frag['activities']),
                'gateway_count': len(frag.get('gateways', [])),
                'entry_points': len(frag.get('entry_points', [])),
                'exit_points': len(frag.get('exit_points', [])),
                'incoming_dependencies': sum(1 for d in self.fragment_dependencies if d['to_fragment'] == frag['id']),
                'outgoing_dependencies': sum(1 for d in self.fragment_dependencies if d['from_fragment'] == frag['id'])
            }
            metrics['fragments'].append(frag_metrics)
        
        return metrics
    
    def visualize_fragments(self, output_file=None):
        """
        Generate a visualization of the fragments and their dependencies.
        
        :param output_file: Optional file path to save the visualization
        :return: matplotlib figure or None if visualization failed
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            # Create a new figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Draw fragments as rectangles
            for i, frag in enumerate(self.fragments):
                x = i % 3 * 4
                y = i // 3 * 4
                width = 3
                height = 3
                
                rect = patches.Rectangle((x, y), width, height, linewidth=1, 
                                        edgecolor='blue', facecolor='lightblue', alpha=0.7)
                ax.add_patch(rect)
                
                # Add fragment label
                ax.text(x + width/2, y + height - 0.2, f"Fragment {frag['id']}", 
                       ha='center', va='center', fontsize=10, fontweight='bold')
                
                # Add activity count
                ax.text(x + width/2, y + height/2, f"{len(frag['activities'])} activities\n{len(frag.get('gateways', []))} gateways", 
                       ha='center', va='center', fontsize=8)
                
                # Add entry/exit points
                entry_text = ", ".join(frag.get('entry_points', [])[:2])
                if len(frag.get('entry_points', [])) > 2:
                    entry_text += "..."
                
                exit_text = ", ".join(frag.get('exit_points', [])[:2])
                if len(frag.get('exit_points', [])) > 2:
                    exit_text += "..."
                
                ax.text(x + width/2, y + 0.4, f"Entry: {entry_text}\nExit: {exit_text}", 
                       ha='center', va='center', fontsize=6)
            
            # Draw dependencies as arrows
            for dep in self.fragment_dependencies:
                from_frag = dep['from_fragment']
                to_frag = dep['to_fragment']
                
                # Calculate positions
                from_x = from_frag % 3 * 4 + 3  # Right side of from_fragment
                from_y = from_frag // 3 * 4 + 1.5  # Middle height
                
                to_x = to_frag % 3 * 4  # Left side of to_fragment
                to_y = to_frag // 3 * 4 + 1.5  # Middle height
                
                # Draw arrow
                ax.annotate("", xy=(to_x, to_y), xytext=(from_x, from_y),
                           arrowprops=dict(arrowstyle="->", color="red", lw=1))
                
                # Add dependency type
                mid_x = (from_x + to_x) / 2
                mid_y = (from_y + to_y) / 2
                ax.text(mid_x, mid_y, dep.get('type', 'sequence'), 
                       ha='center', va='center', fontsize=6, color='red')
            
            # Set axis limits and remove ticks
            ax.set_xlim(-1, 12)
            ax.set_ylim(-1, 12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Set title
            ax.set_title(f"Process Fragments: {len(self.fragments)} fragments, {len(self.fragment_dependencies)} dependencies")
            
            # Save to file if specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Fragment visualization saved to {output_file}")
            
            return fig
        except ImportError:
            logger.warning("Matplotlib is required for visualization. Install with 'pip install matplotlib'.")
            return None
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return None

    def save_fragments(self, output_dir):
        """
        Save fragments to JSON files in the specified directory.
        
        :param output_dir: Directory to save fragment files
        :return: List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        # Save fragments
        for i, fragment in enumerate(self.fragments):
            file_path = os.path.join(output_dir, f"fragment_{i+1}.json")
            with open(file_path, 'w') as f:
                json.dump(fragment, f, indent=2)
            saved_files.append(file_path)
        
        # Save fragment dependencies
        dependencies_path = os.path.join(output_dir, "fragment_dependencies.json")
        with open(dependencies_path, 'w') as f:
            json.dump(self.fragment_dependencies, f, indent=2)
        saved_files.append(dependencies_path)
        
        # Save fragment visualization if available
        if hasattr(self, 'fragment_graph'):
            try:
                import networkx as nx
                import matplotlib.pyplot as plt
                
                # Create a visualization of fragments and their dependencies
                plt.figure(figsize=(12, 8))
                pos = nx.spring_layout(self.fragment_graph)
                nx.draw(self.fragment_graph, pos, with_labels=True, node_color='lightblue', 
                        node_size=1500, edge_color='gray', arrows=True)
                
                # Add edge labels for dependency types
                edge_labels = {(u, v): d.get('type', '') for u, v, d in self.fragment_graph.edges(data=True)}
                nx.draw_networkx_edge_labels(self.fragment_graph, pos, edge_labels=edge_labels)
                
                # Save the visualization
                viz_path = os.path.join(output_dir, "fragment_visualization.png")
                plt.savefig(viz_path)
                plt.close()
                saved_files.append(viz_path)
            except ImportError:
                logger.warning("NetworkX or matplotlib not available, skipping visualization")
        
        return saved_files
