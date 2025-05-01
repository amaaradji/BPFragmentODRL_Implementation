"""
bpmn_parser.py

Provides functionality to parse BPMN XML files into the internal JSON format
required by the BPFragmentODRL system.
"""

import os
import json
import xml.etree.ElementTree as ET
from lxml import etree
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BPMNParser:
    """
    BPMNParser converts BPMN XML files to the internal JSON format used by the system.
    
    The internal format is a dictionary with:
    - activities: list of {name, type, ...}
    - gateways: list of {name, type, ...}
    - flows: list of {from, to, type, gateway, ...}
    """
    
    def __init__(self):
        """Initialize the parser with namespaces commonly used in BPMN files."""
        self.namespaces = {
            'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
            'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI',
            'dc': 'http://www.omg.org/spec/DD/20100524/DC',
            'di': 'http://www.omg.org/spec/DD/20100524/DI',
            'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
        }
        
    def parse_file(self, file_path):
        """
        Parse a single BPMN XML file into the internal JSON format.
        
        Args:
            file_path (str): Path to the BPMN XML file
            
        Returns:
            dict: The parsed BPMN model in internal JSON format
        """
        try:
            # Parse the XML file
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Initialize the model structure
            model = {
                'activities': [],
                'gateways': [],
                'flows': [],
                'source_file': os.path.basename(file_path)
            }
            
            # Extract process elements
            processes = root.findall('.//{%s}process' % self.namespaces['bpmn'])
            
            if not processes:
                logger.warning(f"No process found in {file_path}")
                return None
            
            # Process each process element (usually there's just one)
            for process in processes:
                self._extract_activities(process, model)
                self._extract_gateways(process, model)
                self._extract_flows(process, model)
            
            return model
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            return None
    
    def _extract_activities(self, process, model):
        """Extract activities from the process element."""
        # Find all task elements
        tasks = process.findall('.//{%s}task' % self.namespaces['bpmn'])
        
        # Add start events as activities with start=True
        start_events = process.findall('.//{%s}startEvent' % self.namespaces['bpmn'])
        for event in start_events:
            activity = {
                'id': event.get('id'),
                'name': event.get('name') or f"Start_{event.get('id')}",
                'type': 'event',
                'start': True
            }
            model['activities'].append(activity)
        
        # Add end events as activities with end=True
        end_events = process.findall('.//{%s}endEvent' % self.namespaces['bpmn'])
        for event in end_events:
            activity = {
                'id': event.get('id'),
                'name': event.get('name') or f"End_{event.get('id')}",
                'type': 'event',
                'end': True
            }
            model['activities'].append(activity)
        
        # Process regular tasks
        for task in tasks:
            activity = {
                'id': task.get('id'),
                'name': task.get('name') or f"Task_{task.get('id')}",
                'type': 'task'
            }
            model['activities'].append(activity)
    
    def _extract_gateways(self, process, model):
        """Extract gateways from the process element."""
        # Gateway types to look for
        gateway_types = [
            ('exclusiveGateway', 'XOR'),
            ('parallelGateway', 'AND'),
            ('inclusiveGateway', 'OR'),
            ('complexGateway', 'COMPLEX'),
            ('eventBasedGateway', 'EVENT')
        ]
        
        for xml_type, internal_type in gateway_types:
            gateways = process.findall('.//{%s}%s' % (self.namespaces['bpmn'], xml_type))
            for gateway in gateways:
                gw = {
                    'id': gateway.get('id'),
                    'name': gateway.get('name') or f"{internal_type}_{gateway.get('id')}",
                    'type': internal_type
                }
                model['gateways'].append(gw)
    
    def _extract_flows(self, process, model):
        """Extract sequence flows from the process element."""
        flows = process.findall('.//{%s}sequenceFlow' % self.namespaces['bpmn'])
        
        # Create a lookup for gateway IDs
        gateway_ids = {gw['id']: gw for gw in model['gateways']}
        
        for flow in flows:
            source_ref = flow.get('sourceRef')
            target_ref = flow.get('targetRef')
            
            # Determine if source or target is a gateway
            gateway = None
            if source_ref in gateway_ids:
                gateway = gateway_ids[source_ref]['name']
            elif target_ref in gateway_ids:
                gateway = gateway_ids[target_ref]['name']
            
            # Create the flow object
            flow_obj = {
                'id': flow.get('id'),
                'from': source_ref,
                'to': target_ref,
                'type': 'sequence'
            }
            
            # Add gateway information if present
            if gateway:
                flow_obj['gateway'] = gateway
            
            model['flows'].append(flow_obj)
        
        # Also extract message flows if present
        message_flows = process.findall('.//{%s}messageFlow' % self.namespaces['bpmn'])
        for flow in message_flows:
            flow_obj = {
                'id': flow.get('id'),
                'from': flow.get('sourceRef'),
                'to': flow.get('targetRef'),
                'type': 'message'
            }
            model['flows'].append(flow_obj)
    
    def convert_ids_to_names(self, model):
        """
        Convert ID references in flows to name references for better readability.
        This is a post-processing step after parsing.
        """
        # Create lookup dictionaries
        activity_lookup = {act['id']: act['name'] for act in model['activities']}
        gateway_lookup = {gw['id']: gw['name'] for gw in model['gateways']}
        
        # Update flow references
        for flow in model['flows']:
            if flow['from'] in activity_lookup:
                flow['from'] = activity_lookup[flow['from']]
            elif flow['from'] in gateway_lookup:
                flow['from'] = gateway_lookup[flow['from']]
                
            if flow['to'] in activity_lookup:
                flow['to'] = activity_lookup[flow['to']]
            elif flow['to'] in gateway_lookup:
                flow['to'] = gateway_lookup[flow['to']]
        
        return model
    
    def process_directory(self, directory_path, output_dir=None):
        """
        Process all BPMN XML files in a directory and convert them to JSON.
        
        Args:
            directory_path (str): Path to directory containing BPMN XML files
            output_dir (str, optional): Directory to save JSON files. If None, 
                                       files are not saved to disk.
            
        Returns:
            list: List of parsed models
        """
        if not os.path.exists(directory_path):
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        # Create output directory if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Find all XML files
        xml_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.xml') or file.endswith('.bpmn'):
                    xml_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(xml_files)} BPMN files in {directory_path}")
        
        # Parse each file
        models = []
        for file_path in tqdm(xml_files, desc="Parsing BPMN files"):
            model = self.parse_file(file_path)
            if model:
                # Convert IDs to names for better readability
                model = self.convert_ids_to_names(model)
                models.append(model)
                
                # Save to JSON file if output directory is specified
                if output_dir:
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    output_path = os.path.join(output_dir, f"{base_name}.json")
                    with open(output_path, 'w') as f:
                        json.dump(model, f, indent=2)
        
        logger.info(f"Successfully parsed {len(models)} BPMN models")
        return models


def main():
    """
    Example usage of the BPMNParser.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse BPMN XML files to JSON format')
    parser.add_argument('input', help='Input BPMN XML file or directory')
    parser.add_argument('--output', '-o', help='Output JSON file or directory')
    
    args = parser.parse_args()
    
    bpmn_parser = BPMNParser()
    
    if os.path.isdir(args.input):
        # Process directory
        output_dir = args.output if args.output else os.path.join(args.input, 'json')
        models = bpmn_parser.process_directory(args.input, output_dir)
        print(f"Processed {len(models)} BPMN models")
    else:
        # Process single file
        model = bpmn_parser.parse_file(args.input)
        if model:
            model = bpmn_parser.convert_ids_to_names(model)
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(model, f, indent=2)
                print(f"Saved JSON to {args.output}")
            else:
                print(json.dumps(model, indent=2))
        else:
            print("Failed to parse BPMN file")


if __name__ == "__main__":
    main()
