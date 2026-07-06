import networkx as nx

class PlantGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.initialize_graph()
        
    def initialize_graph(self):
        """Initializes the baseline nodes and edges representing the industrial plant."""
        self.G.clear()
        
        # 1. Add Zone Nodes
        zones = {
            'ZONE_1': {'type': 'Zone', 'name': 'Storage Tanks Area', 'hazard_class': 'A', 'area_m2': 600},
            'ZONE_2': {'type': 'Zone', 'name': 'Manifold Piping Area', 'hazard_class': 'B', 'area_m2': 400},
            'ZONE_3': {'type': 'Zone', 'name': 'Coke Oven Battery', 'hazard_class': 'A', 'area_m2': 450},
            'ZONE_4': {'type': 'Zone', 'name': 'Control Room Block', 'hazard_class': 'SAFE', 'area_m2': 200},
            'ZONE_5': {'type': 'Zone', 'name': 'Cooling Towers Area', 'hazard_class': 'B', 'area_m2': 500},
            'ZONE_6': {'type': 'Zone', 'name': 'Electrical Substation', 'hazard_class': 'A', 'area_m2': 300}
        }
        for zone_id, attrs in zones.items():
            self.G.add_node(zone_id, **attrs)
            
        # 2. Add Equipment Nodes
        equipments = {
            'COKE_OVEN_BATT_03': {'type': 'Equipment', 'name': 'Coke Oven Battery 03', 'critical': True},
            'CRUDE_PUMP_101': {'type': 'Equipment', 'name': 'Crude Feed Pump 101', 'critical': False},
            'TRANSFORMER_TR_1A': {'type': 'Equipment', 'name': 'Substation Transformer 1A', 'critical': True}
        }
        for eq_id, attrs in equipments.items():
            self.G.add_node(eq_id, **attrs)
            
        # Link Equipment to Zones
        self.G.add_edge('COKE_OVEN_BATT_03', 'ZONE_3', relation='located_in')
        self.G.add_edge('CRUDE_PUMP_101', 'ZONE_1', relation='located_in')
        self.G.add_edge('TRANSFORMER_TR_1A', 'ZONE_6', relation='located_in')

        # 3. Add Sensor Nodes
        sensors = {
            'GAS_Z3_001': {'type': 'Sensor', 'sensor_type': 'gas', 'unit': 'LEL%'},
            'TEMP_Z3_001': {'type': 'Sensor', 'sensor_type': 'temp', 'unit': 'C'},
            'PRESS_Z3_001': {'type': 'Sensor', 'sensor_type': 'pressure', 'unit': 'bar'},
            'GAS_Z2_001': {'type': 'Sensor', 'sensor_type': 'gas', 'unit': 'LEL%'}
        }
        for s_id, attrs in sensors.items():
            self.G.add_node(s_id, **attrs)
            
        # Link Sensors to Zones
        self.G.add_edge('GAS_Z3_001', 'ZONE_3', relation='located_in')
        self.G.add_edge('TEMP_Z3_001', 'ZONE_3', relation='located_in')
        self.G.add_edge('PRESS_Z3_001', 'ZONE_3', relation='located_in')
        self.G.add_edge('GAS_Z2_001', 'ZONE_2', relation='located_in')

        # 4. Add Regulation Nodes
        regulations = {
            'OISD_105_4.3': {'type': 'Regulation', 'clause': '4.3', 'description': 'Prohibit hot work in gas > 25% LEL; suspend at 20% LEL'},
            'OISD_105_7.3.2': {'type': 'Regulation', 'clause': '7.3.2', 'description': 'Prohibit hot work within 10m of gas source or active leakage'},
            'FACTORIES_ACT_36': {'type': 'Regulation', 'clause': '36', 'description': 'Confined space entry requires gas testing and oxygen checking'}
        }
        for reg_id, attrs in regulations.items():
            self.G.add_node(reg_id, **attrs)

    def add_active_permit(self, permit_id: str, permit_type: str, zone_id: str, regulations: list):
        """Dynamically adds an active permit node and its links to the graph at runtime."""
        self.G.add_node(permit_id, type='Permit', permit_type=permit_type, zone_id=zone_id)
        # Add edge: Permit -> active_in -> Zone
        self.G.add_edge(permit_id, zone_id, relation='active_in')
        
        # Link to governing regulations
        for reg_id in regulations:
            if reg_id in self.G:
                # Add edge: Permit -> governed_by -> Regulation
                self.G.add_edge(permit_id, reg_id, relation='governed_by')

    def remove_expired_permit(self, permit_id: str):
        """Removes a permit node and its associated links from the graph when it expires."""
        if permit_id in self.G and self.G.nodes[permit_id].get('type') == 'Permit':
            self.G.remove_node(permit_id)

    def get_active_permits(self, zone_id: str) -> list:
        """Finds all active permit dictionaries in a specific zone."""
        permits = []
        for n, d in self.G.nodes(data=True):
            if d.get('type') == 'Permit' and self.G.has_edge(n, zone_id):
                permits.append({
                    'permit_id': n,
                    'permit_type': d.get('permit_type'),
                    'zone_id': zone_id
                })
        return permits

    def get_zone_regulations(self, zone_id: str) -> list:
        """Finds all regulation IDs governing active permits in a zone.
        Traverses Permit -> governed_by -> Regulation.
        """
        # 1. Get all active permit node names in the target zone
        permits = [n for n, d in self.G.nodes(data=True)
                   if d.get('type') == 'Permit' and self.G.has_edge(n, zone_id)]
                   
        # 2. Find target regulations connected via 'governed_by' edges
        regulations = []
        for p in permits:
            for _, v, d in self.G.out_edges(p, data=True):
                if d.get('relation') == 'governed_by':
                    # Retrieve the regulation detail
                    reg_node = self.G.nodes[v]
                    regulations.append({
                        'regulation_id': v,
                        'clause': reg_node.get('clause'),
                        'description': reg_node.get('description')
                    })
        return regulations

    def is_equipment_critical(self, zone_id: str) -> bool:
        """Returns True if there is a safety critical equipment inside the zone."""
        # Find all nodes with type='Equipment' that have an edge pointing to the zone
        for n, d in self.G.nodes(data=True):
            if d.get('type') == 'Equipment' and d.get('critical') is True:
                if self.G.has_edge(n, zone_id):
                    return True
        return False
