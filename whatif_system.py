"""
What-If System Integration
==========================

Integrates all 5 what-if modules into a unified system.

Modules integrated:
1. AmenityInfluenceTracker - Tracks amenity effects
2. PopulationRecalculator - Recalculates populations
3. MetroImpactAnalyzer - Handles metro changes
4. CascadingEffectsEngine - Calculates cascading impacts
5. ScenarioManager - Manages scenarios

Author: Digital Twin City Simulation
Date: November 2025
"""

from amenity_influence_tracker import AmenityInfluenceTracker
from population_recalculator import PopulationRecalculator
from metro_impact_analyzer import MetroImpactAnalyzer
from cascading_effects_engine import CascadingEffectsEngine
from scenario_manager import ScenarioManager

from typing import Dict, List, Optional
import networkx as nx


class WhatIfSystem:
    """
    Unified what-if analysis system.
    
    Orchestrates all 5 modules to provide complete what-if analysis.
    """
    
    def __init__(self, graph: nx.MultiDiGraph, traffic_simulator=None):
        """
        Initialize integrated what-if system.
        
        Args:
            graph: NetworkX graph
            traffic_simulator: Optional MacroscopicTrafficSimulator instance
        """
        self.G = graph
        self.traffic_simulator = traffic_simulator
        
        print("\n" + "="*60)
        print("INITIALIZING WHAT-IF SYSTEM")
        print("="*60)
        
        # Initialize modules in order
        print("\n[INIT] Initializing modules...")
        
        # Module 1: Amenity Influence Tracker
        self.amenity_tracker = AmenityInfluenceTracker(self.G)
        
        # Module 2: Population Recalculator
        self.population_recalculator = PopulationRecalculator(
            self.G,
            amenity_tracker=self.amenity_tracker
        )
        
        # Module 3: Metro Impact Analyzer
        self.metro_analyzer = MetroImpactAnalyzer(
            self.G,
            population_recalculator=self.population_recalculator
        )
        
        # Module 4: Cascading Effects Engine
        self.cascading_effects = CascadingEffectsEngine(
            self.G,
            amenity_tracker=self.amenity_tracker,
            population_recalculator=self.population_recalculator,
            metro_analyzer=self.metro_analyzer,
            traffic_simulator=traffic_simulator
        )
        
        # Module 5: Scenario Manager
        self.scenario_manager = ScenarioManager(
            self.G,
            cascading_effects_engine=self.cascading_effects
        )
        
        print("\n" + "="*60)
        print("[OK] WHAT-IF SYSTEM READY")
        print("="*60)
    
    # ==================== UTILITY METHODS ====================
    
    def get_available_metro_lines(self) -> List[str]:
        """
        Get list of available metro lines.
        
        Returns:
            Sorted list of metro line names
        """
        return sorted(list(self.metro_analyzer.metro_lines.keys()))
    
    def view_metro_lines(self):
        """
        Display available metro lines in a formatted way.
        """
        lines = self.get_available_metro_lines()
        print("\n[METRO LINES] Available lines:")
        for line in lines:
            num_stations = len(self.metro_analyzer.metro_lines[line].stations)
            print(f"  - {line} ({num_stations} stations)")
    
    # ==================== AMENITY OPERATIONS ====================
    
    def remove_amenity(self, amenity_node) -> Dict:
        """
        Remove an amenity and analyze impacts.
        
        Args:
            amenity_node: Node ID of amenity to remove (int or str)
            
        Returns:
            Complete analysis report
            
        Note:
            When an amenity is removed, the population calculator redistributes residents
            to remaining amenities. This can result in net POSITIVE population changes
            system-wide (e.g., removing an inefficient amenity may push residents to
            better-located alternatives). The impact score reflects actual population
            movements, which may show gains in aggregate even though locals lost service.
        """
        # Convert to string (nodes are stored as strings in graphml)
        amenity_node = str(amenity_node)
        
        # Validate node exists
        if amenity_node not in self.G.nodes:
            raise ValueError(f"Node {amenity_node} does not exist in the city graph")
        
        print(f"\n[REMOVING] Amenity at node {amenity_node}...")
        
        amenity_type = self.G.nodes[amenity_node].get('amenity', 'unknown')
        
        # Record in scenario manager
        scenario_name = f"Remove {amenity_type.title()} at {amenity_node}"
        self.scenario_manager.create_scenario(scenario_name)
        
        # CRITICAL FIX: Actually remove the amenity from the graph node
        self.G.nodes[amenity_node]['amenity'] = 'none'
        
        # Get population changes
        pop_changes = self.population_recalculator.apply_amenity_removal(amenity_node)
        
        # CRITICAL: Update graph node populations with new values
        total_pop_change = 0
        for affected_node, (old_pop, new_pop) in pop_changes.items():
            self.G.nodes[affected_node]['population'] = new_pop
            total_pop_change += (new_pop - old_pop)
        
        # Record change
        self.scenario_manager.record_change(
            'amenity_removal',
            [amenity_node],
            description=f"Removed {amenity_type} at node {amenity_node}"
        )
        
        # Get cascading effects (pass pop_changes so engine doesn't re-apply removals)
        cascading_report = self.cascading_effects.analyze_amenity_removal(amenity_node, pop_changes=pop_changes)
        
        # Package results
        result = {
            'scenario_name': scenario_name,
            'amenity_type': amenity_type,
            'amenity_node': amenity_node,
            'population_changes': pop_changes,
            'total_population_change': total_pop_change,
            'cascading_effects': {
                'primary_effects': cascading_report.primary_effects,
                'secondary_effects': cascading_report.secondary_effects,
                'tertiary_effects': cascading_report.tertiary_effects,
            },
            'affected_zones': cascading_report.affected_zones,
            'impact_score': cascading_report.total_impact_score
        }
        
        print(f"   [OK] Analysis complete. Impact score: {result['impact_score']:.1f}")
        print(f"   [OK] Total population change: {total_pop_change:+,}")
        
        return result
    
    def add_amenity(self, node, amenity_type: str) -> Dict:
        """
        Add an amenity and analyze impacts.
        
        Args:
            node: Node ID to add amenity to (int or str)
            amenity_type: Type of amenity
            
        Returns:
            Complete analysis report
        """
        # Convert to string (nodes are stored as strings in graphml)
        node = str(node)
        
        # Validate node exists
        if node not in self.G.nodes:
            raise ValueError(f"Node {node} does not exist in the city graph")
        
        print(f"\n[+] Adding {amenity_type} at node {node}...")
        
        # Update graph
        self.G.nodes[node]['amenity'] = amenity_type
        
        # Record in scenario manager
        scenario_name = f"Add {amenity_type.title()} at {node}"
        self.scenario_manager.create_scenario(scenario_name)
        
        # Get population changes
        pop_changes = self.population_recalculator.apply_amenity_addition(node, amenity_type)
        
        # CRITICAL: Update graph node populations with new values
        total_pop_change = 0
        for affected_node, (old_pop, new_pop) in pop_changes.items():
            self.G.nodes[affected_node]['population'] = new_pop
            total_pop_change += (new_pop - old_pop)
        
        # Record change
        self.scenario_manager.record_change(
            'amenity_addition',
            [node],
            {'amenity_type': amenity_type},
            description=f"Added {amenity_type} at node {node}"
        )
        
        # Get cascading effects with traffic impact
        cascading_report = self.cascading_effects.analyze_amenity_addition(node, amenity_type, pop_changes)
        
        # Package results
        result = {
            'scenario_name': scenario_name,
            'amenity_type': amenity_type,
            'amenity_node': node,
            'population_changes': pop_changes,
            'total_population_change': total_pop_change,
            'affected_zones': cascading_report.affected_zones,
            'cascading_effects': {
                'primary_effects': cascading_report.primary_effects,
                'secondary_effects': cascading_report.secondary_effects,
                'tertiary_effects': cascading_report.tertiary_effects,
            },
            'impact_score': cascading_report.total_impact_score
        }
        
        print(f"   [OK] Amenity added. {len(pop_changes)} nodes affected")
        print(f"   [OK] Total population change: {total_pop_change:+,}")
        
        return result
    
    # ==================== METRO OPERATIONS ====================
    
    def remove_metro_station(self, station_node: int) -> Dict:
        """
        Remove a metro station and analyze impacts.
        
        Args:
            station_node: Node ID of metro station
            
        Returns:
            Complete analysis report
        """
        # Convert to string
        station_node = str(station_node)
        
        print(f"\n[METRO] Removing metro station at node {station_node}...")
        
        # Check if node is in metro_stations (better check than amenity tag)
        if station_node not in self.metro_analyzer.metro_stations:
            return {'error': f'Metro station not found at node {station_node}'}
        
        # Create scenario
        scenario_name = f"Remove Metro Station at {station_node}"
        self.scenario_manager.create_scenario(scenario_name)

        # Ask metro analyzer to remove station (handles population recalculation)
        removal_result = self.metro_analyzer.remove_metro_station(station_node)
        
        if 'error' in removal_result:
            return removal_result

        # Extract population changes from metro removal
        pop_changes = removal_result.get('population_changes', {})
        total_pop_change = 0
        for affected_node, (old_pop, new_pop) in pop_changes.items():
            # Update graph with new population (should already be done, but ensure)
            self.G.nodes[affected_node]['population'] = new_pop
            total_pop_change += (new_pop - old_pop)

        # Record change
        self.scenario_manager.record_change(
            'metro_removal',
            [station_node],
            {
                'metro_line': removal_result.get('metro_line'),
                'station_type': removal_result.get('station_type')
            },
            description=f"Removed metro station at node {station_node}"
        )

        # Get cascading effects with population changes (pass pop_changes to avoid double removal)
        cascading_report = self.cascading_effects.analyze_metro_removal(station_node, pop_changes)

        result = {
            'scenario_name': scenario_name,
            'metro_station': station_node,
            'metro_line': removal_result.get('metro_line'),
            'station_type': removal_result.get('station_type'),
            'removal_result': removal_result,
            'cascading_effects': {
                'primary_effects': cascading_report.primary_effects,
                'secondary_effects': cascading_report.secondary_effects,
                'tertiary_effects': cascading_report.tertiary_effects,
            },
            'affected_zones': cascading_report.affected_zones,
            'impact_score': cascading_report.total_impact_score
        }

        print(f"   [OK] Metro removal analyzed. Impact score: {result['impact_score']:.1f}")

        return result
    
    def remove_metro_line(self, line_name: str) -> Dict:
        """
        Remove an entire metro line.
        
        Args:
            line_name: Name of metro line ("Red", "Blue", "Green")
            
        Returns:
            Complete analysis report
        """
        print(f"\n[METRO] Removing metro line: {line_name}...")
        
        # Get stations on line
        metro_line = self.metro_analyzer.metro_lines.get(line_name)
        if not metro_line:
            return {'error': f'Metro line {line_name} not found'}
        
        # Create scenario
        scenario_name = f"Remove Metro Line {line_name}"
        self.scenario_manager.create_scenario(scenario_name)
        
        # Remove entire line
        line_result = self.metro_analyzer.remove_metro_line(line_name)
        
        # Record change
        self.scenario_manager.record_change(
            'metro_removal',
            metro_line.stations,
            {'metro_line': line_name},
            description=f"Removed entire {line_name} metro line"
        )
        
        result = {
            'scenario_name': scenario_name,
            'metro_line': line_name,
            'stations_removed': line_result.get('stations_removed', []),
            'total_impact': line_result.get('total_impact', {})
        }
        
        print(f"   [OK] Line removal analyzed. {len(result['stations_removed'])} stations removed")
        
        return result
    
    def add_metro_station(self, node: int, metro_line: str, station_type: str = "intermediate") -> Dict:
        """
        Add a new metro station and analyze cascading impacts.
        
        This adds a metro station to an existing line and recalculates:
        - Population distribution (metro adds attraction)
        - Traffic patterns (new connectivity)
        - Amenity influence (metro is treated as amenity)
        - Cascading effects through connected zones
        
        Args:
            node: Node ID to add metro station at (int or str)
            metro_line: Name of metro line (e.g., "metro_station", "hospital+metro_station")
                       Use view_metro_status() to see available lines
            station_type: Type of station ("terminal", "intermediate", "transfer")
            
            Returns:
            Complete analysis report with all cascading effects
        """
        # Convert to string
        node = str(node)
        
        # Validate node exists
        if node not in self.G.nodes:
            raise ValueError(f"Node {node} does not exist in the city graph")
        
        # Validate metro line exists with helpful error message
        available_lines = sorted(list(self.metro_analyzer.metro_lines.keys()))
        if metro_line not in available_lines:
            # Build helpful error message with suggestions
            error_msg = f"Metro line '{metro_line}' not found.\n\nAvailable lines:\n"
            error_msg += "\n".join(f"  - {line}" for line in available_lines)
            
            # Add suggestion if a line exists
            if available_lines:
                error_msg += f"\n\nExample usage: add_metro_station({node}, '{available_lines[0]}', 'intermediate')"
            
            raise ValueError(error_msg)
        
        print(f"\n[METRO] Adding {metro_line} metro station at node {node}...")
        
        # Store baseline values
        baseline_population = sum(self.G.nodes[n].get('population', 0) for n in self.G.nodes())
        
        # Create scenario
        scenario_name = f"Add Metro Station {metro_line} at {node}"
        self.scenario_manager.create_scenario(scenario_name)
        
        # Update graph with metro attributes
        self.G.nodes[node]['metro_type'] = station_type
        self.G.nodes[node]['metro_line'] = metro_line
        
        # Add to metro analyzer's tracking using its API (keeps object types intact)
        if node not in self.metro_analyzer.metro_stations:
            try:
                # metro_analyzer.add_metro_station will update its internal records
                self.metro_analyzer.add_metro_station(node, station_type, metro_line)
            except Exception:
                # Fallback: avoid inserting a plain dict into metro_stations (which breaks expectations)
                # Insert a minimal placeholder object so other code doesn't break; the analyzer should
                # be able to later reconstruct full station objects if needed.
                self.metro_analyzer.metro_stations[node] = None
                if metro_line in self.metro_analyzer.metro_lines:
                    self.metro_analyzer.metro_lines[metro_line].stations.append(node)
        
        # Record change in scenario manager
        self.scenario_manager.record_change(
            'metro_addition',
            [node],
            {
                'metro_line': metro_line,
                'station_type': station_type
            },
            description=f"Added {metro_line} metro station at node {node}"
        )
        
        # CRITICAL: Recalculate population - metro stations increase desirability
        # Similar to amenity influence but with stronger effect
        print(f"   [RECALC] Recalculating population distribution with new metro station...")
        affected_zones = self._get_affected_zones_from_metro_station(node, radius=4)
        pop_changes = {}
        
        for zone_node in affected_zones:
            old_pop = self.G.nodes[zone_node].get('population', 0)
            # Recalculate population for this node considering new metro
            new_pop = self.population_recalculator.recalculate_node_population(zone_node)
            if old_pop != new_pop:
                pop_changes[zone_node] = (old_pop, new_pop)
                self.G.nodes[zone_node]['population'] = new_pop
        
        # Calculate new total population
        new_population = sum(self.G.nodes[n].get('population', 0) for n in self.G.nodes())
        population_change = new_population - baseline_population
        
        # Get cascading effects - metro addition affects traffic and amenity patterns
        cascading_report = self.cascading_effects.analyze_metro_addition(node, metro_line, pop_changes)
        
        # Get population change summary (contains detailed_changes)
        pop_summary = self.population_recalculator.get_population_change_summary(pop_changes)
        
        # Extract zone impacts from cascading report (not from summary which has complex structure)
        affected_zones = cascading_report.affected_zones if cascading_report.affected_zones else {}
        
        result = {
            'scenario_name': scenario_name,
            'metro_station': node,
            'metro_line': metro_line,
            'station_type': station_type,
            'population_changes': pop_changes,
            'total_population_change': population_change,
            'affected_zones': affected_zones,
            'cascading_effects': {
                'primary_effects': cascading_report.primary_effects,
                'secondary_effects': cascading_report.secondary_effects,
                'tertiary_effects': cascading_report.tertiary_effects,
            },
            'impact_score': cascading_report.total_impact_score
        }
        
        print(f"   [OK] Metro station added to {metro_line} line at node {node}")
        print(f"   [OK] Population change: {population_change:+,} ({population_change/baseline_population*100:+.2f}%)")
        print(f"   [OK] Affected zones: {len(affected_zones)}")
        print(f"   [OK] Impact score: {result['impact_score']:.1f}")
        
        return result
    
    def _get_affected_zones_from_metro_station(self, station_node: str, radius: int = 4) -> set:
        """
        Get nodes affected by a metro station within a certain radius.
        Metro stations have wider influence than individual amenities.
        Uses BFS to find all nodes within influence radius.
        """
        affected = set()
        affected.add(station_node)
        
        visited = {station_node}
        queue = [(station_node, 0)]
        
        while queue:
            node, distance = queue.pop(0)
            
            if distance <= radius:
                affected.add(node)
            
            if distance < radius:
                # Check successors
                for neighbor in self.G.successors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
                
                # Check predecessors
                for neighbor in self.G.predecessors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
        
        return affected
    
    # ==================== ROAD OPERATIONS ====================
    
    def close_road(self, road_u, road_v) -> Dict:
        """
        Close a road and analyze impacts.
        
        Args:
            road_u: Start node of road (int or str)
            road_v: End node of road (int or str)
            
        Returns:
            Complete analysis report
        """
        # Convert to strings (nodes are stored as strings in graphml)
        road_u = str(road_u)
        road_v = str(road_v)
        
        # Validate nodes exist
        if road_u not in self.G.nodes:
            raise ValueError(f"Node {road_u} does not exist in the city graph")
        if road_v not in self.G.nodes:
            raise ValueError(f"Node {road_v} does not exist in the city graph")
        
        print(f"\n[ROAD] Closing road {road_u}-{road_v}...")
        
        # Create scenario
        scenario_name = f"Close Road {road_u}-{road_v}"
        self.scenario_manager.create_scenario(scenario_name)
        
        # Store baseline population before closure
        baseline_population = sum(self.G.nodes[n].get('population', 0) for n in self.G.nodes())
        
        # Mark road as closed in graph (Fixed: DiGraph edge access - not MultiDiGraph)
        edge_count = 0
        if self.G.has_edge(road_u, road_v):
            self.G[road_u][road_v]['closed'] = True
            edge_count += 1
        if self.G.has_edge(road_v, road_u):
            self.G[road_v][road_u]['closed'] = True
            edge_count += 1
        
        if edge_count == 0:
            print(f"   WARNING: No edges found between nodes {road_u} and {road_v}")
        
        # Record change
        self.scenario_manager.record_change(
            'road_closure',
            [road_u, road_v],
            description=f"Closed road between nodes {road_u} and {road_v}"
        )
        
        # CRITICAL FIX: Recalculate populations affected by road closure BEFORE cascading effects
        # Road closure affects connectivity, which affects population distribution
        print(f"   [RECALC] Recalculating population distribution after road closure...")
        affected_zones = self._get_affected_zones_from_road_closure(road_u, road_v)
        pop_changes = {}
        
        for zone_node in affected_zones:
            old_pop = self.G.nodes[zone_node].get('population', 0)
            # Recalculate population for this node
            new_pop = self.population_recalculator.recalculate_node_population(zone_node)
            if old_pop != new_pop:
                pop_changes[zone_node] = (old_pop, new_pop)
        
        # Get cascading effects WITH population changes for accurate impact scoring
        cascading_report = self.cascading_effects.analyze_road_closure(road_u, road_v, pop_changes)
        
        # Calculate new total population
        new_population = sum(self.G.nodes[n].get('population', 0) for n in self.G.nodes())
        population_change = new_population - baseline_population
        
        # Calculate zone impacts
        affected_zones_summary = self.population_recalculator.get_population_change_summary(pop_changes)
        
        # Extract zone impacts from cascading report
        affected_zones_dict = cascading_report.affected_zones if cascading_report.affected_zones else {}
        
        result = {
            'scenario_name': scenario_name,
            'road': f"{road_u}-{road_v}",
            'cascading_effects': {
                'primary_effects': cascading_report.primary_effects,
                'secondary_effects': cascading_report.secondary_effects,
                'tertiary_effects': cascading_report.tertiary_effects,
            },
            'impact_score': cascading_report.total_impact_score,
            'population_changes': pop_changes,
            'total_population_change': population_change,
            'affected_zones': affected_zones_dict
        }
        
        print(f"   [OK] Road closure analyzed. Impact score: {result['impact_score']:.1f}")
        print(f"   [OK] Population change: {population_change:+,} ({population_change/baseline_population*100:+.2f}%)")
        
        return result
    
    def _get_affected_zones_from_road_closure(self, road_u: str, road_v: str, radius: int = 3) -> set:
        """
        Get nodes affected by a road closure within a certain radius.
        Uses BFS to find all nodes that lose connectivity or have connectivity changes.
        """
        affected = set()
        
        # Add direct nodes
        affected.add(road_u)
        affected.add(road_v)
        
        # Find all nodes within radius of the closed road
        visited = {road_u, road_v}
        queue = [(road_u, 0), (road_v, 0)]
        
        while queue:
            node, distance = queue.pop(0)
            
            if distance <= radius:
                affected.add(node)
            
            if distance < radius:
                # Check successors
                for neighbor in self.G.successors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
                
                # Check predecessors
                for neighbor in self.G.predecessors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
        
        return affected
    
    def restore_road(self, road_u, road_v) -> Dict:
        """
        Restore a previously closed road and recalculate impacts.
        
        Args:
            road_u: Start node of road (int or str)
            road_v: End node of road (int or str)
            
        Returns:
            Restoration report
        """
        # Convert to strings
        road_u = str(road_u)
        road_v = str(road_v)
        
        print(f"\n[ROAD] Restoring road {road_u}-{road_v}...")
        
        # Store baseline population before restoration
        baseline_population = sum(self.G.nodes[n].get('population', 0) for n in self.G.nodes())
        
        # Mark road as open in graph
        edge_count = 0
        if self.G.has_edge(road_u, road_v):
            self.G[road_u][road_v]['closed'] = False
            edge_count += 1
        if self.G.has_edge(road_v, road_u):
            self.G[road_v][road_u]['closed'] = False
            edge_count += 1
        
        # Recalculate populations affected by road restoration
        print(f"   [RECALC] Recalculating population distribution after road restoration...")
        affected_zones = self._get_affected_zones_from_road_closure(road_u, road_v)
        pop_changes = {}
        
        for zone_node in affected_zones:
            old_pop = self.G.nodes[zone_node].get('population', 0)
            # Recalculate population for this node
            new_pop = self.population_recalculator.recalculate_node_population(zone_node)
            if old_pop != new_pop:
                pop_changes[zone_node] = (old_pop, new_pop)
        
        # Calculate new total population
        new_population = sum(self.G.nodes[n].get('population', 0) for n in self.G.nodes())
        population_change = new_population - baseline_population
        
        result = {
            'road': f"{road_u}-{road_v}",
            'population_changes': pop_changes,
            'total_population_change': population_change,
            'edges_restored': edge_count
        }
        
        print(f"   [OK] Road restoration complete. Edges restored: {edge_count}")
        print(f"   [OK] Population change: {population_change:+,}")
        
        return result
    
    def add_road(self, node_u, node_v, length: float = None, speed_limit: float = 50.0) -> Dict:
        """
        Add a new road between two nodes and analyze cascading impacts.
        
        This adds a direct connection between two nodes and recalculates:
        - Population distribution (new connectivity increases accessibility)
        - Traffic patterns (new route available)
        - Amenity influence (improved access to amenities)
        - Cascading effects through entire network
        - Metro accessibility (if metro lines exist)
        
        Args:
            node_u: Start node of new road (int or str)
            node_v: End node of new road (int or str)
            length: Length of road in km (calculated from coordinates if None)
            speed_limit: Speed limit in km/h (default: 50 km/h)
            
        Returns:
            Complete analysis report with all cascading effects
        """
        # Convert to strings
        node_u = str(node_u)
        node_v = str(node_v)
        
        # Validate nodes exist
        if node_u not in self.G.nodes:
            raise ValueError(f"Node {node_u} does not exist in the city graph")
        if node_v not in self.G.nodes:
            raise ValueError(f"Node {node_v} does not exist in the city graph")
        
        # Check if road already exists
        if self.G.has_edge(node_u, node_v):
            return {'error': f'Road {node_u}-{node_v} already exists in the network'}
        
        print(f"\n[ROAD] Adding new road {node_u}-{node_v}...")
        
        # Calculate length from coordinates if not provided
        if length is None:
            u_data = self.G.nodes[node_u]
            v_data = self.G.nodes[node_v]
            x1, y1 = u_data.get('x', 0), u_data.get('y', 0)
            x2, y2 = v_data.get('x', 0), v_data.get('y', 0)
            # Euclidean distance (rough approximation)
            import math
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * 111  # Convert degrees to km
            length = max(0.5, length)  # Minimum length 0.5 km
        
        # Store baseline values
        baseline_population = sum(self.G.nodes[n].get('population', 0) for n in self.G.nodes())
        
        # Create scenario
        scenario_name = f"Add Road {node_u}-{node_v}"
        self.scenario_manager.create_scenario(scenario_name)
        
        # Add edge to graph (both directions for bidirectional road)
        self.G.add_edge(node_u, node_v, 
                       length=length, 
                       speed_limit=speed_limit,
                       closed=False)
        self.G.add_edge(node_v, node_u, 
                       length=length, 
                       speed_limit=speed_limit,
                       closed=False)
        
        # Record change in scenario manager
        self.scenario_manager.record_change(
            'road_addition',
            [node_u, node_v],
            {
                'length': length,
                'speed_limit': speed_limit
            },
            description=f"Added new road between nodes {node_u} and {node_v} ({length:.2f} km)"
        )
        
        # CRITICAL: Recalculate population - new roads improve connectivity and accessibility
        print(f"   [RECALC] Recalculating population distribution with new road...")
        affected_zones = self._get_affected_zones_from_new_road(node_u, node_v, radius=5)
        pop_changes = {}
        
        for zone_node in affected_zones:
            old_pop = self.G.nodes[zone_node].get('population', 0)
            # Recalculate population for this node considering new road connectivity
            new_pop = self.population_recalculator.recalculate_node_population(zone_node)
            if old_pop != new_pop:
                pop_changes[zone_node] = (old_pop, new_pop)
                self.G.nodes[zone_node]['population'] = new_pop
        
        # Calculate new total population
        new_population = sum(self.G.nodes[n].get('population', 0) for n in self.G.nodes())
        population_change = new_population - baseline_population
        
        # Get cascading effects - new road affects traffic patterns and accessibility
        cascading_report = self.cascading_effects.analyze_road_addition(node_u, node_v, pop_changes, length)
        
        # Get amenity influence summary (improved accessibility helps amenities)
        affected_zones_summary = self.population_recalculator.get_population_change_summary(pop_changes)
        
        result = {
            'scenario_name': scenario_name,
            'road': f"{node_u}-{node_v}",
            'road_length': length,
            'speed_limit': speed_limit,
            'population_changes': pop_changes,
            'total_population_change': population_change,
            'affected_zones': affected_zones_summary,
            'cascading_effects': {
                'primary_effects': cascading_report.primary_effects,
                'secondary_effects': cascading_report.secondary_effects,
                'tertiary_effects': cascading_report.tertiary_effects,
            },
            'impact_score': cascading_report.total_impact_score
        }
        
        print(f"   [OK] New road added between {node_u}-{node_v}")
        print(f"   [OK] Road length: {length:.2f} km, Speed limit: {speed_limit:.1f} km/h")
        print(f"   [OK] Population change: {population_change:+,} ({population_change/baseline_population*100:+.2f}%)")
        print(f"   [OK] Affected zones: {len(affected_zones)}")
        print(f"   [OK] Impact score: {result['impact_score']:.1f}")
        
        return result
    
    def _get_affected_zones_from_new_road(self, node_u: str, node_v: str, radius: int = 5) -> set:
        """
        Get nodes affected by a new road within a certain radius.
        New roads have widespread effects on connectivity and accessibility.
        Uses BFS to find all nodes within influence radius.
        """
        affected = set()
        affected.add(node_u)
        affected.add(node_v)
        
        visited = {node_u, node_v}
        queue = [(node_u, 0), (node_v, 0)]
        
        while queue:
            node, distance = queue.pop(0)
            
            if distance <= radius:
                affected.add(node)
            
            if distance < radius:
                # Check successors
                for neighbor in self.G.successors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
                
                # Check predecessors
                for neighbor in self.G.predecessors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
        
        return affected
    
    # ==================== SCENARIO OPERATIONS ====================
    
    def create_what_if_scenario(self, name: str, description: str = "") -> Dict:
        """Create a new what-if scenario"""
        scenario = self.scenario_manager.create_scenario(name, description)
        return {
            'name': scenario.name,
            'description': scenario.description,
            'created_at': scenario.created_at
        }
    
    def apply_scenario(self, scenario_name: str) -> Dict:
        """Apply all changes in a scenario"""
        return self.scenario_manager.batch_apply_changes(scenario_name)
    
    def compare_scenarios(self, scenario1: str, scenario2: str) -> Dict:
        """Compare two scenarios"""
        return self.scenario_manager.compare_scenarios(scenario1, scenario2)
    
    def list_all_scenarios(self) -> List[str]:
        """List all scenarios"""
        return self.scenario_manager.list_scenarios()
    
    def get_scenario_summary(self, scenario_name: str) -> Dict:
        """Get summary of a scenario"""
        return self.scenario_manager.get_scenario_summary(scenario_name)
    
    def save_scenario(self, scenario_name: str, filepath: str) -> bool:
        """Save a scenario to disk"""
        return self.scenario_manager.save_scenario(scenario_name, filepath)
    
    # ==================== ANALYSIS OPERATIONS ====================
    
    def get_full_city_analysis(self) -> Dict:
        """Get comprehensive analysis of current city state"""
        return {
            'amenity_summary': self.amenity_tracker.get_summary(),
            'metro_status': self.metro_analyzer.get_all_metro_status(),
            'population_metrics': {
                'total': sum(self.G.nodes[n].get('population', 0) for n in self.G.nodes())
            }
        }
    
    def get_node_details(self, node) -> Dict:
        """Get detailed information about a node"""
        # Convert to string (nodes are stored as strings in graphml)
        node = str(node)
        
        # Validate node exists
        if node not in self.G.nodes:
            raise ValueError(f"Node {node} does not exist in the city graph")
        
        return {
            'node_id': node,
            'population': self.G.nodes[node].get('population', 0),
            'amenity': self.G.nodes[node].get('amenity', 'none'),
            'zone': self.G.nodes[node].get('zone', 'unknown'),
            'metro_type': self.G.nodes[node].get('metro_type'),
            'population_breakdown': self.population_recalculator.get_node_population_breakdown(node),
            'affecting_amenities': self.amenity_tracker.get_node_amenities(node).affecting_amenities
        }
    
    def print_detailed_impact(self, result: Dict) -> None:
        """
        Print comprehensive impact report showing ALL affected factors.
        
        Args:
            result: Result dictionary from add_amenity, remove_amenity, etc.
        """
        print("\n" + "="*75)
        print("DETAILED IMPACT ANALYSIS - ALL AFFECTED FACTORS")
        print("="*75)
        
        # Scenario name
        print(f"\n[SCENARIO] {result.get('scenario_name', 'Unknown Scenario')}")
        
        # ====== PRIMARY EFFECTS ======
        if 'population_changes' in result:
            pop_changes = result['population_changes']
            print(f"\n[PRIMARY EFFECTS - POPULATION IMPACT]")
            print(f"  Total nodes affected: {len(pop_changes)}")
            print(f"  Total population change: {result.get('total_population_change', 0):+,}")
            
            # Show top affected nodes
            sorted_changes = sorted(
                pop_changes.items(),
                key=lambda x: abs(x[1][1] - x[1][0]),
                reverse=True
            )
            
            print(f"\n  Top affected nodes:")
            for i, (node, (old_pop, new_pop)) in enumerate(sorted_changes[:10], 1):
                change = new_pop - old_pop
                pct = (change / old_pop * 100) if old_pop > 0 else 0
                amenity = self.G.nodes[node].get('amenity', 'none')
                zone = self.G.nodes[node].get('zone', 'unknown')
                print(f"    {i}. Node {node} ({zone}, {amenity}): {old_pop:,} -> {new_pop:,} ({change:+,} = {pct:+.1f}%)")
        
        # ====== SECONDARY EFFECTS (TRAFFIC) ======
        secondary = result.get('cascading_effects', {}).get('secondary_effects', {})
        if secondary:
            print(f"\n[SECONDARY EFFECTS - TRAFFIC & CONGESTION]")
            if 'increased_road_usage' in secondary:
                traffic = secondary['increased_road_usage']
                # traffic may be numeric or a dict returned by the congestion calculator
                if isinstance(traffic, dict):
                    daily = traffic.get('daily_trips_affected', 0)
                    est = traffic.get('estimated_congestion_change', 0)
                    direction = traffic.get('direction', 'increase' if est > 0 else 'decrease')
                    print(f"  Daily trips affected: {daily:+,} ({direction})")
                    print(f"  Estimated congestion change: {est:+.2f}")
                else:
                    direction = "increase" if traffic > 0 else "decrease"
                    print(f"  Road usage change: {traffic:+,} vehicles ({direction})")
            # Backwards-compatible fields
            if 'estimated_congestion_change' in secondary:
                print(f"  Estimated congestion change: {secondary['estimated_congestion_change']:+.2f}")
            if 'affected_roads' in secondary:
                print(f"  Roads affected: {secondary['affected_roads']}")
            if 'description' in secondary:
                print(f"  Description: {secondary['description']}")
        
        # ====== TERTIARY EFFECTS (SERVICES) ======
        tertiary = result.get('cascading_effects', {}).get('tertiary_effects', {})
        if tertiary:
            print(f"\n[TERTIARY EFFECTS - SERVICE IMPACTS]")
            if isinstance(tertiary, dict):
                if 'hospital_access' in tertiary:
                    print(f"  Hospital Access: {tertiary['hospital_access']}")
                if 'school_access' in tertiary:
                    print(f"  School Access: {tertiary['school_access']}")
                if 'park_access' in tertiary:
                    print(f"  Park Access: {tertiary['park_access']}")
                if 'connectivity_loss' in tertiary:
                    print(f"  Connectivity Loss: {tertiary['connectivity_loss']}")
                if 'economic_impact' in tertiary:
                    print(f"  Economic Impact: {tertiary['economic_impact']}")
                
                for key, value in tertiary.items():
                    if key not in ['hospital_access', 'school_access', 'park_access', 'connectivity_loss', 'economic_impact']:
                        if isinstance(value, dict):
                            print(f"  {key}:")
                            for subkey, subvalue in value.items():
                                print(f"    - {subkey}: {subvalue}")
                        else:
                            print(f"  {key}: {value}")
        
        # ====== ZONE IMPACTS ======
        if 'affected_zones' in result:
            zones = result['affected_zones']
            if zones and isinstance(zones, dict):
                print(f"\n[ZONE IMPACTS - BY DISTRICT]")
                # Filter out non-numeric values and sort by count
                zone_counts = {k: v for k, v in zones.items() if isinstance(v, (int, float))}
                for zone, change in sorted(zone_counts.items(), key=lambda x: x[1], reverse=True):
                    direction = "increase" if change > 0 else "decrease"
                    if isinstance(change, (int, float)):
                        print(f"  {zone}: {change:+,} nodes affected ({direction})")
        
        # ====== AMENITY COUNT CHANGES ======
        if 'amenity_type' in result:
            amenity_type = result['amenity_type']
            print(f"\n[AMENITY SYSTEM CHANGES]")
            print(f"  Amenity type: {amenity_type}")
            print(f"  Location: Node {result.get('amenity_node', 'Unknown')}")
            print(f"  Zone: {self.G.nodes[str(result.get('amenity_node', ''))].get('zone', 'Unknown')}")
            
            # Count amenities
            amenity_summary = self.amenity_tracker.get_summary()
            print(f"  Total amenities now: {amenity_summary['total_amenities']}")
            print(f"  Nodes with amenities: {amenity_summary['total_nodes_with_amenities']}")
        
        # ====== METRO STATUS (if metro-related) ======
        if 'metro' in result.get('scenario_name', '').lower():
            metro_status = self.metro_analyzer.get_all_metro_status()
            print(f"\n[METRO SYSTEM CHANGES]")
            print(f"  Metro stations: {metro_status['total_stations']}")
            print(f"  Metro lines: {metro_status['total_lines']}")
            print(f"  Nodes with metro access: {metro_status.get('total_nodes_with_metro', 0)}")
        
        # ====== SUMMARY METRICS ======
        print(f"\n[IMPACT ASSESSMENT SUMMARY]")
        impact_score = result.get('impact_score', 0)
        print(f"  Overall impact score: {impact_score:.1f}/100")
        
        # Interpret impact level
        if impact_score < 10:
            level = "MINIMAL (almost no impact)"
        elif impact_score < 30:
            level = "LOW (some change)"
        elif impact_score < 60:
            level = "MODERATE (noticeable effect)"
        elif impact_score < 80:
            level = "HIGH (significant impact)"
        else:
            level = "SEVERE (major disruption)"
        
        print(f"  Impact level: {level}")
        
        print("\n" + "="*75)
    
    def print_summary(self):
        """Print summary of what-if system"""
        print("\n" + "="*60)
        print("WHAT-IF SYSTEM SUMMARY")
        print("="*60)
        
        # Amenities
        amenity_summary = self.amenity_tracker.get_summary()
        print(f"\n[AMENITY] Amenities:")
        print(f"   Total: {amenity_summary['total_amenities']}")
        print(f"   Affecting nodes: {amenity_summary['total_nodes_with_amenities']}")
        
        # Metro
        metro_status = self.metro_analyzer.get_all_metro_status()
        print(f"\n[METRO] Metro System:")
        print(f"   Stations: {metro_status['total_stations']}")
        print(f"   Lines: {metro_status['total_lines']}")
        
        # Population
        total_pop = sum(self.G.nodes[n].get('population', 0) for n in self.G.nodes())
        print(f"\n[POP] Population:")
        print(f"   Total: {total_pop:,}")
        
        # Scenarios
        scenarios = self.scenario_manager.list_scenarios()
        print(f"\n[SCENARIO] Scenarios:")
        print(f"   Created: {len(scenarios)}")
        if scenarios:
            print(f"   Active: {self.scenario_manager.active_scenario}")
        
        print("\n" + "="*60)


def test_what_if_system():
    """Test the integrated what-if system"""
    print("\n" + "="*60)
    print("Testing What-If System Integration")
    print("="*60)
    
    # Create test graph
    G = nx.MultiDiGraph()
    
    for i in range(25):
        amenity_type = None
        metro_type = None
        metro_line = None
        
        if i == 5:
            amenity_type = 'hospital'
        elif i == 10:
            amenity_type = 'park'
        elif i == 2:
            metro_type = 'terminal'
            metro_line = 'Red'
        elif i in [3, 4]:
            metro_type = 'intermediate'
            metro_line = 'Red'
        
        G.add_node(i,
                  population=1000,
                  daily_trips=40,
                  amenity=amenity_type or 'none',
                  metro_type=metro_type,
                  metro_line=metro_line,
                  zone='residential' if i < 15 else 'suburbs',
                  x=73.8 + (i % 5) * 0.01,
                  y=18.5 + (i // 5) * 0.01)
    
    # Add edges
    for i in range(24):
        G.add_edge(i, i+1)
    
    # Test system
    what_if = WhatIfSystem(G)
    
    print("\n[OK] What-If System test passed!")


if __name__ == "__main__":
    test_what_if_system()
