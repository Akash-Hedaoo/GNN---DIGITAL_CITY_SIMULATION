"""
Enhanced Interactive What-If Analysis Interface
===============================================

Provides an interactive menu system for what-if analysis.

Features:
1. Remove/add amenities dynamically
2. Modify metro system
3. Block/unblock roads
4. Create and compare scenarios
5. Get impact analysis

Author: Digital Twin City Simulation
Date: November 2025
"""

import networkx as nx
import pickle
import os
import sys

try:
    from whatif_system import WhatIfSystem
    from macroscopic_traffic_simulation import MacroscopicTrafficSimulator, SimulationConfig
except ImportError as e:
    print(f"[FAIL] Error importing modules: {e}")
    print("Make sure all module files are in the same directory")
    sys.exit(1)


class InteractiveWhatIfInterface:
    """Interactive interface for what-if analysis"""
    
    def __init__(self, graph_file: str = "city_graph.graphml"):
        """
        Initialize the interface.
        
        Args:
            graph_file: Path to GraphML file with city graph
        """
        self.graph_file = graph_file
        self.G = None
        self.what_if_system = None
        self.traffic_simulator = None
        
        # Load graph
        self._load_graph()
    
    def _load_graph(self):
        """Load the city graph"""
        if not os.path.exists(self.graph_file):
            print(f"[FAIL] Graph file not found: {self.graph_file}")
            sys.exit(1)
        
        print(f"[LOAD] Loading graph from {self.graph_file}...")
        self.G = nx.read_graphml(self.graph_file)
        print(f"[OK] Loaded {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        # Initialize traffic simulator
        config = SimulationConfig(
            base_congestion_multiplier=3.0,
            ripple_decay=0.7,
            ripple_depth=3,
            time_quantum=1.0,
            recovery_rate=0.85
        )
        self.traffic_simulator = MacroscopicTrafficSimulator(self.G, config)
        
        # Initialize what-if system
        print("\n[TOOL] Initializing what-if system...")
        self.what_if_system = WhatIfSystem(self.G, self.traffic_simulator)
    
    def main_menu(self):
        """Display main menu"""
        while True:
            print("\n" + "="*60)
            print("[CITY] WHAT-IF ANALYSIS SYSTEM")
            print("="*60)
            print("\n1. [+] Add Amenity")
            print("2. [-] Remove Amenity")
            print("3. [METRO] Modify Metro System")
            print("4. [ROAD] Road Management")
            print("5. [SCENARIO] Scenario Management")
            print("6. [CHART] Analysis & Reports")
            print("7. [INFO] View City Status")
            print("8. [SAVE] Save/Load Scenarios")
            print("9. [EXIT] Exit")
            
            choice = input("\nSelect option (1-9): ").strip()
            
            if choice == '1':
                self.add_amenity_menu()
            elif choice == '2':
                self.remove_amenity_menu()
            elif choice == '3':
                self.metro_menu()
            elif choice == '4':
                self.road_menu()
            elif choice == '5':
                self.scenario_menu()
            elif choice == '6':
                self.analysis_menu()
            elif choice == '7':
                self.view_city_status()
            elif choice == '8':
                self.save_load_menu()
            elif choice == '9':
                print("\n[BYE] Goodbye!")
                break
            else:
                print("[FAIL] Invalid option")
    
    def add_amenity_menu(self):
        """Add amenity submenu"""
        print("\n" + "="*60)
        print("[+] ADD AMENITY")
        print("="*60)
        
        max_node = self.G.number_of_nodes() - 1
        print(f"\n[PIN] Valid node numbers: 0 to {max_node}")
        
        try:
            node = int(input("Enter node number: "))
            
            # Show valid range
            if node < 0 or node > max_node:
                print(f"[FAIL] Invalid node number. Valid range: 0-{max_node}")
                return
            
            print("\nAmenity types: hospital, park, school, office, community_center, mall")
            amenity_type = input("Enter amenity type: ").strip().lower()
            
            result = self.what_if_system.add_amenity(node, amenity_type)
            
            # Show detailed impact with all affected factors
            self.what_if_system.print_detailed_impact(result)
            
        except ValueError as e:
            print(f"[FAIL] {e}")
        except Exception as e:
            print(f"[FAIL] Error: {e}")
    
    def remove_amenity_menu(self):
        """Remove amenity submenu"""
        print("\n" + "="*60)
        print("[-] REMOVE AMENITY")
        print("="*60)
        
        max_node = self.G.number_of_nodes() - 1
        print(f"\n[PIN] Valid node numbers: 0 to {max_node}")
        
        try:
            node = int(input("Enter node number with amenity to remove: "))
            
            # Show valid range
            if node < 0 or node > max_node:
                print(f"[FAIL] Invalid node number. Valid range: 0-{max_node}")
                return
            
            result = self.what_if_system.remove_amenity(node)
            
            # Show detailed impact with all affected factors
            self.what_if_system.print_detailed_impact(result)
            
        except ValueError as e:
            print(f"[FAIL] {e}")
        except Exception as e:
            print(f"[FAIL] Error: {e}")
    
    def metro_menu(self):
        """Metro system submenu"""
        print("\n" + "="*60)
        print("[METRO] METRO SYSTEM MANAGEMENT")
        print("="*60)
        
        print("\n1. View metro status")
        print("2. [+] Add metro station")
        print("3. [-] Remove metro station")
        print("4. [-] Remove entire metro line")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            self.view_metro_status()
        elif choice == '2':
            self.add_metro_station()
        elif choice == '3':
            self.remove_metro_station()
        elif choice == '4':
            self.remove_metro_line()
        else:
            print("[FAIL] Invalid option")
    
    def view_metro_status(self):
        """View metro system status"""
        status = self.what_if_system.metro_analyzer.get_all_metro_status()
        
        print("\n" + "="*60)
        print("[METRO] METRO SYSTEM STATUS")
        print("="*60)
        print(f"\nTotal Stations: {status['total_stations']}")
        print(f"Total Lines: {status['total_lines']}")
        
        for line_name, line_info in status['lines'].items():
            print(f"\n[PIN] {line_name} Line:")
            print(f"   Stations: {line_info['stations']}")
            print(f"   Total population at stations: {line_info['total_population_at_stations']:,}")
    
    def add_metro_station(self):
        """Add a new metro station to an existing line"""
        try:
            node = int(input("Enter node number to add metro station: "))
            
            # Show valid range
            max_node = self.G.number_of_nodes() - 1
            if node < 0 or node > max_node:
                print(f"[FAIL] Invalid node number. Valid range: 0-{max_node}")
                return
            
            # Get available metro lines
            available_lines = list(self.what_if_system.metro_analyzer.metro_lines.keys())
            print(f"\n[PIN] Available metro lines:")
            for line in sorted(available_lines):
                print(f"  - {line}")
            metro_line = input("\nEnter metro line name: ").strip()
            
            if metro_line not in available_lines:
                print(f"[FAIL] Metro line '{metro_line}' not found. Available: {', '.join(sorted(available_lines))}")
                return
            
            # Station type options
            print("\nStation types:")
            print("  1. terminal (main station)")
            print("  2. intermediate (regular stop)")
            print("  3. transfer (connects multiple lines)")
            
            station_type_choice = input("Select station type (1-3): ").strip()
            
            station_types = {
                '1': 'terminal',
                '2': 'intermediate',
                '3': 'transfer'
            }
            
            station_type = station_types.get(station_type_choice, 'intermediate')
            
            result = self.what_if_system.add_metro_station(node, metro_line, station_type)
            
            # Show detailed impact with all affected factors
            self.what_if_system.print_detailed_impact(result)
            
        except ValueError as e:
            print(f"[FAIL] {e}")
        except Exception as e:
            print(f"[FAIL] Error: {e}")
    
    def remove_metro_station(self):
        """Remove a metro station"""
        try:
            station_node = int(input("Enter metro station node number: "))
            
            # Show valid range
            max_node = self.G.number_of_nodes() - 1
            if station_node < 0 or station_node > max_node:
                print(f"[FAIL] Invalid node number. Valid range: 0-{max_node}")
                return
            
            result = self.what_if_system.remove_metro_station(station_node)
            
            if 'error' in result:
                print(f"[FAIL] {result['error']}")
            else:
                # Show detailed impact with all affected factors
                self.what_if_system.print_detailed_impact(result)
            
        except ValueError as e:
            print(f"[FAIL] {e}")
        except Exception as e:
            print(f"[FAIL] Error: {e}")
    
    def remove_metro_line(self):
        """Remove entire metro line"""
        try:
            line_name = input("Enter metro line name (Red/Blue/Green): ").strip().capitalize()
            
            result = self.what_if_system.remove_metro_line(line_name)
            
            if 'error' in result:
                print(f"[FAIL] {result['error']}")
            else:
                print(f"\n[OK] Removed {line_name} metro line")
                print(f"   Stations removed: {len(result['stations_removed'])}")
                print(f"   Total impact: {result['total_impact']}")
            
        except Exception as e:
            print(f"[FAIL] Error: {e}")
    
    def road_menu(self):
        """Road management submenu"""
        print("\n" + "="*60)
        print("[ROAD] ROAD MANAGEMENT")
        print("="*60)
        
        print("\n1. View road status")
        print("2. [+] Add new road")
        print("3. [-] Close road")
        print("4. [->] Restore closed road")
        print("5. View closed roads")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            self.view_road_status()
        elif choice == '2':
            self.add_road()
        elif choice == '3':
            self.close_road()
        elif choice == '4':
            self.restore_road()
        elif choice == '5':
            self.view_closed_roads()
        else:
            print("[FAIL] Invalid option")
    
    def view_road_status(self):
        """View sample of road status"""
        edges = list(self.G.edges(keys=True))
        
        print("\n" + "="*60)
        print("[ROAD] ROAD STATUS (Sample)")
        print("="*60)
        print(f"{'From':<8} {'To':<8} {'Length':<10} {'Speed':<10} {'Status':<15}")
        print("-" * 60)
        
        for u, v, key in edges[:20]:
            edge_data = self.G[u][v][key]
            length = edge_data.get('length', 1.0)
            speed = edge_data.get('speed_limit', 40.0)
            status = "[CLOSED]" if edge_data.get('closed', False) else "[OPEN]"
            
            print(f"{u:<8} {v:<8} {length:<10.2f} {speed:<10.1f} {status:<15}")
        
        print(f"\nTotal roads: {len(edges)}")
    
    def close_road(self):
        """Close a road"""
        try:
            u = int(input("Enter start node: "))
            v = int(input("Enter end node: "))
            
            # Show valid range
            max_node = self.G.number_of_nodes() - 1
            if u < 0 or u > max_node or v < 0 or v > max_node:
                print(f"[FAIL] Invalid node numbers. Valid range: 0-{max_node}")
                return
            
            result = self.what_if_system.close_road(u, v)
            
            print(f"\n[OK] Closed road {u}-{v}")
            print(f"   Impact Score: {result['impact_score']:.1f}/100")
            
        except ValueError as e:
            print(f"[FAIL] {e}")
        except Exception as e:
            print(f"[FAIL] Error: {e}")
    
    def view_closed_roads(self):
        """View all closed roads"""
        closed = []
        for u, v, key in self.G.edges(keys=True):
            if self.G[u][v][key].get('closed', False):
                closed.append((u, v))
        
        if closed:
            print("\n" + "="*60)
            print("[CLOSED] CLOSED ROADS")
            print("="*60)
            for u, v in closed:
                print(f"   Road {u}-{v}")
        else:
            print("\n[OK] No roads are currently closed")
    
    def add_road(self):
        """Add a new road between two nodes"""
        try:
            u = int(input("Enter start node: "))
            v = int(input("Enter end node: "))
            
            # Show valid range
            max_node = self.G.number_of_nodes() - 1
            if u < 0 or u > max_node or v < 0 or v > max_node:
                print(f"[FAIL] Invalid node numbers. Valid range: 0-{max_node}")
                return
            
            # Get optional parameters
            speed_input = input("Enter speed limit (km/h) [default 50]: ").strip()
            speed_limit = float(speed_input) if speed_input else 50.0
            
            # Length will be calculated from coordinates
            print("\n[CALC] Calculating road length from node coordinates...")
            
            result = self.what_if_system.add_road(u, v, speed_limit=speed_limit)
            
            if 'error' in result:
                print(f"[FAIL] {result['error']}")
            else:
                # Show detailed impact with all affected factors
                self.what_if_system.print_detailed_impact(result)
            
        except ValueError as e:
            print(f"[FAIL] {e}")
        except Exception as e:
            print(f"[FAIL] Error: {e}")
    
    def restore_road(self):
        """Restore a previously closed road"""
        try:
            u = int(input("Enter start node: "))
            v = int(input("Enter end node: "))
            
            # Show valid range
            max_node = self.G.number_of_nodes() - 1
            if u < 0 or u > max_node or v < 0 or v > max_node:
                print(f"[FAIL] Invalid node numbers. Valid range: 0-{max_node}")
                return
            
            result = self.what_if_system.restore_road(u, v)
            
            print(f"\n[OK] Road {u}-{v} restored")
            print(f"   Edges restored: {result['edges_restored']}")
            print(f"   Population change: {result['total_population_change']:+,}")
            
        except ValueError as e:
            print(f"[FAIL] {e}")
        except Exception as e:
            print(f"[FAIL] Error: {e}")
    
    def scenario_menu(self):
        """Scenario management submenu"""
        print("\n" + "="*60)
        print("[SCENARIO] SCENARIO MANAGEMENT")
        print("="*60)
        
        print("\n1. Create new scenario")
        print("2. List all scenarios")
        print("3. Compare scenarios")
        print("4. Get scenario summary")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            name = input("Enter scenario name: ").strip()
            desc = input("Enter description (optional): ").strip()
            scenario = self.what_if_system.create_what_if_scenario(name, desc)
            print(f"[OK] Created scenario: {name}")
            
        elif choice == '2':
            scenarios = self.what_if_system.list_all_scenarios()
            print("\n[SCENARIO] SCENARIOS:")
            for scenario in scenarios:
                print(f"   - {scenario}")
        
        elif choice == '3':
            scenarios = self.what_if_system.list_all_scenarios()
            if len(scenarios) < 2:
                print("[FAIL] Need at least 2 scenarios to compare")
            else:
                print("\nAvailable scenarios:")
                for i, s in enumerate(scenarios):
                    print(f"   {i+1}. {s}")
                
                try:
                    idx1 = int(input("Select first scenario (number): ")) - 1
                    idx2 = int(input("Select second scenario (number): ")) - 1
                    
                    comparison = self.what_if_system.compare_scenarios(
                        scenarios[idx1], scenarios[idx2]
                    )
                    
                    print(f"\n[COMPARE] COMPARISON:")
                    print(f"   {scenarios[idx1]}: Impact {comparison.get('impact_score1', 0):.1f}")
                    print(f"   {scenarios[idx2]}: Impact {comparison.get('impact_score2', 0):.1f}")
                
                except (ValueError, IndexError):
                    print("[FAIL] Invalid selection")
        
        elif choice == '4':
            scenarios = self.what_if_system.list_all_scenarios()
            if not scenarios:
                print("[FAIL] No scenarios created")
            else:
                print("\nAvailable scenarios:")
                for i, s in enumerate(scenarios):
                    print(f"   {i+1}. {s}")
                
                try:
                    idx = int(input("Select scenario (number): ")) - 1
                    summary = self.what_if_system.get_scenario_summary(scenarios[idx])
                    
                    print(f"\n[SUMMARY] SCENARIO SUMMARY: {summary['name']}")
                    print(f"   Changes: {summary['num_changes']}")
                    print(f"   Impact Score: {summary['impact_score']}")
                
                except (ValueError, IndexError):
                    print("[FAIL] Invalid selection")
        
        else:
            print("[FAIL] Invalid option")
    
    def analysis_menu(self):
        """Analysis and reports submenu"""
        print("\n" + "="*60)
        print("[ANALYSIS] ANALYSIS & REPORTS")
        print("="*60)
        
        print("\n1. Get full city analysis")
        print("2. Analyze specific node")
        print("3. Get system summary")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            analysis = self.what_if_system.get_full_city_analysis()
            print("\n[ANALYSIS] CITY ANALYSIS:")
            print(f"   Amenities: {analysis['amenity_summary']['total_amenities']}")
            print(f"   Metro stations: {analysis['metro_status']['total_stations']}")
            print(f"   Total population: {analysis['population_metrics']['total']:,}")
        
        elif choice == '2':
            try:
                node = int(input("Enter node number: "))
                details = self.what_if_system.get_node_details(node)
                
                print(f"\n[NODE] NODE {node} DETAILS:")
                print(f"   Population: {details['population']:,}")
                print(f"   Zone: {details['zone']}")
                print(f"   Amenity: {details['amenity']}")
                print(f"   Metro type: {details['metro_type']}")
            
            except ValueError:
                print("[FAIL] Invalid input")
        
        elif choice == '3':
            self.what_if_system.print_summary()
        
        else:
            print("[FAIL] Invalid option")
    
    def view_city_status(self):
        """View current city status"""
        print("\n" + "="*60)
        print("[INFO] CITY INFORMATION")
        print("="*60)
        
        print(f"\n[PIN] Total nodes: {self.G.number_of_nodes()} (numbered 0-{self.G.number_of_nodes()-1})")
        print(f"[ROAD] Total roads: {self.G.number_of_edges()}")
        print(f"[POP] Total population: {sum(self.G.nodes[n].get('population', 0) for n in self.G.nodes):,}")
        
        # Count amenities
        amenities_by_type = {}
        for node in self.G.nodes:
            amenity = self.G.nodes[node].get('amenity')
            if amenity:
                amenities_by_type[amenity] = amenities_by_type.get(amenity, 0) + 1
        
        if amenities_by_type:
            print(f"\n[AMENITY] Amenities ({len(amenities_by_type)} types):")
            for amenity_type, count in sorted(amenities_by_type.items()):
                print(f"   {amenity_type}: {count}")
        else:
            print("\n[AMENITY] No amenities found")
        
        # Show example nodes with amenities
        nodes_with_amenities = [n for n in self.G.nodes if self.G.nodes[n].get('amenity')]
        if nodes_with_amenities:
            print(f"\n[EXAMPLE] Example nodes with amenities (first 10):")
            for node in nodes_with_amenities[:10]:
                amenity = self.G.nodes[node].get('amenity')
                pop = self.G.nodes[node].get('population', 0)
                print(f"   Node {node}: {amenity} (population: {pop:,})")
        
        self.what_if_system.print_summary()
    
    def save_load_menu(self):
        """Save/load scenarios submenu"""
        print("\n" + "="*60)
        print("[SAVE] SAVE/LOAD SCENARIOS")
        print("="*60)
        
        print("\n1. Save scenario to file")
        print("2. Load scenario from file")
        
        choice = input("\nSelect option (1-2): ").strip()
        
        if choice == '1':
            scenarios = self.what_if_system.list_all_scenarios()
            if not scenarios:
                print("[FAIL] No scenarios to save")
            else:
                print("\nAvailable scenarios:")
                for i, s in enumerate(scenarios):
                    print(f"   {i+1}. {s}")
                
                try:
                    idx = int(input("Select scenario (number): ")) - 1
                    filepath = input("Enter filepath to save to: ").strip()
                    
                    if self.what_if_system.save_scenario(scenarios[idx], filepath):
                        print(f"[OK] Saved scenario to {filepath}")
                    else:
                        print("[FAIL] Failed to save scenario")
                
                except (ValueError, IndexError):
                    print("[FAIL] Invalid selection")
        
        elif choice == '2':
            filepath = input("Enter filepath to load from: ").strip()
            scenario = self.what_if_system.scenario_manager.load_scenario(filepath)
            
            if scenario:
                print(f"[OK] Loaded scenario: {scenario.name}")
            else:
                print("[FAIL] Failed to load scenario")
        
        else:
            print("[FAIL] Invalid option")


def main():
    """Run interactive interface"""
    print("\n" + "="*60)
    print("[EARTH] WHAT-IF ANALYSIS SYSTEM")
    print("="*60)
    
    interface = InteractiveWhatIfInterface()
    interface.main_menu()


if __name__ == "__main__":
    main()
