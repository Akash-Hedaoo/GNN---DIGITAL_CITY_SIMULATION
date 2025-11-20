"""
Interactive Traffic Simulation - User can block multiple roads
==============================================================

This script provides an interactive interface where users can:
1. Select multiple roads to block
2. View real-time traffic statistics
3. Reopen roads dynamically
4. See congestion propagation in action
"""

import networkx as nx
from macroscopic_traffic_simulation import (
    MacroscopicTrafficSimulator,
    SimulationConfig
)


def show_road_list(sim: MacroscopicTrafficSimulator, page: int = 1, per_page: int = 20):
    """Display paginated list of roads"""
    edges = list(sim.G.edges(keys=True))
    total_pages = (len(edges) + per_page - 1) // per_page
    
    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, len(edges))
    
    print("\n" + "="*70)
    print(f"🛣️  AVAILABLE ROADS (Page {page}/{total_pages})")
    print("="*70)
    print(f"{'#':<5} {'From':<10} {'To':<10} {'Length(km)':<12} {'Speed':<12} {'Status':<10}")
    print("-" * 70)
    
    for idx in range(start_idx, end_idx):
        u, v, key = edges[idx]
        edge_data = sim.G[u][v][key]
        length = edge_data.get('length', 1.0)
        speed = edge_data.get('speed_limit', 40.0)
        
        # Check if road is closed
        if (u, v, key) in sim.closed_edges:
            status = "🚧 CLOSED"
        else:
            current_time = sim.current_travel_times.get((u, v, key), 0)
            base_time = edge_data.get('base_travel_time', 0)
            if base_time > 0:
                congestion = current_time / base_time
                if congestion > 2.0:
                    status = "🔴 Heavy"
                elif congestion > 1.5:
                    status = "🟡 Medium"
                else:
                    status = "🟢 Clear"
            else:
                status = "🟢 Clear"
        
        print(f"{idx+1:<5} {str(u):<10} {str(v):<10} {length:<12.2f} {speed:<12.1f} {status:<10}")
    
    print("-" * 70)
    print(f"Total roads: {len(edges)} | Showing: {start_idx+1}-{end_idx}")
    print("="*70)


def interactive_simulation():
    """
    Main interactive simulation where users can block/unblock roads.
    """
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "🌆 INTERACTIVE TRAFFIC SIMULATOR" + " "*20 + "║")
    print("╚" + "="*68 + "╝")
    
    # Load graph
    print("\n📂 Loading city graph...")
    try:
        G = nx.read_graphml('city_graph.graphml')
        print(f"✅ Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except FileNotFoundError:
        print("❌ city_graph.graphml not found!")
        print("   Run: python generate_complex_city.py")
        return
    
    # Initialize simulator
    print("\n⚙️  Initializing simulator...")
    config = SimulationConfig(
        base_congestion_multiplier=3.0,
        ripple_decay=0.7,
        ripple_depth=3,
        random_event_probability=0.01
    )
    
    sim = MacroscopicTrafficSimulator(G, config)
    
    # Track closed roads
    closed_roads = []
    current_page = 1
    simulation_running = False
    
    while True:
        print("\n" + "╔" + "="*68 + "╗")
        print("║" + " "*25 + "📋 MAIN MENU" + " "*30 + "║")
        print("╚" + "="*68 + "╝")
        print("\n1️⃣  View Roads (list all roads)")
        print("2️⃣  Block Road (close a road)")
        print("3️⃣  Unblock Road (reopen a road)")
        print("4️⃣  Show Statistics (current traffic state)")
        print("5️⃣  Run Simulation (advance time)")
        print("6️⃣  Auto-Run (simulate 30 minutes)")
        print("7️⃣  Find Path (calculate shortest route)")
        print("8️⃣  Reset Simulation")
        print("0️⃣  Exit")
        
        choice = input("\n👉 Select option: ").strip()
        
        # VIEW ROADS
        if choice == '1':
            while True:
                show_road_list(sim, current_page)
                nav = input("\n[n]ext page, [p]revious page, or [q]uit: ").strip().lower()
                if nav == 'n':
                    current_page += 1
                elif nav == 'p' and current_page > 1:
                    current_page -= 1
                elif nav == 'q':
                    break
        
        # BLOCK ROAD
        elif choice == '2':
            show_road_list(sim, 1, 20)
            
            print("\n💡 How to block a road:")
            print("   • Enter road number from list")
            print("   • Or type 'custom' to enter node IDs")
            print("   • Or type 'random' for random selection")
            
            selection = input("\n👉 Enter choice: ").strip().lower()
            
            edges = list(sim.G.edges(keys=True))
            
            if selection == 'random':
                import random
                available = [e for e in edges if e not in sim.closed_edges]
                if available:
                    u, v, key = random.choice(available)
                    print(f"\n🎲 Random: {u} → {v}")
                    sim.close_road(u, v, key)
                    closed_roads.append((u, v, key))
                    print("✅ Road blocked!")
                else:
                    print("❌ All roads already closed!")
            
            elif selection == 'custom':
                try:
                    source = input("   Source node: ").strip()
                    target = input("   Target node: ").strip()
                    
                    # Try integer conversion
                    try:
                        source = int(source)
                        target = int(target)
                    except:
                        pass
                    
                    if sim.G.has_edge(source, target):
                        key = list(sim.G[source][target].keys())[0]
                        if (source, target, key) not in sim.closed_edges:
                            sim.close_road(source, target, key)
                            closed_roads.append((source, target, key))
                            print(f"✅ Road {source} → {target} blocked!")
                        else:
                            print("❌ Road already closed!")
                    else:
                        print(f"❌ No road from {source} to {target}")
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            else:
                try:
                    idx = int(selection) - 1
                    if 0 <= idx < len(edges):
                        u, v, key = edges[idx]
                        if (u, v, key) not in sim.closed_edges:
                            sim.close_road(u, v, key)
                            closed_roads.append((u, v, key))
                            print(f"✅ Road {u} → {v} blocked!")
                        else:
                            print("❌ Road already closed!")
                    else:
                        print("❌ Invalid road number")
                except ValueError:
                    print("❌ Invalid input")
        
        # UNBLOCK ROAD
        elif choice == '3':
            if not closed_roads:
                print("\n❌ No roads are currently blocked!")
            else:
                print("\n🚧 Currently Blocked Roads:")
                print("-" * 50)
                for idx, (u, v, key) in enumerate(closed_roads, 1):
                    print(f"{idx}. Road {u} → {v} (key={key})")
                print("-" * 50)
                
                try:
                    selection = input("\n👉 Enter road # to unblock (or 'all'): ").strip().lower()
                    
                    if selection == 'all':
                        for u, v, key in closed_roads:
                            sim.reopen_road(u, v, key)
                        print(f"✅ Unblocked {len(closed_roads)} roads!")
                        closed_roads.clear()
                    else:
                        idx = int(selection) - 1
                        if 0 <= idx < len(closed_roads):
                            u, v, key = closed_roads[idx]
                            sim.reopen_road(u, v, key)
                            closed_roads.pop(idx)
                            print(f"✅ Road {u} → {v} reopened!")
                        else:
                            print("❌ Invalid selection")
                except ValueError:
                    print("❌ Invalid input")
        
        # SHOW STATISTICS
        elif choice == '4':
            sim.print_statistics()
            
            if closed_roads:
                print("\n🚧 Blocked Roads:")
                for u, v, key in closed_roads:
                    print(f"   • {u} → {v}")
        
        # RUN SIMULATION (1 step)
        elif choice == '5':
            try:
                minutes = input("\n⏱️  Minutes to simulate (default=1): ").strip()
                minutes = int(minutes) if minutes else 1
                
                for i in range(minutes):
                    sim.step(delta_time=1.0)
                
                print(f"\n✅ Simulated {minutes} minute(s)")
                sim.print_statistics()
            except ValueError:
                print("❌ Invalid number")
        
        # AUTO-RUN
        elif choice == '6':
            try:
                duration = input("\n⏱️  Duration in minutes (default=30): ").strip()
                duration = int(duration) if duration else 30
                
                print(f"\n🚀 Running simulation for {duration} minutes...")
                
                for minute in range(duration):
                    sim.step(delta_time=1.0)
                    
                    if (minute + 1) % 10 == 0:
                        print(f"\n--- Minute {minute + 1}/{duration} ---")
                        stats = sim.get_statistics()
                        print(f"Network delay: {stats['total_network_delay']:.1f}m")
                        print(f"Congested edges: {stats['congested_edges']}/{stats['total_edges']}")
                
                print("\n✅ Simulation complete!")
                sim.print_statistics()
                
            except ValueError:
                print("❌ Invalid input")
        
        # FIND PATH
        elif choice == '7':
            try:
                nodes = list(sim.G.nodes())
                print(f"\n🗺️  Path Finder (Total nodes: {len(nodes)})")
                print(f"   Sample nodes: {nodes[:10]}")
                
                source = input("   Start node: ").strip()
                target = input("   End node: ").strip()
                
                # Try integer conversion
                try:
                    source = int(source)
                    target = int(target)
                except:
                    pass
                
                if source in sim.G.nodes() and target in sim.G.nodes():
                    path = sim.get_shortest_path(source, target)
                    
                    if path:
                        travel_time = sim.get_path_travel_time(path)
                        print(f"\n✅ Path found!")
                        print(f"   Nodes: {len(path)}")
                        print(f"   Route: {' → '.join(map(str, path[:5]))}{'...' if len(path) > 5 else ''}")
                        print(f"   Travel time: {travel_time:.1f} minutes")
                    else:
                        print("\n❌ No path found!")
                else:
                    print("\n❌ Invalid nodes!")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # RESET
        elif choice == '8':
            confirm = input("\n⚠️  Reset simulation? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                # Reopen all roads
                for u, v, key in closed_roads:
                    sim.reopen_road(u, v, key)
                closed_roads.clear()
                
                # Reinitialize
                sim = MacroscopicTrafficSimulator(G, config)
                print("✅ Simulation reset!")
        
        # EXIT
        elif choice == '0':
            print("\n👋 Exiting simulator...")
            
            # Ask if user wants to save data
            save = input("💾 Save training data? (yes/no): ").strip().lower()
            if save in ['yes', 'y']:
                filename = input("   Filename (default: interactive_data.pkl): ").strip()
                filename = filename if filename else 'interactive_data.pkl'
                sim.export_training_data(filename)
            
            print("✅ Goodbye!")
            break
        
        else:
            print("❌ Invalid option. Please try again.")


if __name__ == "__main__":
    interactive_simulation()
