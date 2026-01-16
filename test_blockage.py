import traci_control
import traci
import os
import sys

def test_blockage():
    print("Testing blockage simulation...")
    # Start SUMO with GUI or headless just for this test
    sumo_cmd = ["sumo", "-c", "outputs/metro.sumocfg", "--no-step-log", "true", "--no-warnings", "true"]
    
    try:
        traci.start(sumo_cmd)
        
        # Run a few steps to warm up
        for _ in range(100):
            traci.simulationStep()
            
        # Pick an edge to block
        edges = traci.edge.getIDList()
        valid_edges = [e for e in edges if not e.startswith(":")]
        
        if not valid_edges:
            print("No valid edges found to block.")
            return

        target_edge = valid_edges[0]
        print(f"Blocking edge: {target_edge}")
        
        # Call the routine
        affected_edges = traci_control.simulate_blockage(target_edge, duration_steps=100) # Shorten duration for test
        
        print(f"Blockage test complete. Affected edges count: {len(affected_edges)}")
        traci.close()
        
    except Exception as e:
        print(f"Test failed: {e}")
        try:
            traci.close()
        except:
            pass

if __name__ == "__main__":
    test_blockage()
