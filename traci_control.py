import os
import sys
import traci
import csv
import logging

# Setup logging
log_file = os.path.join("logs", "traci.log")
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

OUTPUT_FILE = os.path.join("outputs", "sumo_training_data.csv")

def run_simulation():
    sumo_log = os.path.join("logs", "sumo.log")
    sumo_cmd = ["sumo", "-c", "outputs/metro.sumocfg", "--no-step-log", "true", "--no-warnings", "true", "--log", sumo_log] # Headless with log

    try:
        logging.info("Starting SUMO simulation...")
        traci.start(sumo_cmd)
        
        with open(OUTPUT_FILE, 'w', newline='') as csvfile:
            fieldnames = ['step', 'edge_id', 'occupancy', 'speed']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            step = 0
            while step < 3600:
                traci.simulationStep()
                
                if step % 10 == 0:
                    edge_ids = traci.edge.getIDList()
                    for edge_id in edge_ids:
                        # Skip internal edges if needed, but often good to verify
                        if edge_id.startswith(":"):
                            continue
                            
                        occupancy = traci.edge.getLastStepOccupancy(edge_id)
                        speed = traci.edge.getLastStepMeanSpeed(edge_id)
                        
                        writer.writerow({
                            'step': step,
                            'edge_id': edge_id,
                            'occupancy': occupancy,
                            'speed': speed
                        })
                
                step += 1
                
        logging.info("Simulation finished after 3600 steps.")
        traci.close()
        
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        try:
            traci.close()
        except:
            pass
        sys.exit(1)

def simulate_blockage(edge_id, duration_steps=500):
    """
    Simulates a blockage on a given edge.
    """
    logging.info(f"Simulating blockage on edge {edge_id}")
    try:
        # Get lanes for this edge to reduce speed on all lanes
        num_lanes = traci.edge.getLaneNumber(edge_id)
        lane_ids = [f"{edge_id}_{i}" for i in range(num_lanes)]
        
        # 1. Set edge speed to near zero (via lanes)
        original_speeds = {}
        for lane_id in lane_ids:
            original_speeds[lane_id] = traci.lane.getMaxSpeed(lane_id)
            traci.lane.setMaxSpeed(lane_id, 0.1)
        
        initial_congestion = {}
        all_edges = traci.edge.getIDList()
        for e in all_edges:
             if not e.startswith(":"):
                 initial_congestion[e] = traci.edge.getLastStepOccupancy(e)
        
        # 2. Run duration_steps
        for _ in range(duration_steps):
            traci.simulationStep()
            
        final_congestion = {}
        increased_congestion_edges = []
        
        for e in all_edges:
            if not e.startswith(":"):
                occ = traci.edge.getLastStepOccupancy(e)
                if occ > initial_congestion.get(e, 0):
                    increased_congestion_edges.append(e)
        
        # Restore speed
        for lane_id in lane_ids:
            traci.lane.setMaxSpeed(lane_id, original_speeds[lane_id])
        
        logging.info(f"Blockage simulation complete. {len(increased_congestion_edges)} edges saw increased congestion.")
        return increased_congestion_edges

    except Exception as e:
        logging.error(f"Blockage simulation failed: {e}")
        return []

if __name__ == "__main__":
    # Check if we want to run the standard simulation or test the blockage routine.
    # Default behavior as per step 3: Run simulation for 3600 steps.
    
    # We can perform a quick test of blockage at the end or just run the main sim.
    # The requirement says "Implement a callable routine".
    run_simulation()
