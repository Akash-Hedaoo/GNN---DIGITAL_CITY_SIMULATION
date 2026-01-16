import os
import sys
import traci
import traci.constants as tc
import sumolib
import logging

class SumoManager:
    def __init__(self, net_file=None, trips_file=None):
        self.net_file = net_file or os.path.join("outputs", "pune.net.xml")
        self.trips_file = trips_file or os.path.join("outputs", "trips.trips.xml")
        self.sumo_cmd = ["sumo", "-n", self.net_file, "-r", self.trips_file, "--no-step-log", "true", "--no-warnings", "true"]
        self.running = False

    def start(self, gui=False):
        if gui:
            self.sumo_cmd[0] = "sumo-gui"
        
        logging.info(f"Starting SUMO: {self.sumo_cmd}")
        try:
            traci.start(self.sumo_cmd, label="sumo_backend")
            self.running = True
            logging.info("SUMO started successfully.")
        except Exception as e:
            logging.error(f"Failed to start SUMO: {e}")
            self.running = False
            raise e

    def step(self):
        if self.running:
            traci.simulationStep()
            return True
        return False

    def get_live_state(self):
        """
        Returns dictionary {edge_id: color_hex} based on occupancy.
        """
        state = {}
        if not self.running:
            return state

        edge_ids = traci.edge.getIDList()
        for edge_id in edge_ids:
            if edge_id.startswith(":"):
                continue
            
            try:
                occ = traci.edge.getLastStepOccupancy(edge_id)
                color = self.color_mapper(occ)
                state[edge_id] = color
            except Exception:
                pass
        
        return state

    def block_edge(self, edge_id):
        """
        Sets max speed of all lanes in the edge to 0.1 m/s.
        """
        if not self.running:
            return False
            
        logging.info(f"Blocking edge: {edge_id}")
        try:
            num_lanes = traci.edge.getLaneNumber(edge_id)
            for i in range(num_lanes):
                lane_id = f"{edge_id}_{i}"
                traci.lane.setMaxSpeed(lane_id, 0.1)
            return True
        except Exception as e:
            logging.error(f"Failed to block edge {edge_id}: {e}")
            return False

    def color_mapper(self, occ):
        if occ < 0.2:
            return "#00FF00" # Green
        elif occ < 0.5:
            return "#FFFF00" # Yellow
        elif occ < 0.8:
            return "#FFA500" # Orange
        else:
            return "#FF0000" # Red

    def close(self):
        if self.running:
            traci.close()
            self.running = False
            logging.info("SUMO closed.")

if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)
    mgr = SumoManager()
    try:
        mgr.start()
        for _ in range(10):
            mgr.step()
        logging.info("Ran 10 steps successfully.")
        
        state = mgr.get_live_state()
        logging.info(f"Retrieved state for {len(state)} edges.")
        
        mgr.close()
    except Exception as e:
        logging.error(f"Test failed: {e}")
