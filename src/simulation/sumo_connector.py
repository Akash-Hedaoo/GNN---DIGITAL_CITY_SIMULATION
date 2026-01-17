import os
import sys
import traci
import sumolib

class SumoManager:
    def __init__(self, net_file, trips_file, gui=False):
        self.net_file = net_file
        self.trips_file = trips_file
        self.gui = gui
        self.blocked_edges = set()
        self.sumo_cmd = []

    def start(self):
        """Start SUMO simulation."""
        binary = "sumo-gui" if self.gui else "sumo"
        # Check if binary exists, else fallback to sumo
        try:
             sumolib.checkBinary(binary)
        except:
             if self.gui:
                 print("sumo-gui not found, falling back to sumo")
                 binary = "sumo"
        
        sumo_bin = sumolib.checkBinary(binary)
        
        self.sumo_cmd = [
            sumo_bin,
            "-n", self.net_file,
            "-r", self.trips_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",
            "--ignore-route-errors", "true" # Prevent exit if generated trip is invalid
            # Mission says "physics engine... log high fidelity".
            # Teleporting removes vehicles from edges, which data collection might miss?
            # Standard is usually 300.
        ]
        
        traci.start(self.sumo_cmd)

    def step(self):
        """Advance one step."""
        traci.simulationStep()

    def close(self):
        """Close Traci."""
        traci.close()

    def toggle_blockage(self, edge_id):
        """Toggle blockage on an edge (limit speed to 0.1 m/s)."""
        if edge_id in self.blocked_edges:
            # Unblock
            traci.edge.setMaxSpeed(edge_id, -1) # Restore default
            self.blocked_edges.remove(edge_id)
            return False # Unblocked
        else:
            # Block
            # 0.1 m/s is effectively a crawl/block without crashing stopping logic completely (stops lane changing sometimes)
            traci.edge.setMaxSpeed(edge_id, 0.1)
            self.blocked_edges.add(edge_id)
            return True # Blocked

    def get_live_state(self):
        """
        Get state of all edges.
        Returns dict: {edge_id: color_hex}
        """
        edge_colors = {}
        # Get all edge IDs - cache this if performance is slow, but for 10k edges it might be ok per step?
        # Better to iterate known edges.
        # But simulation might have many edges.
        # traci.edge.getIDList() is fast.
        
        for edge_id in traci.edge.getIDList():
            if edge_id in self.blocked_edges:
                edge_colors[edge_id] = "#000000" # Black
                continue
            
            # Occupancy: % of edge occupied by vehicles
            # traci.edge.getLastStepOccupancy(edge_id) returns [0, 1]
            occ = traci.edge.getLastStepOccupancy(edge_id)
            
            if occ > 0.8:
                color = "#FF0000" # Red
            elif occ > 0.5:
                color = "#FFFF00" # Yellow
            elif occ < 0.2:
                color = "#00FF00" # Green
            else:
                color = "#00FF00" # Default Green-ish for 0.2-0.5? Logic said:
                # Green < 0.2, Yellow < 0.5, Red > 0.8. 
                # Gap 0.5 to 0.8? Assume Orange or Yellow extend.
                # Let's say Yellow is 0.2 to 0.8 to cover the gap.
                color = "#FFFF00" 
            
            edge_colors[edge_id] = color
            
        return edge_colors

    def get_edge_data(self, edge_id):
        """Get specific data for CSV logging."""
        return {
            "occupancy": traci.edge.getLastStepOccupancy(edge_id),
            "mean_speed": traci.edge.getLastStepMeanSpeed(edge_id),
            "vehicle_count": traci.edge.getLastStepVehicleNumber(edge_id)
        }
