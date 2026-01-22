import sys
import logging
import simpy
import networkx as nx
import random
import pandas as pd
import matplotlib.pyplot as plt
import string
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from stable_baselines3 import PPO
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np

# ----------------------------
# GLOBAL METRICS
# ----------------------------
metrics_summary = {
    'RREQ_sent': 0,
    'RREP_sent': 0,
    'RERR_sent': 0,
    'Forwarding_sent': 0,
    'Alarms_sent': 0,
    'Routes_established': 0,
    'Nodes_deactivated': 0,
    'ACK_sent': 0,
    'ACK_received': 0,
    'DoS_packets_sent': 0
}

# ----------------------------
# LOGGING CONFIG
# ----------------------------
logger = logging.getLogger("SimulationLogger")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("data/simulation_output.txt", mode="w")
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ----------------------------
# CONSTANTS
# ----------------------------
ENERGY_CONSUMPTION = 0.2  # base energy consumption factor
MAX_SEQUENCE_NO = 100
ALARM_THRESHOLD = 5
PACKET_LOSS_THRESHOLD = 5  
INITIAL_ENERGY = 100

# Back-off constants for dynamic route discovery
BACK_OFF_TIME_BASE = 3    # base back-off time in time units
BACK_OFF_TIME_MAX = 10    # maximum allowed back-off

# Route expiration (time units)
ROUTE_EXPIRY_TIME = 30

# Routing table: Increase cached routes from 3 to 5 per destination.
MAX_ROUTES = 5            

# Packet type for block messages.
BLOCK = 'block'

# New packet types for ACK and DoS
ACK = 'ACK'
DOS = 'DOS'

# Packet types
FORWARDING = 'forwarding'
ALARM = 'alarm'
RREQ = 'RREQ'
RREP = 'RREP'
RERR = 'RERR'
SFA_ROUTE_REQUEST = 'SFA_RREQ'
SFA_ROUTE_REPLY = 'SFA_RREP'
SFA_ROUTE_ERROR = 'SFA_RERR'

# Protocol names
DSR = 'DSR'
SFA = 'SFA'
BASE_STATION_ID = 'BS'

DEFAULT_TTL = 10

# Sleep management: if no activity for this many time units, the node goes to sleep.
SLEEP_THRESHOLD = 10

# New constants for energy management
MIN_OPERATING_ENERGY = 0.1  # if a node's energy falls below this value, it is considered dead
MAX_ENERGY_COST = 1.0       # cap: no single packet transmission will cost more than this amount

# New: MINIMUM SEND INTERVAL TO LIMIT BURST ENERGY CONSUMPTION
MIN_SEND_INTERVAL = 0.5   # time units between packet transmissions

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def compute_distance(node1, node2):
    (x1, y1) = node1.position
    (x2, y2) = node2.position
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def compute_energy_cost(distance, base_distance=50):
    min_cost = 0.05
    cost = ENERGY_CONSUMPTION * (distance / base_distance)
    # Cap the maximum cost.
    cost = min(cost, MAX_ENERGY_COST)
    return max(min_cost, cost)

def consume_energy(node, amount=ENERGY_CONSUMPTION):
    if not node.active:
        return

    # If the node does not have enough energy to cover the consumption, log and deactivate it.
    if node.energy < amount:
        logger.info(f"Time {node.env.now}: Node {node.id} has insufficient energy ({node.energy:.2f}J) "
                    f"to consume {amount:.2f}J. Deactivating node.")
        node.energy = 0
        node.active = False
        metrics_summary['Nodes_deactivated'] += 1
        node._clear_queue()
    else:
        node.energy -= amount
        logger.info(f"Time {node.env.now}: Node {node.id} consumes {amount:.2f}J. Energy left: {node.energy:.2f}")

# ----------------------------
# PACKET CLASS
# ----------------------------
class Packet:
    def __init__(
        self, packet_type, src, dest, seq_no=None, payload=None,
        path=None, previous_node=None, ttl=DEFAULT_TTL, final_dest=None
    ):
        self.type = packet_type
        self.src = src            
        self.dest = dest          
        self.seq_no = seq_no
        self.payload = payload
        self.path = path or []
        self.previous_node = previous_node
        self.ttl = ttl
        self.final_dest = final_dest

# ----------------------------
# DSR PROTOCOL
# ----------------------------
class DSRProtocol:
    def __init__(self, node):
        self.node = node

    def send_rreq(self, dest):
        self.node.sequence_no += 1
        metrics_summary['RREQ_sent'] += 1
        neighbors = self.node.network.get_neighbors(self.node.id)
        for neighbor in neighbors:
            packet = Packet(
                RREQ, self.node.id, neighbor,
                seq_no=self.node.sequence_no,
                path=[self.node.id],
                previous_node=self.node.id,
                ttl=DEFAULT_TTL,
                final_dest=dest
            )
            logger.debug(f"Node {self.node.id} (DSR): Broadcasting RREQ to {neighbor} for {dest}")
            self.node.send_packet(packet)

    def send_forwarding_packet(self, final_dest, payload):
        self.node.sequence_no += 1
        metrics_summary['Forwarding_sent'] += 1
        route_entry = self.node.get_route(final_dest)
        if route_entry:
            next_hop = route_entry[0]
            packet = Packet(
                FORWARDING, self.node.id, next_hop, seq_no=self.node.sequence_no,
                payload=payload, final_dest=final_dest, previous_node=self.node.id
            )
            logger.debug(f"Node {self.node.id} (DSR): Forwarding data to {final_dest} via {next_hop}")
            self.node.send_packet(packet)
        else:
            logger.warning(f"Node {self.node.id} (DSR): No route to {final_dest}. Initiating route discovery.")
            self.send_rreq(final_dest)

    def send_rerr(self, dest):
        metrics_summary['RERR_sent'] += 1
        packet = Packet(RERR, self.node.id, dest)
        logger.debug(f"Node {self.node.id} (DSR): Sending RERR to {dest}")
        self.node.send_packet(packet)

# ----------------------------
# SFA PROTOCOL
# ----------------------------
class SFAProtocol:
    def __init__(self, node):
        self.node = node

    def send_rreq(self, dest):
        self.node.sequence_no += 1
        metrics_summary['RREQ_sent'] += 1
        packet = Packet(
            SFA_ROUTE_REQUEST, self.node.id, dest, seq_no=self.node.sequence_no, path=[self.node.id]
        )
        logger.debug(f"Node {self.node.id} (SFA): Sending RREQ for {dest}")
        self.node.send_packet(packet)

    def send_forwarding_packet(self, dest, payload):
        self.node.sequence_no += 1
        metrics_summary['Forwarding_sent'] += 1
        packet = Packet(FORWARDING, self.node.id, dest, seq_no=self.node.sequence_no, payload=payload, previous_node=self.node.id)
        logger.debug(f"Node {self.node.id} (SFA): Forwarding data to {dest}")
        self.node.send_packet(packet)

    def send_rerr(self, dest):
        metrics_summary['RERR_sent'] += 1
        packet = Packet(SFA_ROUTE_ERROR, self.node.id, dest)
        logger.debug(f"Node {self.node.id} (SFA): Sending RERR to {dest}")
        self.node.send_packet(packet)

# ----------------------------
# RL ENVIRONMENT
# ----------------------------
class RoutingEnv(Env):
    def __init__(self, network, node, dest):
        super(RoutingEnv, self).__init__()
        self.network = network
        self.node = node
        self.dest = dest
        self.neighbors = self.network.get_neighbors(self.node.id)
        self.action_space = Discrete(len(self.neighbors))
        self.observation_space = Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        self.state = np.array([self.node.energy, self.node.sequence_no, self._route_count()], dtype=np.float32)

    def _route_count(self):
        count = 0
        for route_options in self.node.routing_table:
            for entry in route_options:
                if entry is not None:
                    count += 1
        return count

    def reset(self):
        self.state = np.array([self.node.energy, self.node.sequence_no, self._route_count()], dtype=np.float32)
        return self.state

    def step(self, action):
        if action >= len(self.neighbors):
            action = len(self.neighbors) - 1
        self.node.protocol_handler.send_forwarding_packet(self.dest, payload="RL Data Packet")
        reward = 1 - ENERGY_CONSUMPTION
        done = False
        self.state = np.array([self.node.energy, self.node.sequence_no, self._route_count()], dtype=np.float32)
        return self.state, reward, done, {}

    def render(self, mode="human"):
        pass

# ----------------------------
# NODE CLASS WITH ADVANCED FEATURES
# ----------------------------
class Node:
    def __init__(self, env, node_id, network, protocol=DSR, is_base_station=False,
                 dl_model=None, scaler=None, rl_model=None):
        self.env = env
        self.id = node_id
        self.network = network
        self.protocol = protocol
        self.is_base_station = is_base_station
        self.energy = INITIAL_ENERGY
        self.active = True
        self.state = 'active'
        self.last_active_time = self.env.now
        # Random position for adaptive transmission power (x, y) in a 100x100 area.
        self.position = (random.uniform(0, 100), random.uniform(0, 100))
        self.routing_table = {}  # will be set up by Network
        self.sequence_no = 0
        self.queue = simpy.Store(env)

        # For DSR duplicate RREQ tracking
        if protocol == DSR:
            self.seen_rreq = {}

        self.packets_sent = 0
        self.packets_received = 0
        self.process = env.process(self.run())
        self.env.process(self.manage_sleep_cycle())
        # For rate-limiting route discoveries (stores last discovery time per destination)
        self.last_rreq_time = {}

        # NEW: Initialize last send time to enforce a minimum send interval
        self.last_send_time = -MIN_SEND_INTERVAL

        if not self.is_base_station:
            if self.protocol == DSR:
                self.protocol_handler = DSRProtocol(self)
            elif self.protocol == SFA:
                self.protocol_handler = SFAProtocol(self)

        self.dl_model = dl_model
        self.scaler = scaler
        self.rl_model = rl_model

        # Start periodic sensor data transmission (WSN behavior)
        if not self.is_base_station:
            self.env.process(self.sense_and_transmit())

    def _clear_queue(self):
        while self.queue.items:
            try:
                self.queue.items.pop(0)
            except Exception as e:
                logger.error(f"Error clearing queue for {self.id}: {e}")

    def run(self):
        # Modify the run loop to terminate once the node is inactive
        while self.active:
            packet = yield self.queue.get()
            # Check if the node has been deactivated meanwhile; if so, exit the loop.
            if not self.active:
                logger.info(f"Time {self.env.now}: Node {self.id} is inactive; terminating packet processing loop.")
                break
            # Wake up if sleeping.
            if self.state == "sleep":
                self.state = "active"
                logger.info(f"Time {self.env.now}: Node {self.id} is waking up to process a packet.")
            self.last_active_time = self.env.now
            # Special handling for DOS and ACK packets
            if packet.type == DOS:
                logger.warning(f"Time {self.env.now}: {self.id} received DOS packet from {packet.src}.")
                consume_energy(self, ENERGY_CONSUMPTION * 0.5)  # DOS packets drain extra energy
                continue
            if packet.type == ACK:
                self.handle_ack(packet)
                continue

            if self.is_base_station:
                self.handle_alarm(packet)
            else:
                self.receive_packet(packet)
            # Add a small processing delay to smooth energy consumption
            yield self.env.timeout(0.1)

    def manage_sleep_cycle(self):
        while self.active:
            yield self.env.timeout(5)
            if (self.env.now - self.last_active_time) > SLEEP_THRESHOLD and self.state != "sleep":
                self.state = "sleep"
                logger.info(f"Time {self.env.now}: Node {self.id} is entering sleep mode due to inactivity.")

    def _send_packet(self, packet):
        if not self.active:
            return
        with self.network.send_lock.request() as req:
            yield req
            # Recheck active status after acquiring the lock.
            if not self.active:
                return
            # Enforce a minimum interval between sends to avoid burst transmissions
            time_since_last = self.env.now - self.last_send_time
            if time_since_last < MIN_SEND_INTERVAL:
                yield self.env.timeout(MIN_SEND_INTERVAL - time_since_last)
            self.last_send_time = self.env.now

            if self.state == "sleep":
                self.state = "active"
                logger.info(f"Time {self.env.now}: Node {self.id} woke up (after lock) to send a packet.")
            self.last_active_time = self.env.now

            # Determine energy cost based on distance (or use the base cost)
            dest_node = self.network.nodes_dict.get(packet.dest)
            if dest_node:
                distance = compute_distance(self, dest_node)
                cost = compute_energy_cost(distance)
            else:
                cost = ENERGY_CONSUMPTION

            # Check if the node has sufficient energy to send the packet
            if self.energy < cost + MIN_OPERATING_ENERGY:
                logger.info(f"Time {self.env.now}: Node {self.id} has insufficient energy "
                            f"({self.energy:.2f}J) to send {packet.type} (requires at least {cost + MIN_OPERATING_ENERGY:.2f}J). Packet dropped.")
                return

            # Consume energy and abort sending if deactivated during consumption.
            consume_energy(self, cost)
            if not self.active:
                return

        self.packets_sent += 1
        logger.info(f"Time {self.env.now}: {self.id} sends {packet.type} -> {packet.dest} (Prot: {self.protocol})")
        self.network.send_packet(packet)

    def send_packet(self, packet):
        if not self.active:
            logger.info(f"Time {self.env.now}: Node {self.id} is inactive; not sending packet {packet.type}.")
            return
        # Wake up if needed.
        if self.state == "sleep":
            self.state = "active"
            logger.info(f"Time {self.env.now}: Node {self.id} woke up to send a packet.")
        self.last_active_time = self.env.now
        self.env.process(self._send_packet(packet))

    def receive_packet(self, packet):
        if not self.active:
            logger.info(f"Time {self.env.now}: Node {self.id} is inactive; not processing incoming {packet.type} packet.")
            return
        # Receiving consumes a small fixed amount of energy.
        consume_energy(self, ENERGY_CONSUMPTION * 0.5)
        if not self.active:
            return
        self.last_active_time = self.env.now
        self.packets_received += 1
        logger.info(f"Time {self.env.now}: {self.id} received {packet.type} from {packet.src} (Prot: {self.protocol})")
        if packet.type == BLOCK:
            self.handle_block_message(packet)
            return

        if self.protocol == DSR:
            if packet.type == RREQ:
                self.handle_rreq(packet)
            elif packet.type == RREP:
                self.handle_rrep(packet)
            elif packet.type == FORWARDING:
                self.handle_forwarding(packet)
            elif packet.type == RERR:
                self.handle_rerr(packet)
            elif packet.type == ALARM:
                self.handle_alarm(packet)
        else:  # SFA
            if packet.type == SFA_ROUTE_REQUEST:
                self.handle_sfa_rreq(packet)
            elif packet.type == SFA_ROUTE_REPLY:
                self.handle_sfa_rrep(packet)
            elif packet.type == FORWARDING:
                self.handle_forwarding(packet)
            elif packet.type == SFA_ROUTE_ERROR:
                self.handle_sfa_rerr(packet)
            elif packet.type == ALARM:
                self.handle_alarm(packet)

    # -------------------
    # ACK METHODS
    # -------------------
    def send_ack(self, original_packet):
        ack_packet = Packet(ACK, self.id, original_packet.previous_node, seq_no=self.sequence_no, payload={"ack_for": original_packet.type})
        metrics_summary['ACK_sent'] += 1
        logger.info(f"Time {self.env.now}: {self.id} sending ACK to {original_packet.previous_node} for packet type {original_packet.type}")
        self.send_packet(ack_packet)

    def handle_ack(self, packet):
        logger.info(f"Time {self.env.now}: {self.id} received ACK from {packet.src} for packet {packet.payload.get('ack_for') if packet.payload else 'unknown'}")
        metrics_summary['ACK_received'] += 1

    # -------------------
    # WSN SENSOR DATA TRANSMISSION
    # -------------------
    def sense_and_transmit(self):
        while self.active:
            # Wait a random interval between sensor readings
            yield self.env.timeout(random.uniform(8, 12))
            if self.active:
                sensor_data = f"Sensor reading from {self.id} at time {self.env.now}"
                logger.info(f"Time {self.env.now}: {self.id} sensing data: {sensor_data}")
                # Use the forwarding mechanism to send sensor data to the Base Station
                if self.protocol == DSR:
                    self.protocol_handler.send_forwarding_packet(BASE_STATION_ID, sensor_data)
                else:
                    self.protocol_handler.send_forwarding_packet(BASE_STATION_ID, sensor_data)

    # -------------------
    # Routing Table Helpers with Expiration
    # -------------------
    def add_route(self, dest, next_hop, seq_no):
        dest_index = self.network.node_index[dest]
        current_time = self.env.now
        new_entry = [next_hop, seq_no, current_time]
        updated = False
        for i in range(MAX_ROUTES):
            if self.routing_table[dest_index][i] is None:
                self.routing_table[dest_index][i] = new_entry
                updated = True
                break
            elif seq_no > self.routing_table[dest_index][i][1]:
                self.routing_table[dest_index][i] = new_entry
                updated = True
                break
        if not updated:
            # Replace the entry with the smallest seq_no
            min_index = 0
            min_seq = self.routing_table[dest_index][0][1] if self.routing_table[dest_index][0] is not None else -1
            for i in range(1, MAX_ROUTES):
                if self.routing_table[dest_index][i] is not None and self.routing_table[dest_index][i][1] < min_seq:
                    min_seq = self.routing_table[dest_index][i][1]
                    min_index = i
            self.routing_table[dest_index][min_index] = new_entry

    def get_route(self, dest):
        dest_index = self.network.node_index[dest]
        best_route = None
        best_energy = -1
        current_time = self.env.now
        for route in self.routing_table[dest_index]:
            if route is not None:
                # Check expiration
                if (current_time - route[2]) > ROUTE_EXPIRY_TIME:
                    continue
                next_hop = route[0]
                neighbor = self.network.nodes_dict.get(next_hop)
                if neighbor is None or not neighbor.active:
                    continue
                # Prefer the route whose next hop has the highest residual energy.
                if neighbor.energy > best_energy:
                    best_energy = neighbor.energy
                    best_route = route
        return best_route

    def remove_route(self, dest):
        dest_index = self.network.node_index[dest]
        self.routing_table[dest_index] = [None] * MAX_ROUTES

    def prune_routing_table(self):
        """Periodically prune expired or inactive routes."""
        while self.active:
            yield self.env.timeout(5)
            current_time = self.env.now
            for dest_index, routes in enumerate(self.routing_table):
                for i, entry in enumerate(routes):
                    if entry is not None:
                        next_hop = entry[0]
                        neighbor = self.network.nodes_dict.get(next_hop)
                        if neighbor is None or not neighbor.active or (current_time - entry[2] > ROUTE_EXPIRY_TIME):
                            self.routing_table[dest_index][i] = None
                            logger.info(f"Node {self.id}: Pruned route for dest index {dest_index} via {next_hop}.")

    # -------------------
    # DSR Handlers
    # -------------------
    def handle_rreq(self, packet):
        key = (packet.src, packet.seq_no)
        if key in self.seen_rreq:
            if packet.ttl <= self.seen_rreq[key]:
                logger.debug(f"Node {self.id}: Duplicate RREQ {key} with TTL {packet.ttl} <= stored TTL, dropping.")
                return
            else:
                logger.debug(f"Node {self.id}: Duplicate RREQ {key} with higher TTL {packet.ttl}, processing.")
                self.seen_rreq[key] = packet.ttl
        else:
            self.seen_rreq[key] = packet.ttl

        if self.id == packet.src:
            return

        for idx, node_id in enumerate(packet.path):
            if idx > 0:
                self.add_route(node_id, packet.path[idx - 1], packet.seq_no)

        if self.id == packet.final_dest:
            route = packet.path.copy()
            if route[-1] != self.id:
                route.append(self.id)
            rev_route = list(reversed(route))
            if len(rev_route) < 2:
                logger.error(f"Node {self.id}: Route too short for RREP: {rev_route}")
                return
            next_hop = rev_route[1]
            rrep = Packet(RREP, self.id, next_hop, seq_no=packet.seq_no, path=rev_route)
            metrics_summary['RREP_sent'] += 1
            self.send_packet(rrep)
            metrics_summary['Routes_established'] += 1
            return

        new_path = packet.path + [self.id] if self.id not in packet.path else packet.path
        for neighbor in self.network.get_neighbors(self.id):
            if neighbor == packet.previous_node or neighbor in new_path:
                continue
            new_packet = Packet(RREQ, packet.src, neighbor, seq_no=packet.seq_no,
                                path=new_path, previous_node=self.id, ttl=packet.ttl,
                                final_dest=packet.final_dest)
            self.send_packet(new_packet)

    def handle_rrep(self, packet):
        if self.id == packet.path[-1]:
            dest = packet.src
            if len(packet.path) >= 2:
                self.add_route(dest, packet.path[-2], packet.seq_no)
            return
        try:
            idx = packet.path.index(self.id)
        except ValueError:
            logger.error(f"Node {self.id}: RREP route missing self: {packet.path}")
            return

        if idx < len(packet.path) - 1:
            next_hop = packet.path[idx + 1]
            packet.dest = next_hop
            self.add_route(packet.src, next_hop, packet.seq_no)
            self.send_packet(packet)
        else:
            logger.error(f"Node {self.id}: RREP route exhausted: {packet.path}")

    def handle_forwarding(self, packet):
        # For DSR: use final_dest; for SFA: use dest field.
        if self.protocol == DSR:
            if packet.final_dest == self.id:
                logger.info(f"Time {self.env.now}: Node {self.id}: Data arrived from {packet.src}")
                # Send ACK to the previous hop if available
                if packet.previous_node:
                    self.send_ack(packet)
                return
            route_entry = self.get_route(packet.final_dest)
            if route_entry:
                next_hop = route_entry[0]
                fwd_pkt = Packet(FORWARDING, self.id, next_hop, seq_no=packet.seq_no,
                                  payload=packet.payload, previous_node=self.id,
                                  ttl=packet.ttl, final_dest=packet.final_dest)
                self.send_packet(fwd_pkt)
            else:
                logger.warning(f"Node {self.id}: No DSR route to {packet.final_dest}. Initiating route discovery.")
                self.protocol_handler.send_rreq(packet.final_dest)
        else:
            if packet.dest == self.id:
                logger.info(f"Time {self.env.now}: Node {self.id}: Data arrived from {packet.src}")
                if packet.previous_node:
                    self.send_ack(packet)
                return
            route = self.get_route(packet.dest)
            if route:
                next_hop = route[0]
                fwd_pkt = Packet(FORWARDING, self.id, packet.dest, seq_no=packet.seq_no,
                                  payload=packet.payload, previous_node=self.id)
                self.send_packet(fwd_pkt)
            else:
                logger.warning(f"Node {self.id}: No SFA route to {packet.dest}. Initiating route discovery.")
                self.protocol_handler.send_rreq(packet.dest)

    def handle_rerr(self, packet):
        logger.warning(f"Node {self.id}: RERR from {packet.src} (DSR). Removing route to {packet.src}.")
        self.remove_route(packet.src)

    # -------------------
    # SFA Handlers
    # -------------------
    def handle_sfa_rreq(self, packet):
        if self.id not in packet.path:
            packet.path.append(self.id)
            if self.id == packet.dest:
                rrep = Packet(SFA_ROUTE_REPLY, self.id, packet.src, seq_no=self.sequence_no, path=packet.path.copy())
                metrics_summary['RREP_sent'] += 1
                self.send_packet(rrep)
                metrics_summary['Routes_established'] += 1
            else:
                for neighbor in self.network.get_neighbors(self.id):
                    if len(packet.path) > 1 and neighbor == packet.path[-2]:
                        continue
                    forward_rreq = Packet(SFA_ROUTE_REQUEST, self.id, packet.dest, seq_no=packet.seq_no,
                                           path=packet.path.copy(), previous_node=self.id)
                    self.send_packet(forward_rreq)

    def handle_sfa_rrep(self, packet):
        existing_route = self.get_route(packet.src)
        if existing_route is None or existing_route[1] < packet.seq_no:
            if len(packet.path) > 1:
                self.add_route(packet.src, packet.path[-2], packet.seq_no)
        if self.id != packet.dest:
            route = self.get_route(packet.dest)
            if route:
                next_hop = route[0]
                packet.path.append(self.id)
                forward_rrep = Packet(SFA_ROUTE_REPLY, self.id, packet.dest,
                                      seq_no=packet.seq_no, path=packet.path.copy(),
                                      previous_node=self.id)
                self.send_packet(forward_rrep)
            else:
                rerr = Packet(SFA_ROUTE_ERROR, self.id, packet.src)
                self.send_packet(rerr)
        else:
            logger.info(f"Node {self.id}: SFA route established successfully.")

    def handle_sfa_rerr(self, packet):
        logger.warning(f"Node {self.id}: SFA_RERR from {packet.src}, removing route.")
        self.remove_route(packet.src)

    # -------------------
    # ALARM and BLOCK HANDLING
    # -------------------
    def handle_alarm(self, packet):
        if packet.type == ALARM:
            if packet.payload and 'src' in packet.payload:
                src_node_id = packet.payload['src']
                packet_loss = packet.payload.get("packet_loss", 0)
                logger.warning(f"Time {self.env.now}: {self.id} got ALARM about {src_node_id} (packet_loss: {packet_loss}).")
                if self.is_base_station:
                    last_seq = getattr(self, 'last_alarm_seq', {})
                    stored_seq = last_seq.get(packet.src, 0)
                    if packet.seq_no is None or packet.seq_no <= stored_seq:
                        logger.warning("Stale or invalid alarm sequence number. Dropping alarm.")
                        return
                    last_seq[packet.src] = packet.seq_no
                    self.last_alarm_seq = last_seq
                    suspected = self.network.nodes_dict.get(src_node_id)
                    if suspected:
                        expected_energy = INITIAL_ENERGY - (suspected.packets_sent + suspected.packets_received) * ENERGY_CONSUMPTION
                        if suspected.energy > expected_energy + ALARM_THRESHOLD or packet_loss > PACKET_LOSS_THRESHOLD:
                            logger.warning(f"BS detects malicious node {src_node_id}: energy discrepancy (actual: {suspected.energy:.2f}, expected: {expected_energy:.2f}) or high packet loss ({packet_loss}). Broadcasting block message.")
                            self.broadcast_block_message(src_node_id)
                        else:
                            logger.info(f"BS: Alarm from {src_node_id} does not indicate malicious behavior.")
                    else:
                        logger.warning(f"BS: Unknown node {src_node_id} in alarm payload.")
                else:
                    logger.info(f"Node {self.id} (non-BS) received ALARM about {src_node_id}.")
            else:
                logger.warning(f"Time {self.env.now}: {self.id} got ALARM with no 'src' info.")

    def broadcast_block_message(self, malicious_node_id):
        for node in self.network.nodes:
            if node.id != malicious_node_id and node.active:
                block_packet = Packet(BLOCK, src=self.id, dest=node.id, seq_no=self.sequence_no, payload={"block": malicious_node_id})
                logger.info(f"BS broadcasting BLOCK for {malicious_node_id} to {node.id}")
                self.send_packet(block_packet)

    def handle_block_message(self, packet):
        blocked_node = packet.payload.get("block")
        if blocked_node:
            logger.warning(f"Node {self.id}: Received BLOCK message for {blocked_node}. Removing any routes to it.")
            self.remove_route(blocked_node)
        else:
            logger.warning(f"Node {self.id}: Received BLOCK message with no 'block' info.")

    # -------------------
    # ML HOOKS
    # -------------------
    def predict_traffic(self):
        if self.dl_model and self.scaler:
            protocol_code = 0 if self.protocol == DSR else 1
            X = np.array([[self.env.now, protocol_code]])
            X_scaled = self.scaler.transform(X)
            pred = self.dl_model.predict(X_scaled)
            return pred[0][0]
        return None

    def make_routing_decision(self, dest):
        predicted = self.predict_traffic()
        if predicted and predicted > 50:
            logger.info(f"Node {self.id}: High predicted traffic. Considering routing adjustments.")
        self.initiate_route_discovery(dest)

    def make_routing_decision_rl(self, dest):
        if self.rl_model:
            rl_env = RoutingEnv(self.network, self, dest)
            obs = rl_env.reset()
            action, _ = self.rl_model.predict(obs)
            neighbors = self.network.get_neighbors(self.id)
            if action < len(neighbors):
                self.protocol_handler.send_forwarding_packet(dest, payload="RL Data Packet")
        else:
            self.initiate_route_discovery(dest)

    def initiate_route_discovery(self, dest):
        current_time = self.env.now
        energy_factor = INITIAL_ENERGY / (self.energy if self.energy > 0 else 1)
        dynamic_backoff = min(BACK_OFF_TIME_BASE * energy_factor, BACK_OFF_TIME_MAX)
        if dest in self.last_rreq_time and (current_time - self.last_rreq_time[dest]) < dynamic_backoff:
            logger.info(f"Node {self.id}: Backing off route discovery for {dest} (back-off = {dynamic_backoff}).")
            return
        self.last_rreq_time[dest] = current_time
        if self.protocol == DSR:
            self.protocol_handler.send_rreq(dest)
        else:
            self.protocol_handler.send_rreq(dest)

# ----------------------------
# MALICIOUS NODE (DoS)
# ----------------------------
class MaliciousNode(Node):
    def __init__(self, env, node_id, network, protocol=DSR):
        super().__init__(env, node_id, network, protocol=protocol)
        self.malicious = True
        self.env.process(self.launch_dos_attack())
    
    def launch_dos_attack(self):
        while self.active:
            # Flood the network with DOS packets to all neighbors
            neighbors = self.network.get_neighbors(self.id)
            for neighbor in neighbors:
                self.sequence_no += 1
                dos_packet = Packet(DOS, self.id, neighbor, seq_no=self.sequence_no, payload={"dos": True})
                metrics_summary['DoS_packets_sent'] += 1
                logger.warning(f"Time {self.env.now}: Malicious node {self.id} flooding DOS packet to {neighbor}")
                self.send_packet(dos_packet)
            yield self.env.timeout(1)

# ----------------------------
# BASE STATION
# ----------------------------
class BaseStation(Node):
    def __init__(self, env, node_id, network):
        super().__init__(env, node_id, network, is_base_station=True)
        self.last_alarm_seq = {}

# ----------------------------
# NETWORK CLASS
# ----------------------------
class Network:
    def __init__(self, env, edge_list=None, protocol=DSR):
        self.env = env
        self.graph = nx.Graph()
        self.nodes = []
        self.nodes_dict = {}
        self.protocol = protocol
        self.send_lock = simpy.Resource(env, capacity=1)
        self.create_nodes()
        if edge_list:
            self.create_edges(edge_list)
        else:
            self.create_random_edges()
        self.node_index = {}
        node_ids = [node.id for node in self.nodes]
        for idx, node_id in enumerate(node_ids):
            self.node_index[node_id] = idx
        num_nodes = len(self.nodes)
        for node in self.nodes:
            node.routing_table = [[None for _ in range(MAX_ROUTES)] for _ in range(num_nodes)]
            node.index = self.node_index[node.id]

    def create_nodes(self):
        node_names = list(string.ascii_uppercase)[:20]
        for name in node_names:
            node_id = f'Node {name}'
            node = Node(self.env, node_id, self, protocol=self.protocol)
            self.nodes.append(node)
            self.nodes_dict[node_id] = node
            self.graph.add_node(node_id)
        bs = BaseStation(self.env, BASE_STATION_ID, self)
        self.nodes.append(bs)
        self.nodes_dict[BASE_STATION_ID] = bs
        self.graph.add_node(BASE_STATION_ID)
        # Add a malicious node for DoS attacks
        mal_node = MaliciousNode(self.env, "Malicious Node", self, protocol=self.protocol)
        self.nodes.append(mal_node)
        self.nodes_dict["Malicious Node"] = mal_node
        self.graph.add_node("Malicious Node")

    def create_edges(self, edge_list):
        for edge in edge_list:
            if edge[0] in self.nodes_dict and edge[1] in self.nodes_dict:
                self.graph.add_edge(*edge)
            else:
                logger.error(f"Invalid edge: {edge} - node doesn't exist.")
        logger.info(f"Network edges: {list(self.graph.edges())}")

    def create_random_edges(self):
        all_nodes = [n.id for n in self.nodes]
        for node in self.nodes:
            if node.id == BASE_STATION_ID:
                num_neighbors = min(3, len(self.nodes) - 1)
                neighbors = random.sample([x for x in all_nodes if x != BASE_STATION_ID], num_neighbors)
            else:
                num_neighbors = random.randint(1, min(4, len(self.nodes) - 1))
                neighbors = random.sample([x for x in all_nodes if x != node.id], num_neighbors)
            for neighbor in neighbors:
                self.graph.add_edge(node.id, neighbor)
        logger.info(f"Random edges: {list(self.graph.edges())}")

    def send_packet(self, packet):
        delay = random.uniform(0.5, 2)
        self.env.process(self.deliver_packet(packet, delay))

    def deliver_packet(self, packet, delay):
        try:
            yield self.env.timeout(delay)
            dest_node = self.nodes_dict.get(packet.dest)
            if dest_node and dest_node.active:
                yield dest_node.queue.put(packet)
            else:
                logger.warning(f"Time {self.env.now}: Destination {packet.dest} inactive or not found.")
        except Exception as e:
            logger.error(f"Error in deliver_packet: {e}")

    def get_neighbors(self, node_id):
        return list(self.graph.neighbors(node_id))

    def get_all_nodes(self):
        return self.nodes

    def remove_node_routes(self, node_id):
        for node in self.nodes:
            node.remove_route(node_id)
            logger.info(f"Node {node.id} removed route to {node_id}")

# ----------------------------
# SETUP SIMULATION
# ----------------------------
def setup_simulation(env, edge_list, protocol=DSR, dl_model=None, scaler=None, rl_model=None):
    net = Network(env, edge_list=edge_list, protocol=protocol)
    for n in net.nodes:
        n.dl_model = dl_model
        n.scaler = scaler
        n.rl_model = rl_model
    return net

# ----------------------------
# DATA COLLECTION
# ----------------------------
def collect_data(env, networks, data_store):
    while True:
        yield env.timeout(5)
        data = {
            'Time': env.now,
            'Protocol': [],
            'Node': [],
            'Energy': [],
            'Active': [],
            'Packets_Sent': [],
            'Packets_Received': [],
            'Alarms_Sent': []
        }
        for proto, net in networks.items():
            for node in net.nodes:
                data['Protocol'].append(proto)
                data['Node'].append(node.id)
                data['Energy'].append(node.energy)
                data['Active'].append(node.active)
                data['Packets_Sent'].append(node.packets_sent)
                data['Packets_Received'].append(node.packets_received)
                data['Alarms_Sent'].append(metrics_summary['Alarms_sent'])
        df = pd.DataFrame(data)
        data_store.append(df)

# ----------------------------
# ML MODEL TRAINING
# ----------------------------
def train_traffic_prediction_model(data_store):
    if not data_store:
        logger.warning("No data to train on.")
        return None, None
    full_data = pd.concat(data_store, ignore_index=True)
    full_data['Protocol'] = full_data['Protocol'].astype('category').cat.codes
    features = full_data[['Time', 'Protocol']]
    targets = full_data['Packets_Sent']
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, targets, test_size=0.2, random_state=42
    )
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Trained traffic model. Test MAE = {mae}")
    return model, scaler

# ----------------------------
# RL TRAINING
# ----------------------------
def train_rl_agent(network, node, dest, timesteps=5000):
    rl_env = RoutingEnv(network, node, dest)
    model = PPO('MlpPolicy', rl_env, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model

# ----------------------------
# PROCESSES FOR DATA & ROUTE OPERATIONS
# ----------------------------
def initiate_routes_dsr(env, network):
    while True:
        yield env.timeout(random.randint(1, 5))
        # yield env.timeout(3)  # Fixed timeout for DSR
        active_nodes = [n for n in network.nodes if not n.is_base_station and n.active]
        if not active_nodes:
            continue
        sender = random.choice(active_nodes)
        possible = [n for n in network.nodes if n.id != sender.id and n.active]
        if not possible:
            continue
        receiver = random.choice(possible)
        logger.info(f"Time {env.now}: {sender.id} -> RREQ to {receiver.id} (DSR)")
        sender.initiate_route_discovery(receiver.id)

def initiate_routes_sfa(env, network):
    while True:
        yield env.timeout(random.randint(1, 5))
        # yield env.timeout(3)  # Same fixed timeout for SFA
        active_nodes = [n for n in network.nodes if not n.is_base_station and n.active]
        if not active_nodes:
            continue
        sender = random.choice(active_nodes)
        possible = [n for n in network.nodes if n.id != sender.id and n.active]
        if not possible:
            continue
        receiver = random.choice(possible)
        logger.info(f"Time {env.now}: {sender.id} -> RREQ to {receiver.id} (SFA)")
        sender.initiate_route_discovery(receiver.id)

def send_forwarding_packets_dsr(env, network):
    while True:
        yield env.timeout(random.randint(2, 6))
        senders = [n for n in network.nodes if not n.is_base_station and n.active and any(any(x is not None for x in r) for r in n.routing_table)]
        if not senders:
            continue
        sender = random.choice(senders)
        possible_destinations = []
        for dest_index, routes in enumerate(sender.routing_table):
            for entry in routes:
                if entry is not None:
                    for node_id, idx in sender.network.node_index.items():
                        if idx == dest_index:
                            possible_destinations.append(node_id)
                            break
        if not possible_destinations:
            continue
        dest = random.choice(possible_destinations)
        payload = f"Data from {sender.id} to {dest} (DSR)"
        logger.info(f"Time {env.now}: {sender.id} -> Data to {dest} (DSR)")
        sender.protocol_handler.send_forwarding_packet(dest, payload)

def send_forwarding_packets_sfa(env, network):
    while True:
        yield env.timeout(random.randint(2, 6))
        senders = [n for n in network.nodes if not n.is_base_station and n.active]
        if not senders:
            continue
        sender = random.choice(senders)
        possible_dests = [n.id for n in network.nodes if n.id != sender.id and n.active]
        if not possible_dests:
            continue
        dest = random.choice(possible_dests)
        payload = f"Data from {sender.id} to {dest} (SFA)"
        logger.info(f"Time {env.now}: {sender.id} -> Data to {dest} (SFA)")
        sender.protocol_handler.send_forwarding_packet(dest, payload)

def initiate_routes_rl(env, network, node, dest):
    while True:
        yield env.timeout(random.randint(1, 5))
        if node.active:
            node.make_routing_decision_rl(dest)

# ----------------------------
# NEW: ALARM TRIGGER PROCESS BASED ON PACKET LOSS
# ----------------------------
def send_random_alarms(env, network):
    while True:
        yield env.timeout(4)
        # Only consider nodes that have high packet loss
        nodes_exceeding = [n for n in network.nodes if n.active and not n.is_base_station and (n.packets_sent - n.packets_received) > PACKET_LOSS_THRESHOLD]
        if not nodes_exceeding:
            continue
        for suspected in nodes_exceeding:
            bs_node = network.nodes_dict.get(BASE_STATION_ID)
            if bs_node and bs_node.active:
                suspected.sequence_no += 1
                alarm_packet = Packet(
                    ALARM,
                    src=suspected.id,
                    dest=bs_node.id,
                    seq_no=suspected.sequence_no,
                    payload={
                        "src": suspected.id,
                        "packet_loss": suspected.packets_sent - suspected.packets_received
                    }
                )
                metrics_summary['Alarms_sent'] += 1
                logger.warning(f"Time {env.now}: Generating ALARM about {suspected.id}, sending to BS.")
                suspected.send_packet(alarm_packet)
            else:
                logger.warning("No active Base Station available for alarming.")

# ----------------------------
# MAIN SIMULATION
# ----------------------------
if __name__ == "__main__":
    # Edges for DSR
    edge_list_dsr = [
        ('Node A', 'Node P'), ('Node A', 'Node N'), ('Node B', 'Node F'), ('Node B', 'Node H'),
        ('Node B', 'Node D'), ('Node B', 'Node Q'), ('Node C', 'Node P'), ('Node C', 'Node I'),
        ('Node C', 'Node M'), ('Node C', 'Node R'), ('Node D', 'Node T'), ('Node D', 'Node H'),
        ('Node D', 'Node N'), ('Node D', 'Node R'), ('Node D', 'Node J'), ('Node D', 'Node M'),
        ('Node E', 'Node J'), ('Node E', 'Node F'), ('Node E', 'Node R'), ('Node F', 'Node S'),
        ('Node F', 'Node T'), ('Node G', 'Node L'), ('Node G', 'Node M'), ('Node G', 'Node R'),
        ('Node G', 'Node P'), ('Node G', 'BS'), ('Node H', 'Node Q'), ('Node H', 'Node P'),
        ('Node H', 'Node J'), ('Node H', 'Node L'), ('Node I', 'Node P'), ('Node I', 'Node M'),
        ('Node J', 'Node P'), ('Node K', 'Node L'), ('Node K', 'Node Q'), ('Node K', 'BS'),
        ('Node L', 'Node T'), ('Node L', 'Node Q'), ('Node L', 'Node O'), ('Node L', 'Node S'),
        ('Node M', 'Node O'), ('Node M', 'Node S'), ('Node M', 'BS'), ('Node N', 'Node Q'),
        ('Node O', 'Node R'), ('Node O', 'Node T'), ('Node P', 'Node T'), ('Node Q', 'Node R'),
        ('Node S', 'Node T')
    ]

    # Edges for SFA
    edge_list_sfa = [
        ('Node A', 'Node H'), ('Node A', 'Node I'), ('Node A', 'Node C'), ('Node A', 'Node P'),
        ('Node A', 'Node F'), ('Node A', 'Node L'), ('Node B', 'Node C'), ('Node B', 'Node P'),
        ('Node C', 'BS'), ('Node C', 'Node L'), ('Node C', 'Node Q'), ('Node C', 'Node R'),
        ('Node D', 'Node P'), ('Node D', 'Node E'), ('Node E', 'BS'), ('Node E', 'Node F'),
        ('Node E', 'Node N'), ('Node F', 'Node N'), ('Node F', 'Node O'), ('Node G', 'Node Q'),
        ('Node G', 'Node K'), ('Node G', 'Node N'), ('Node G', 'BS'), ('Node H', 'Node P'),
        ('Node H', 'Node J'), ('Node H', 'Node K'), ('Node H', 'Node O'), ('Node H', 'BS'),
        ('Node I', 'Node K'), ('Node I', 'Node M'), ('Node I', 'Node J'), ('Node I', 'Node P'),
        ('Node J', 'Node Q'), ('Node J', 'Node O'), ('Node J', 'Node P'), ('Node K', 'Node M'),
        ('Node L', 'Node T'), ('Node M', 'Node N'), ('Node N', 'Node P'), ('Node N', 'Node T'),
        ('Node O', 'Node S'), ('Node P', 'Node Q'), ('Node Q', 'BS'), ('Node S', 'BS')
    ]

    # First simulation environment
    env = simpy.Environment()
    dsr_network = setup_simulation(env, edge_list_dsr, protocol=DSR)
    sfa_network = setup_simulation(env, edge_list_sfa, protocol=SFA)
    networks = {DSR: dsr_network, SFA: sfa_network}
    data_store = []
    env.process(collect_data(env, networks, data_store))
    env.process(initiate_routes_dsr(env, dsr_network))
    env.process(initiate_routes_sfa(env, sfa_network))
    env.process(send_forwarding_packets_dsr(env, dsr_network))
    env.process(send_forwarding_packets_sfa(env, sfa_network))
    env.process(send_random_alarms(env, dsr_network))
    env.process(send_random_alarms(env, sfa_network))
    simulation_end_time = 100
    env.run(until=simulation_end_time)

    if data_store:
        full_data = pd.concat(data_store, ignore_index=True)
        logger.info("\nSample of first simulation data:\n{}".format(full_data.head()))
        full_data.to_csv("data/simulation_data.csv", index=False)
        logger.info("Exported first simulation data to simulation_data.csv.")
    else:
        logger.error("No data from first simulation.")

    # Train a dummy traffic-prediction model
    dl_model, scaler = train_traffic_prediction_model(data_store)

    # Second simulation environment with ML and RL enhancements
    env2 = simpy.Environment()
    dsr_network2 = setup_simulation(env2, edge_list_dsr, protocol=DSR, dl_model=dl_model, scaler=scaler)
    sfa_network2 = setup_simulation(env2, edge_list_sfa, protocol=SFA, dl_model=dl_model, scaler=scaler)
    networks2 = {DSR: dsr_network2, SFA: sfa_network2}
    env2.process(collect_data(env2, networks2, data_store))
    selected_node = dsr_network2.nodes[0]
    destination = 'BS'
    rl_model = train_rl_agent(dsr_network2, selected_node, destination, timesteps=3000)
    selected_node.rl_model = rl_model
    env2.process(initiate_routes_dsr(env2, dsr_network2))
    env2.process(initiate_routes_sfa(env2, sfa_network2))
    env2.process(send_forwarding_packets_dsr(env2, dsr_network2))
    env2.process(send_forwarding_packets_sfa(env2, sfa_network2))
    env2.process(send_random_alarms(env2, dsr_network2))
    env2.process(send_random_alarms(env2, sfa_network2))
    env2.process(initiate_routes_rl(env2, dsr_network2, selected_node, destination))
    enhanced_simulation_time = 100
    env2.run(until=enhanced_simulation_time)

    if data_store:
        full_data = pd.concat(data_store, ignore_index=True)
        full_data.to_csv("data/simulation_data.csv", index=False)
        logger.info("Second simulation data appended and exported.")
        metrics = {}
        for proto, net in networks2.items():
            total_sent = sum(n.packets_sent for n in net.nodes)
            total_recv = sum(n.packets_received for n in net.nodes)
            total_forward = metrics_summary['Forwarding_sent']
            energy_used = sum(INITIAL_ENERGY - n.energy for n in net.nodes)
            pdr = (total_recv / total_sent) if total_sent > 0 else 0
            detection_rate = (metrics_summary['Alarms_sent'] / total_forward) if total_forward > 0 else 0
            metrics[proto] = {
                'Total Packets Sent': total_sent,
                'Total Packets Received': total_recv,
                'Packet Delivery Ratio': pdr,
                'Total Energy Consumed': energy_used,
                'Detection Rate': detection_rate
            }
        metrics_df = pd.DataFrame(metrics).T
        logger.info("\nPerformance Metrics:\n%s", metrics_df)
        if metrics[SFA]['Packet Delivery Ratio'] > metrics[DSR]['Packet Delivery Ratio']:
            logger.info("SFA demonstrates a higher packet delivery ratio than DSR.")
        else:
            logger.info("DSR demonstrates a higher packet delivery ratio than SFA.")
        if metrics[SFA]['Total Energy Consumed'] < metrics[DSR]['Total Energy Consumed']:
            logger.info("SFA consumes less energy than DSR on average.")
        else:
            logger.info("DSR consumes less energy than SFA on average.")
        if metrics[SFA]['Detection Rate'] > metrics[DSR]['Detection Rate']:
            logger.info("SFA has a higher malicious detection rate compared to DSR.")
        else:
            logger.info("DSR has a higher malicious detection rate compared to SFA.")
        metrics_df.to_csv("data/performance_metrics.csv", index=True)
    else:
        logger.error("No data after second simulation.")

    logger.info("Simulation complete.")
    summary_text = (
        "Simulation Functionality Summary\n"
        "--------------------------------\n"
        f"Total RREQs sent: {metrics_summary['RREQ_sent']}\n"
        f"Total RREPs sent: {metrics_summary['RREP_sent']}\n"
        f"Total RERRs sent: {metrics_summary['RERR_sent']}\n"
        f"Total Forwarding Packets sent: {metrics_summary['Forwarding_sent']}\n"
        f"Total ALARM messages sent: {metrics_summary['Alarms_sent']}\n"
        f"Total ACKs sent: {metrics_summary['ACK_sent']}\n"
        f"Total ACKs received: {metrics_summary['ACK_received']}\n"
        f"Total DoS Packets sent: {metrics_summary['DoS_packets_sent']}\n"
        f"Total Successful Routes Established: {metrics_summary['Routes_established']}\n"
        f"Total Nodes Deactivated due to Energy Loss: {metrics_summary['Nodes_deactivated']}\n"
        "--------------------------------\n"
        "Detailed simulation data, performance metrics, and charts have been saved.\n"
        "Refer to the CSV files for analysis.\n"
        "--------------------------------\n"
        "End of Simulation Functionality Summary."
    )
    with open("data/simulation_functionality.txt", "w") as f:
        f.write(summary_text)
    logger.info("Saved simulation summary to simulation_functionality.txt.")
