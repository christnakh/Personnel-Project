# Project: Optimizing the Dynamic Source Routing (DSR) Protocol

## Overview
This project focuses on enhancing the Dynamic Source Routing (DSR) protocol within a simulated network environment. The improvements are aimed at optimizing energy consumption, improving route discovery and management, refining alarm handling for fault detection, and incorporating machine learning (ML) and reinforcement learning (RL) techniques to further enhance routing decisions.

The simulation is built using Python with libraries such as **SimPy** for process-based discrete-event simulation, **NetworkX** for network graph management, **TensorFlow** and **Stable Baselines3** for ML and RL integration, and **Pandas** for data collection and analysis.

## Key Features and Enhancements

### 1. **Energy-Efficient Routing Strategies**
- **Load Balancing:** The protocol now considers the residual energy of nodes during route selection to evenly distribute traffic and prevent overburdening specific nodes.
  - **Implementation:** Nodes with higher energy reserves are prioritized in route selection (`get_route()` method).

- **Energy-Aware Routing:** The route discovery process incorporates dynamic back-off times based on the energy levels of the nodes, discouraging low-energy nodes from frequent participation in routing.
  - **Implementation:** The `initiate_route_discovery()` method computes a dynamic back-off period inversely proportional to the node's energy level.

### 2. **Improved Alarm Sensitivity and Handling**
- **Refined Alarm Criteria:** Alarms are triggered based on a combination of energy discrepancies and packet loss rates, providing a more comprehensive detection mechanism.
  - **Implementation:** The `handle_alarm()` method now checks both residual energy and packet loss.

- **Preemptive Actions on Alarms:** Upon detecting anomalies, the system proactively broadcasts block messages to isolate potentially faulty nodes.
  - **Implementation:** The `broadcast_block_message()` method in the `BaseStation` class sends block messages to all nodes when malicious activity is detected.

### 3. **Optimized Route Request (RREQ) and Duplicate Handling**
- **Reduced Redundant RREQs:** The protocol tracks RREQs using sequence numbers and TTL values to minimize unnecessary broadcasts.
  - **Implementation:** The `handle_rreq()` method allows processing of duplicate RREQs if they have higher TTL values, balancing redundancy with efficiency.

- **Cooldown Period:** A cooldown mechanism prevents frequent RREQs from the same node, reducing network congestion.
  - **Implementation:** The `initiate_route_discovery()` method uses a timestamp-based back-off mechanism.

### 4. **Enhanced Node Energy Management**
- **Adaptive Transmission Power:** Nodes dynamically adjust their transmission power based on the distance to the next hop, conserving energy for shorter transmissions.
  - **Implementation:** The `_send_packet()` method calculates energy consumption using Euclidean distance between nodes.

- **Sleep/Wake Cycles for Idle Nodes:** Nodes enter low-power sleep mode after a period of inactivity and wake up when needed.
  - **Implementation:** The `manage_sleep_cycle()` method monitors node activity and transitions nodes between active and sleep states.

### 5. **Increased Route Cache Size**
- **Expanded Cached Routes:** The number of alternate routes stored per destination has been increased, reducing the need for frequent route discoveries.
  - **Implementation:** The `MAX_ROUTES` constant is increased from 3 to 5, and routing table management functions are adjusted accordingly.

### 6. **Machine Learning and Reinforcement Learning Integration**
- **Traffic Prediction Model:** A neural network model predicts traffic load, aiding in routing decisions under high-traffic conditions.
  - **Implementation:** The `train_traffic_prediction_model()` function uses TensorFlow to train a model on collected simulation data.

- **Reinforcement Learning for Dynamic Routing:** The PPO algorithm from Stable Baselines3 is used to train an RL agent that makes routing decisions based on the current network state.
  - **Implementation:** The `train_rl_agent()` function sets up an RL environment where the agent learns optimal routing strategies.

## Simulation Components

### 1. **Network Setup**
- Nodes are created using the `Node` class, each with unique IDs and randomly assigned positions.
- The network topology is defined using an edge list, managed by the `Network` class, which leverages NetworkX for graph management.

### 2. **Packet Handling**
- The `Packet` class defines different packet types (RREQ, RREP, RERR, FORWARDING, ALARM, BLOCK) and their attributes.
- Nodes handle packet transmission and reception with energy consumption considerations and protocol-specific behaviors.

### 3. **Base Station**
- A specialized node that handles alarm aggregation and broadcasts block messages to isolate malicious or faulty nodes.

### 4. **Data Collection and Analysis**
- Simulation data is collected periodically and stored in CSV files for further analysis.
- Key metrics include total packets sent/received, energy consumption, packet delivery ratio (PDR), and detection rate.

## Performance Metrics
- **Packet Delivery Ratio (PDR):** Measures the efficiency of data delivery across the network.
- **Total Energy Consumed:** Tracks energy usage to evaluate the effectiveness of energy-saving strategies.
- **Detection Rate:** Assesses the accuracy of the alarm system in identifying faulty or malicious nodes.

## Running the Simulation
1. How to install python on window and mac:
window : [text](https://www.youtube.com/watch?v=IPOr0ran2Oo)
mac: [text](https://www.youtube.com/watch?v=nhv82tvFfkM)
2. Ensure all required Python packages are installed:
   ```bash
   window:
   pip install simpy networkx pandas matplotlib tensorflow stable-baselines3 gym numpy

   mac: 
   pip3 install simpy networkx pandas matplotlib tensorflow stable-baselines3 gym numpy
   ```

3. Run the main simulation script:
   ```bash
   window:
   python simulation.py
   mac:
   python3 simulation.py
   ```

4. Analyze the generated CSV files (`simulation_data.csv`, `performance_metrics.csv`) and the log file (`simulation_output.txt`) for detailed performance insights.

## Conclusion
This project demonstrates a comprehensive approach to optimizing the DSR protocol through energy-efficient routing, advanced fault detection, and intelligent decision-making using ML and RL. The simulation provides a robust framework for testing and further enhancing network protocols in dynamic and resource-constrained environments.


