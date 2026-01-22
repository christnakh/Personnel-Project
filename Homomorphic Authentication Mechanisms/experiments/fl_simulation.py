#!/usr/bin/env python3
import argparse
import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fl_pipeline.client import FLClient
from src.fl_pipeline.server import FLServer

def run_fl_round(clients, server, round_num):
    print(f"\n--- FL Round {round_num} ---")
    global_model = server.get_global_model()
    client_updates = []
    for client in tqdm(clients, desc=f"Round {round_num}: Client updates"):
        update_data = client.prepare_update(global_model)
        client_updates.append(update_data)
    print(f"Round {round_num}: Server aggregation...")
    result = server.receive_updates(client_updates)
    if isinstance(result["aggregated_update"], np.ndarray):
        server.update_global_model(result["aggregated_update"])
    return result

def main():
    parser = argparse.ArgumentParser(description="FL Simulation with Homomorphic Authentication")
    parser.add_argument("--clients", type=int, default=10,
                       help="Number of clients")
    parser.add_argument("--rounds", type=int, default=5,
                       help="Number of FL rounds")
    parser.add_argument("--model_dim", type=int, default=1000,
                       help="Model dimension")
    parser.add_argument("--auth_scheme", type=str, default="BLS",
                       choices=["BLS", "LHS", "Waters", "BonehBoyen", "RSA", "EdDSA", 
                               "Additive_HMAC", "Linear_HMAC", "Polynomial_HMAC", "Lattice_HMAC"],
                       help="Authentication scheme to use")
    parser.add_argument("--output", type=str, default="results/fl_simulation/",
                       help="Output directory")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Federated Learning Simulation with Homomorphic Authentication")
    print("=" * 80)
    print(f"Authentication Scheme: {args.auth_scheme}")
    print(f"Number of Clients: {args.clients}")
    print(f"Number of Rounds: {args.rounds}")
    print(f"Model Dimension: {args.model_dim}")
    print(f"Use HE: {args.use_he}")
    print()
    
    clients = []
    secret_key = None
    for i in range(args.clients):
        client = FLClient(client_id=i, model_dim=args.model_dim, 
                         auth_scheme=args.auth_scheme)
        if i == 0 and hasattr(client.auth_handler, 'secret_key'):
            secret_key = client.auth_handler.secret_key
        clients.append(client)
    
    server = FLServer(model_dim=args.model_dim, auth_scheme=args.auth_scheme)
    if secret_key is not None and hasattr(server.auth_verifier, 'secret_key'):
        server.auth_verifier.secret_key = secret_key
    
    all_results = []
    for round_num in range(1, args.rounds + 1):
        result = run_fl_round(clients, server, round_num)
        all_results.append(result)
    
    # Create output directory with scheme name
    output_dir = os.path.join(args.output, f"fl_simulation_{args.auth_scheme}")
    os.makedirs(output_dir, exist_ok=True)
    
    import json
    output_file = os.path.join(output_dir, "fl_simulation_metrics.json")
    with open(output_file, 'w') as f:
        json.dump({
            "rounds": len(all_results),
            "clients": args.clients,
            "auth_scheme": args.auth_scheme,
            "results": all_results
        }, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("FL Simulation Completed!")
    print(f"✓ Results saved to {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
