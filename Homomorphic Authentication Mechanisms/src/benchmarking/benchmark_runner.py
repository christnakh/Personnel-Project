import time
import os
import yaml
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm

from src.algorithms import (
    BLSSignature, LHSSignature, WatersHomomorphicSignature,
    BonehBoyenHomomorphicSignature, RSASignature, EdDSASignature,
    AdditiveHMAC, LinearHMAC, PolynomialHMAC, LatticeHMAC
)
from src.benchmarking.metrics_collector import MetricsCollector
from src.benchmarking.security_benchmarks import SecurityBenchmark
from src.fl_pipeline.client import FLClient
from src.fl_pipeline.server import FLServer

class BenchmarkRunner:
    def __init__(self, config_path: str = "config/benchmark_config.yaml"):
        self.config = self._load_config(config_path)
        self.algorithms = {}
        self.metrics_collector = MetricsCollector()
        self.security_benchmark = SecurityBenchmark()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def initialize_algorithms(self):
        """Initialize algorithms, skipping those that fail"""
        if self.config.get("algorithms", {}).get("homomorphic_signatures"):
            for algo_config in self.config["algorithms"]["homomorphic_signatures"]:
                if algo_config.get("enabled", True):
                    name = algo_config["name"]
                    try:
                        if name == "BLS":
                            self.algorithms[name] = BLSSignature()
                        elif name == "LHS":
                            vector_dim = self.config.get("benchmark", {}).get("vector_dimensions", [100])[0]
                            self.algorithms[name] = LHSSignature(vector_dim=vector_dim)
                        elif name == "Waters":
                            self.algorithms[name] = WatersHomomorphicSignature()
                        elif name == "BonehBoyen":
                            self.algorithms[name] = BonehBoyenHomomorphicSignature()
                        elif name == "RSA":
                            self.algorithms[name] = RSASignature()
                        elif name == "EdDSA":
                            self.algorithms[name] = EdDSASignature()
                        print(f"  ✓ {name} initialized")
                    except Exception as e:
                        print(f"  ✗ {name} failed: {e}")
                        continue
        
        if self.config.get("algorithms", {}).get("homomorphic_mac"):
            for algo_config in self.config["algorithms"]["homomorphic_mac"]:
                if algo_config.get("enabled", True):
                    name = algo_config["name"]
                    try:
                        if name == "Additive_HMAC":
                            self.algorithms[name] = AdditiveHMAC()
                        elif name == "Linear_HMAC":
                            vector_dim = self.config.get("benchmark", {}).get("vector_dimensions", [100])[0]
                            self.algorithms[name] = LinearHMAC(vector_dim=vector_dim)
                        elif name == "Polynomial_HMAC":
                            self.algorithms[name] = PolynomialHMAC()
                        elif name == "Lattice_HMAC":
                            self.algorithms[name] = LatticeHMAC()
                        print(f"  ✓ {name} initialized")
                    except Exception as e:
                        print(f"  ✗ {name} failed: {e}")
                        continue
        
        if not self.algorithms:
            raise RuntimeError("No algorithms initialized!")
        print(f"\n✓ Initialized {len(self.algorithms)} algorithm(s)")
    
    def benchmark_key_generation(self, algorithm_name: str, iterations: int = 10) -> Dict[str, Any]:
        algo = self.algorithms.get(algorithm_name)
        if not algo:
            return {}
        times = []
        for _ in range(iterations):
            start = time.time()
            if hasattr(algo, 'key_generation'):
                algo.key_generation()
            times.append(time.time() - start)
        avg_time = np.mean(times)
        self.metrics_collector.record_performance(
            algorithm=algorithm_name,
            operation="key_generation",
            key_gen_time=avg_time
        )
        return {"avg_time": avg_time, "iterations": iterations}
    
    def benchmark_signing(self, algorithm_name: str, message_sizes: List[int], iterations: int = 10) -> Dict[str, Any]:
        algo = self.algorithms.get(algorithm_name)
        if not algo:
            return {}
        results = {}
        for msg_size in message_sizes:
            times = []
            for _ in range(iterations):
                message = os.urandom(msg_size)
                if hasattr(algo, 'sign_vector'):
                    vector_dim = getattr(algo, 'vector_dim', 100)
                    vector = np.random.randn(vector_dim).astype(np.float32)
                    _, sign_time = algo.sign_vector(vector, message[:16])
                    times.append(sign_time)
                elif hasattr(algo, 'sign'):
                    _, sign_time = algo.sign(message)
                    times.append(sign_time)
                elif hasattr(algo, 'generate_tag'):
                    _, gen_time = algo.generate_tag(message, b"test_id")
                    times.append(gen_time)
            avg_time = np.mean(times)
            results[msg_size] = {"avg_time": avg_time}
            self.metrics_collector.record_performance(
                algorithm=algorithm_name,
                operation=f"signing_{msg_size}",
                sign_time=avg_time
            )
        return results
    
    def benchmark_verification(self, algorithm_name: str, message_sizes: List[int], iterations: int = 10) -> Dict[str, Any]:
        algo = self.algorithms.get(algorithm_name)
        if not algo:
            return {}
        results = {}
        for msg_size in message_sizes:
            times = []
            for _ in range(iterations):
                message = os.urandom(msg_size)
                if hasattr(algo, 'sign_vector'):
                    vector_dim = getattr(algo, 'vector_dim', 100)
                    vector = np.random.randn(vector_dim).astype(np.float32)
                    file_id = message[:16]
                    sig, _ = algo.sign_vector(vector, file_id)
                    _, verify_time = algo.verify_vector(vector, sig, file_id)
                    times.append(verify_time)
                elif hasattr(algo, 'sign'):
                    sig, _ = algo.sign(message)
                    _, verify_time = algo.verify(message, sig)
                    times.append(verify_time)
                elif hasattr(algo, 'generate_tag'):
                    tag, _ = algo.generate_tag(message, b"test_id")
                    _, verify_time = algo.verify_tag(message, tag, b"test_id")
                    times.append(verify_time)
            results[msg_size] = {"avg_time": np.mean(times)}
            self.metrics_collector.record_performance(
                algorithm=algorithm_name,
                operation=f"verification_{msg_size}",
                verify_time=np.mean(times)
            )
        return results
    
    def benchmark_aggregation(self, algorithm_name: str, num_clients: List[int], message_size: int = 4096, iterations: int = 5) -> Dict[str, Any]:
        algo = self.algorithms.get(algorithm_name)
        if not algo:
            return {}
        results = {}
        for n_clients in num_clients:
            times = []
            for _ in range(iterations):
                signatures = []
                for _ in range(n_clients):
                    message = os.urandom(message_size)
                    if hasattr(algo, 'sign_vector'):
                        vector_dim = getattr(algo, 'vector_dim', 100)
                        vector = np.random.randn(vector_dim).astype(np.float32)
                        file_id = message[:16]
                        sig, _ = algo.sign_vector(vector, file_id)
                        signatures.append(sig)
                    elif hasattr(algo, 'sign'):
                        sig, _ = algo.sign(message)
                        signatures.append(sig)
                    elif hasattr(algo, 'generate_tag'):
                        tag, _ = algo.generate_tag(message, b"test_id")
                        signatures.append(tag)
                start = time.time()
                if hasattr(algo, 'aggregate_signatures'):
                    _, agg_time = algo.aggregate_signatures(signatures)
                    times.append(agg_time)
                elif hasattr(algo, 'combine_tags'):
                    _, combine_time = algo.combine_tags(signatures)
                    times.append(combine_time)
                else:
                    times.append(0.0)
            results[n_clients] = {"avg_time": np.mean(times)}
            self.metrics_collector.record_performance(
                algorithm=algorithm_name,
                operation=f"aggregation_{n_clients}",
                aggregate_time=np.mean(times)
            )
        return results
    
    def benchmark_communication_overhead(self, algorithm_name: str) -> Dict[str, Any]:
        algo = self.algorithms.get(algorithm_name)
        if not algo:
            return {}
        sig_size = algo.get_signature_size() if hasattr(algo, 'get_signature_size') else (algo.get_tag_size() if hasattr(algo, 'get_tag_size') else 0)
        pub_key_size = algo.get_public_key_size() if hasattr(algo, 'get_public_key_size') else 0
        priv_key_size = algo.get_key_size() if hasattr(algo, 'get_key_size') else 0
        self.metrics_collector.record_communication(
            algorithm=algorithm_name,
            signature_size=sig_size,
            public_key_size=pub_key_size,
            private_key_size=priv_key_size
        )
        return {"signature_size": sig_size, "public_key_size": pub_key_size, "private_key_size": priv_key_size}
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        print("Initializing algorithms...")
        self.initialize_algorithms()
        benchmark_config = self.config.get("benchmark", {})
        fl_config = self.config.get("fl_simulation", {})
        num_clients = benchmark_config.get("num_clients", [10, 50, 100])
        message_sizes = benchmark_config.get("message_sizes", [128, 512, 1024, 4096])
        iterations = benchmark_config.get("iterations", 10)
        results = {}
        
        # Standard algorithm benchmarks
        for algo_name in tqdm(self.algorithms.keys(), desc="Benchmarking"):
            print(f"\nBenchmarking {algo_name}...")
            algo_results = {}
            algo_results["key_generation"] = self.benchmark_key_generation(algo_name, iterations)
            algo_results["signing"] = self.benchmark_signing(algo_name, message_sizes, iterations)
            algo_results["verification"] = self.benchmark_verification(algo_name, message_sizes, iterations)
            algo_results["aggregation"] = self.benchmark_aggregation(algo_name, num_clients, message_sizes[1], iterations // 2)
            algo_results["communication"] = self.benchmark_communication_overhead(algo_name)
            security_result = self.security_benchmark.run_security_suite(self.algorithms[algo_name], algo_name)
            algo_results["security"] = security_result
            security_metrics = {k: v for k, v in security_result.items() if k != 'algorithm'}
            self.metrics_collector.record_security(algorithm=algo_name, **security_metrics)
            results[algo_name] = algo_results
        
        # FL benchmarking
        print("\n" + "="*80)
        print("FL Benchmarking")
        print("="*80)
        fl_rounds = fl_config.get("num_rounds", [1, 5, 10, 20])
        fl_model_sizes = fl_config.get("model_sizes", [100, 500, 1000, 5000])
        fl_client_counts = fl_config.get("client_counts", [10, 50, 100])
        
        # Benchmark FL with different auth schemes
        print(f"\nFL Benchmarking - Available schemes: {list(self.algorithms.keys())}")
        for auth_scheme in ["BLS", "Additive_HMAC", "Linear_HMAC"]:
            # Check if scheme is available (handle underscore variations)
            scheme_available = False
            if auth_scheme in self.algorithms:
                scheme_available = True
            else:
                # Check without underscores
                auth_normalized = auth_scheme.replace("_", "")
                for algo_name in self.algorithms.keys():
                    if algo_name.replace("_", "") == auth_normalized:
                        scheme_available = True
                        break
            
            if scheme_available:
                print(f"\nFL Benchmarking with {auth_scheme}...")
                try:
                    fl_results = self.benchmark_fl_scenarios(
                        auth_scheme=auth_scheme,
                        num_rounds=fl_rounds[:3],  # Limit for speed
                        model_sizes=fl_model_sizes[:3],
                        client_counts=fl_client_counts[:2]
                    )
                    if fl_results:
                        results[f"FL_{auth_scheme}"] = fl_results
                        print(f"  ✓ Collected {len(fl_results)} FL scenarios for {auth_scheme}")
                    else:
                        print(f"  ⚠ No FL results collected for {auth_scheme}")
                except Exception as e:
                    print(f"  ✗ FL benchmarking failed for {auth_scheme}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  ⚠ Skipping {auth_scheme} - not available in algorithms")
        
        return results
    
    def benchmark_fl_scenarios(self, auth_scheme: str = "BLS", 
                               num_rounds: List[int] = [1, 5, 10, 20],
                               model_sizes: List[int] = [100, 500, 1000, 5000],
                               client_counts: List[int] = [10, 50, 100]) -> Dict[str, Any]:
        """Benchmark FL scenarios with different rounds, model sizes, and client counts"""
        results = {}
        
        for model_dim in model_sizes:
            for num_clients in client_counts:
                for rounds in num_rounds:
                    key = f"{auth_scheme}_model{model_dim}_clients{num_clients}_rounds{rounds}"
                    print(f"  Benchmarking FL: {key}...")
                    
                    # Setup FL
                    server = FLServer(model_dim=model_dim, auth_scheme=auth_scheme)
                    clients = [FLClient(client_id=i, model_dim=model_dim, auth_scheme=auth_scheme) 
                              for i in range(num_clients)]
                    
                    # Run FL rounds
                    round_times = []
                    verify_times = []
                    for round_num in range(rounds):
                        start = time.time()
                        
                        # Clients prepare updates
                        client_updates = []
                        for client in clients:
                            update_data = client.prepare_update(server.get_global_model())
                            client_updates.append(update_data)
                        
                        # Server receives and verifies (server-side verification)
                        agg_result = server.receive_updates(client_updates)
                        aggregated = agg_result.get("aggregated_update")
                        
                        if aggregated is not None:
                            # Aggregate auth tags
                            auth_tags = [u.get("auth_tag") for u in client_updates]
                            auth_metadata = [u.get("auth_metadata", {}) for u in client_updates]
                            updates = [u.get("update") for u in client_updates]
                            client_ids = [u.get("client_id") for u in client_updates]
                            
                            agg_tag, _ = server.aggregate_auth_tags(auth_tags, auth_metadata, updates, client_ids)
                            
                            # Server-side verification
                            verify_start = time.time()
                            valid, verify_meta = server.verify_aggregated(aggregated, agg_tag, updates, auth_tags, auth_metadata, client_ids)
                            verify_times.append(time.time() - verify_start)
                            
                            if valid:
                                # Update global model
                                server.update_global_model(aggregated, learning_rate=0.01)
                                
                                # Client-side verification
                                for client in clients:
                                    client.verify_server_response(aggregated, agg_tag, verify_meta)
                        
                        round_times.append(time.time() - start)
                    
                    results[key] = {
                        "model_dim": model_dim,
                        "num_clients": num_clients,
                        "num_rounds": rounds,
                        "avg_round_time": np.mean(round_times),
                        "avg_verify_time": np.mean(verify_times) if verify_times else 0,
                        "total_time": sum(round_times)
                    }
                    
                    # Record metrics
                    self.metrics_collector.record_performance(
                        algorithm=f"FL_{auth_scheme}",
                        operation=f"fl_rounds_{rounds}",
                        total_time=sum(round_times),
                        verify_time=np.mean(verify_times) if verify_times else 0
                    )
        
        return results
    
    def export_results(self, output_dir: str = "results/"):
        """Export all benchmark results"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nExporting results to {output_dir}...")
        try:
            self.metrics_collector.export_all_formats(output_dir, prefix="metrics")
            print(f"✓ All results exported to {output_dir}")
        except Exception as e:
            print(f"Export failed: {e}")
            import traceback
            traceback.print_exc()
