import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import os
from datetime import datetime

class Visualizer:
    # Algorithms to exclude from all graphs
    EXCLUDED_ALGORITHMS = {"RSA", "EdDSA"}
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def _filter_algorithms(self, algorithms: List[str]) -> List[str]:
        """Filter out excluded algorithms (RSA, EdDSA)"""
        return [algo for algo in algorithms if algo not in self.EXCLUDED_ALGORITHMS]
    
    def _filter_metrics_data(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out excluded algorithms from metrics data"""
        return {k: v for k, v in metrics_data.items() if k not in self.EXCLUDED_ALGORITHMS}
    
    def plot_performance_comparison(self, metrics_data: Dict[str, Any], 
                                   output_file: str = "performance_comparison.png"):
        # Filter out RSA and EdDSA
        metrics_data = self._filter_metrics_data(metrics_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        algorithms = self._filter_algorithms(list(metrics_data.keys()))
        
        key_gen_times = [metrics_data[algo].get("key_generation", {}).get("avg_time", 0) 
                        for algo in algorithms]
        axes[0, 0].bar(algorithms, key_gen_times)
        axes[0, 0].set_title("Key Generation Time", fontweight='bold')
        axes[0, 0].set_ylabel("Time (seconds)", fontsize=10)
        axes[0, 0].tick_params(axis='x', rotation=45, labelsize=9)
        if key_gen_times and max(key_gen_times) > 0:
            if max(key_gen_times) / (min([t for t in key_gen_times if t > 0]) or 1) > 100:
                axes[0, 0].set_yscale('log')
        
        sign_times = []
        for algo in algorithms:
            signing_data = metrics_data[algo].get("signing", {})
            if signing_data:
                times = [v.get("avg_time", 0) for k, v in signing_data.items() if isinstance(k, int)]
                sign_times.append(times[1] if len(times) > 1 else times[0] if times else 0)
            else:
                sign_times.append(0)
        axes[0, 1].bar(algorithms, sign_times)
        axes[0, 1].set_title("Signing Time (4KB message)", fontweight='bold')
        axes[0, 1].set_ylabel("Time (seconds)", fontsize=10)
        axes[0, 1].tick_params(axis='x', rotation=45, labelsize=9)
        if sign_times and max(sign_times) > 0:
            if max(sign_times) / (min([t for t in sign_times if t > 0]) or 1) > 100:
                axes[0, 1].set_yscale('log')
        
        verify_times = []
        for algo in algorithms:
            verify_data = metrics_data[algo].get("verification", {})
            if verify_data:
                times = [v.get("avg_time", 0) for k, v in verify_data.items() if isinstance(k, int)]
                verify_times.append(times[1] if len(times) > 1 else times[0] if times else 0)
            else:
                verify_times.append(0)
        axes[1, 0].bar(algorithms, verify_times)
        axes[1, 0].set_title("Verification Time (4KB message)", fontweight='bold')
        axes[1, 0].set_ylabel("Time (seconds)", fontsize=10)
        axes[1, 0].tick_params(axis='x', rotation=45, labelsize=9)
        if verify_times and max(verify_times) > 0:
            if max(verify_times) / (min([t for t in verify_times if t > 0]) or 1) > 100:
                axes[1, 0].set_yscale('log')
        
        agg_times = []
        for algo in algorithms:
            agg_data = metrics_data[algo].get("aggregation", {})
            if agg_data:
                times = [v.get("avg_time", 0) for k, v in agg_data.items() if isinstance(k, int)]
                agg_times.append(times[-1] if times else 0)
            else:
                agg_times.append(0)
        axes[1, 1].bar(algorithms, agg_times)
        axes[1, 1].set_title("Aggregation Time (100 clients)", fontweight='bold')
        axes[1, 1].set_ylabel("Time (seconds)", fontsize=10)
        axes[1, 1].tick_params(axis='x', rotation=45, labelsize=9)
        if agg_times and max(agg_times) > 0:
            if max(agg_times) / (min([t for t in agg_times if t > 0]) or 1) > 100:
                axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, output_file)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {output_file}")
    
    def plot_scalability(self, metrics_data: Dict[str, Any],
                        output_file: str = "scalability.png"):
        # Filter out RSA and EdDSA
        metrics_data = self._filter_metrics_data(metrics_data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        algorithms = self._filter_algorithms(list(metrics_data.keys()))
        
        for algo in algorithms:
            agg_data = metrics_data[algo].get("aggregation", {})
            if agg_data:
                num_clients = sorted([k for k in agg_data.keys() if isinstance(k, int)])
                times = [agg_data[n].get("avg_time", 0) for n in num_clients]
                if times and max(times) > 0:
                    ax.plot(num_clients, times, marker='o', label=algo, linewidth=2, markersize=8)
        
        ax.set_xlabel("Number of Clients", fontsize=12)
        ax.set_ylabel("Aggregation Time (seconds)", fontsize=12)
        ax.set_title("Scalability: Aggregation Time vs Number of Clients", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Use log scale if values vary widely
        if times:
            max_time = max([max([agg_data[n].get("avg_time", 0) for n in sorted([k for k in agg_data.keys() if isinstance(k, int)])]) 
                          for algo, agg_data in [(a, metrics_data[a].get("aggregation", {})) for a in algorithms] if agg_data])
            min_time = min([min([agg_data[n].get("avg_time", 0) for n in sorted([k for k in agg_data.keys() if isinstance(k, int)])]) 
                          for algo, agg_data in [(a, metrics_data[a].get("aggregation", {})) for a in algorithms] if agg_data])
            if max_time > 0 and max_time / min_time > 100:
                ax.set_yscale('log')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, output_file)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {output_file}")
    
    def plot_communication_overhead(self, metrics_data: Dict[str, Any],
                                   output_file: str = "communication_overhead.png"):
        # Filter out RSA and EdDSA
        metrics_data = self._filter_metrics_data(metrics_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        algorithms = self._filter_algorithms(list(metrics_data.keys()))
        
        sig_sizes = []
        for algo in algorithms:
            comm_data = metrics_data[algo].get("communication", {})
            sig_sizes.append(comm_data.get("signature_size", 0))
        
        axes[0].bar(algorithms, sig_sizes)
        axes[0].set_title("Signature/Tag Size", fontweight='bold')
        axes[0].set_ylabel("Size (bytes)", fontsize=10)
        axes[0].tick_params(axis='x', rotation=45, labelsize=9)
        if sig_sizes and max(sig_sizes) > 0:
            if max(sig_sizes) / (min([s for s in sig_sizes if s > 0]) or 1) > 100:
                axes[0].set_yscale('log')
        
        pub_key_sizes = []
        for algo in algorithms:
            comm_data = metrics_data[algo].get("communication", {})
            pub_key_sizes.append(comm_data.get("public_key_size", 0))
        
        axes[1].bar(algorithms, pub_key_sizes)
        axes[1].set_title("Public Key Size", fontweight='bold')
        axes[1].set_ylabel("Size (bytes)", fontsize=10)
        axes[1].tick_params(axis='x', rotation=45, labelsize=9)
        if pub_key_sizes and max(pub_key_sizes) > 0:
            if max(pub_key_sizes) / (min([s for s in pub_key_sizes if s > 0]) or 1) > 100:
                axes[1].set_yscale('log')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, output_file)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {output_file}")
    
    def plot_message_size_impact(self, metrics_data: Dict[str, Any],
                                output_file: str = "message_size_impact.png"):
        # Filter out RSA and EdDSA
        metrics_data = self._filter_metrics_data(metrics_data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        algorithms = self._filter_algorithms(list(metrics_data.keys()))
        message_sizes = [1024, 4096, 16384]
        
        for algo in algorithms:
            signing_data = metrics_data[algo].get("signing", {})
            if signing_data:
                times = [signing_data.get(size, {}).get("avg_time", 0) for size in message_sizes]
                ax.plot(message_sizes, times, marker='o', label=algo, linewidth=2)
        
        ax.set_xlabel("Message Size (bytes)", fontsize=12)
        ax.set_ylabel("Signing Time (seconds)", fontsize=12)
        ax.set_title("Message Size Impact on Signing Time", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        # Use log scale for y-axis if values vary widely
        if times:
            all_times = [signing_data.get(size, {}).get("avg_time", 0) for algo in algorithms for size in message_sizes if metrics_data[algo].get("signing", {})]
            if all_times and max(all_times) > 0:
                if max(all_times) / (min([t for t in all_times if t > 0]) or 1) > 100:
                    ax.set_yscale('log')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, output_file)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {output_file}")
    
    def plot_fl_benchmarks(self, metrics_data: Dict[str, Any],
                          output_file: str = "fl_benchmarks.png"):
        """Plot FL benchmarking results"""
        fl_data = {k: v for k, v in metrics_data.items() if k.startswith("FL_")}
        if not fl_data:
            print("  ⚠ No FL benchmarking data found, skipping FL plots")
            return
        
        print(f"  Found FL data for schemes: {list(fl_data.keys())}")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Collect data for plotting
        # FL data structure: {"FL_BLS": {"BLS_model100_clients10_rounds1": {...}, ...}, ...}
        schemes_data = {}
        for scheme_key, nested_data in fl_data.items():
            scheme_name = scheme_key.replace("FL_", "")
            
            # Handle nested structure: each scheme has multiple scenarios
            if isinstance(nested_data, dict):
                if scheme_name not in schemes_data:
                    schemes_data[scheme_name] = {"rounds": [], "round_times": [], "model_sizes": [], 
                                                  "total_times": [], "verify_times": [], "client_counts": []}
                
                # Iterate through all scenarios for this scheme
                for scenario_key, scenario_data in nested_data.items():
                    if isinstance(scenario_data, dict) and "model_dim" in scenario_data:
                        num_rounds = scenario_data.get("num_rounds", 0)
                        avg_round_time = scenario_data.get("avg_round_time", 0)
                        model_dim = scenario_data.get("model_dim", 0)
                        total_time = scenario_data.get("total_time", 0)
                        avg_verify_time = scenario_data.get("avg_verify_time", 0)
                        num_clients = scenario_data.get("num_clients", 0)
                        
                        # Add all data points (they're paired by scenario)
                        schemes_data[scheme_name]["rounds"].append(num_rounds)
                        schemes_data[scheme_name]["round_times"].append(avg_round_time)
                        schemes_data[scheme_name]["model_sizes"].append(model_dim)
                        schemes_data[scheme_name]["total_times"].append(total_time)
                        schemes_data[scheme_name]["verify_times"].append(avg_verify_time)
                        schemes_data[scheme_name]["client_counts"].append(num_clients)
        
        # Debug: Print collected data
        print(f"  Collected data for {len(schemes_data)} schemes")
        for scheme, data in schemes_data.items():
            print(f"    {scheme}: {len(data['rounds'])} data points")
            if data['rounds']:
                print(f"      Rounds: {data['rounds'][:5]}...")
                print(f"      Times: {data['round_times'][:5]}...")
        
        # Plot 1: Round time vs number of rounds
        has_data_plot1 = False
        all_rounds_plot = []
        all_times_plot = []
        for scheme, data in schemes_data.items():
            # Filter valid data points
            valid_pairs = [(r, t) for r, t in zip(data["rounds"], data["round_times"]) if r > 0 and t > 0]
            if valid_pairs:
                # Group by rounds and average round times for same number of rounds
                rounds_dict = {}
                for r, t in valid_pairs:
                    if r not in rounds_dict:
                        rounds_dict[r] = []
                    rounds_dict[r].append(t)
                
                if rounds_dict:
                    sorted_rounds = sorted(rounds_dict.keys())
                    avg_times = [np.mean(rounds_dict[r]) for r in sorted_rounds]
                    axes[0, 0].plot(sorted_rounds, avg_times, marker='o', label=scheme, linewidth=2, markersize=8)
                    all_rounds_plot.extend(sorted_rounds)
                    all_times_plot.extend(avg_times)
                    has_data_plot1 = True
                    print(f"    Plot 1 - {scheme}: {len(sorted_rounds)} points, rounds={sorted_rounds}, times={[f'{t:.4f}' for t in avg_times]}")
        
        axes[0, 0].set_xlabel("Number of Rounds", fontsize=12)
        axes[0, 0].set_ylabel("Average Round Time (seconds)", fontsize=12)
        axes[0, 0].set_title("FL Round Time vs Number of Rounds", fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        if has_data_plot1 and all_rounds_plot and all_times_plot:
            axes[0, 0].legend(fontsize=10)
            # Set reasonable axis limits based on actual data
            axes[0, 0].set_xlim(left=0, right=max(all_rounds_plot) * 1.1 if all_rounds_plot else 10)
            axes[0, 0].set_ylim(bottom=0, top=max(all_times_plot) * 1.2 if all_times_plot else 1)
        else:
            axes[0, 0].text(0.5, 0.5, 'No data available', ha='center', va='center', 
                           transform=axes[0, 0].transAxes, fontsize=12)
            axes[0, 0].set_xlim(0, 10)
            axes[0, 0].set_ylim(0, 1)
        
        # Plot 2: Total time vs model size
        has_data_plot2 = False
        all_model_sizes_plot = []
        all_total_times_plot = []
        for scheme, data in schemes_data.items():
            model_sizes = [s for s in data["model_sizes"] if s > 0]
            total_times = [t for i, t in enumerate(data["total_times"]) if data["model_sizes"][i] > 0 and t > 0]
            if model_sizes and total_times:
                axes[0, 1].scatter(model_sizes, total_times, s=100, alpha=0.6, label=scheme)
                all_model_sizes_plot.extend(model_sizes)
                all_total_times_plot.extend(total_times)
                has_data_plot2 = True
                print(f"    Plot 2 - {scheme}: {len(model_sizes)} points, sizes={model_sizes[:3]}, times={[f'{t:.4f}' for t in total_times[:3]]}")
        
        axes[0, 1].set_xlabel("Model Size (dimensions)", fontsize=12)
        axes[0, 1].set_ylabel("Total Time (seconds)", fontsize=12)
        axes[0, 1].set_title("FL Total Time vs Model Size", fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        if has_data_plot2 and all_model_sizes_plot and all_total_times_plot:
            axes[0, 1].legend(fontsize=10)
            # Only use log scale if values vary widely AND are all positive
            min_time = min([t for t in all_total_times_plot if t > 0]) if all_total_times_plot else 1
            max_time = max(all_total_times_plot) if all_total_times_plot else 1
            if max_time > 0 and min_time > 0 and max_time / min_time > 100:
                axes[0, 1].set_yscale('log')
                axes[0, 1].set_xscale('log')
                # For log scale, set positive limits
                axes[0, 1].set_xlim(left=min(all_model_sizes_plot) * 0.9 if all_model_sizes_plot else 1, 
                                  right=max(all_model_sizes_plot) * 1.1 if all_model_sizes_plot else 1000)
                axes[0, 1].set_ylim(bottom=min_time * 0.9, top=max_time * 1.2)
            else:
                # Linear scale with proper limits
                axes[0, 1].set_xlim(left=0, right=max(all_model_sizes_plot) * 1.1 if all_model_sizes_plot else 1000)
                axes[0, 1].set_ylim(bottom=0, top=max_time * 1.2 if max_time > 0 else 10)
        else:
            axes[0, 1].text(0.5, 0.5, 'No data available', ha='center', va='center', 
                           transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_xlim(0, 1000)
            axes[0, 1].set_ylim(0, 10)
        
        # Plot 3: Verification time by scheme
        scheme_names = []
        verify_times = []
        for scheme, data in schemes_data.items():
            valid_verify_times = [t for t in data["verify_times"] if t > 0]
            if valid_verify_times:
                avg_verify = np.mean(valid_verify_times)
                scheme_names.append(scheme)
                verify_times.append(avg_verify)
                print(f"    Plot 3 - {scheme}: avg_verify_time={avg_verify:.6f}")
        
        axes[1, 0].set_ylabel("Average Verification Time (seconds)", fontsize=12)
        axes[1, 0].set_title("FL Verification Time by Scheme", fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        if verify_times:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            axes[1, 0].bar(range(len(verify_times)), verify_times, color=colors[:len(verify_times)])
            axes[1, 0].set_xticks(range(len(scheme_names)))
            axes[1, 0].set_xticklabels(scheme_names, rotation=45, ha='right', fontsize=9)
            max_verify = max(verify_times) if verify_times else 1
            min_verify = min(verify_times) if verify_times else 0
            # Use log scale if values vary widely
            if max_verify > 0 and min_verify > 0 and max_verify / min_verify > 100:
                axes[1, 0].set_yscale('log')
                axes[1, 0].set_ylim(bottom=min_verify * 0.9, top=max_verify * 1.2)
            else:
                axes[1, 0].set_ylim(bottom=0, top=max_verify * 1.2)
        else:
            axes[1, 0].text(0.5, 0.5, 'No data available', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_xlim(-0.5, 4.5)
            axes[1, 0].set_ylim(0, 1)
        
        # Plot 4: Client count impact
        has_data_plot4 = False
        all_client_counts_plot = []
        all_round_times_plot = []
        for scheme, data in schemes_data.items():
            # Pair client_counts with round_times (they're already paired by index)
            valid_pairs = [(c, t) for c, t in zip(data["client_counts"], data["round_times"]) if c > 0 and t > 0]
            if valid_pairs:
                client_counts = [c for c, t in valid_pairs]
                round_times = [t for c, t in valid_pairs]
                axes[1, 1].scatter(client_counts, round_times, s=100, alpha=0.6, label=scheme)
                all_client_counts_plot.extend(client_counts)
                all_round_times_plot.extend(round_times)
                has_data_plot4 = True
                print(f"    Plot 4 - {scheme}: {len(client_counts)} points, clients={client_counts[:3]}, times={[f'{t:.4f}' for t in round_times[:3]]}")
        
        axes[1, 1].set_xlabel("Number of Clients", fontsize=12)
        axes[1, 1].set_ylabel("Average Round Time (seconds)", fontsize=12)
        axes[1, 1].set_title("FL Round Time vs Client Count", fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        if has_data_plot4 and all_client_counts_plot and all_round_times_plot:
            axes[1, 1].legend(fontsize=10)
            # Only use log scale if values vary widely AND are all positive
            min_round_time = min([t for t in all_round_times_plot if t > 0]) if all_round_times_plot else 1
            max_round_time = max(all_round_times_plot) if all_round_times_plot else 1
            if max_round_time > 0 and min_round_time > 0 and max_round_time / min_round_time > 100:
                axes[1, 1].set_yscale('log')
                # For log scale, set positive limits
                axes[1, 1].set_xlim(left=0, right=max(all_client_counts_plot) * 1.1 if all_client_counts_plot else 100)
                axes[1, 1].set_ylim(bottom=min_round_time * 0.9, top=max_round_time * 1.2)
            else:
                # Linear scale with proper limits
                axes[1, 1].set_xlim(left=0, right=max(all_client_counts_plot) * 1.1 if all_client_counts_plot else 100)
                axes[1, 1].set_ylim(bottom=0, top=max_round_time * 1.2 if max_round_time > 0 else 1)
        else:
            axes[1, 1].text(0.5, 0.5, 'No data available', ha='center', va='center', 
                           transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_xlim(0, 100)
            axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, output_file)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {output_file}")
    
    def create_summary_table(self, metrics_data: Dict[str, Any],
                           output_file: str = "summary_table.txt"):
        """Create comprehensive summary table with ALL metrics"""
        # Filter out RSA and EdDSA
        metrics_data = self._filter_metrics_data(metrics_data)
        
        try:
            from tabulate import tabulate
        except ImportError:
            tabulate = None
        
        algorithms = self._filter_algorithms(list(metrics_data.keys()))
        
        # Table 1: Performance Summary
        perf_table_data = []
        for algo in algorithms:
            perf = metrics_data[algo]
            comm = perf.get("communication", {})
            key_gen_data = perf.get("key_generation", {})
            signing_data = perf.get("signing", {})
            verify_data = perf.get("verification", {})
            agg_data = perf.get("aggregation", {})
            
            key_gen = key_gen_data.get("avg_time", 0)
            key_gen_std = key_gen_data.get("std_time", 0)
            
            # Get signing times for different message sizes
            sign_128 = signing_data.get(128, {}).get("avg_time", 0)
            sign_1k = signing_data.get(1024, {}).get("avg_time", 0)
            sign_4k = signing_data.get(4096, {}).get("avg_time", 0)
            sign_64k = signing_data.get(65536, {}).get("avg_time", 0)
            
            # Get verification times
            verify_128 = verify_data.get(128, {}).get("avg_time", 0)
            verify_1k = verify_data.get(1024, {}).get("avg_time", 0)
            verify_4k = verify_data.get(4096, {}).get("avg_time", 0)
            verify_64k = verify_data.get(65536, {}).get("avg_time", 0)
            
            # Aggregation times for different client counts
            agg_10 = agg_data.get(10, {}).get("avg_time", 0)
            agg_100 = agg_data.get(100, {}).get("avg_time", 0)
            agg_1000 = agg_data.get(1000, {}).get("avg_time", 0)
            
            sig_size = comm.get("signature_size", 0)
            pub_key_size = comm.get("public_key_size", 0)
            agg_sig_size = comm.get("aggregate_signature_size", 0)
            
            perf_table_data.append([
                algo,
                f"{key_gen:.6f}",
                f"{sign_128:.6f}",
                f"{sign_1k:.6f}",
                f"{sign_4k:.6f}",
                f"{sign_64k:.6f}",
                f"{verify_4k:.6f}",
                f"{agg_100:.6f}",
                sig_size,
                agg_sig_size,
                pub_key_size
            ])
        
        perf_headers = ["Algorithm", "Key Gen", "Sign 128B", "Sign 1KB", "Sign 4KB", 
                       "Sign 64KB", "Verify 4KB", "Agg 100", "Sig (B)", "Agg Sig (B)", "Pub Key (B)"]
        
        if tabulate:
            perf_table = tabulate(perf_table_data, headers=perf_headers, tablefmt="grid", floatfmt=".6f")
        else:
            perf_table_lines = [" | ".join(perf_headers)]
            perf_table_lines.append("-" * 120)
            for row in perf_table_data:
                perf_table_lines.append(" | ".join([str(cell) for cell in row]))
            perf_table = "\n".join(perf_table_lines)
        
        # Table 2: Communication Summary
        comm_table_data = []
        for algo in algorithms:
            perf = metrics_data[algo]
            comm = perf.get("communication", {})
            comm_table_data.append([
                algo,
                comm.get("signature_size", 0),
                comm.get("public_key_size", 0),
                comm.get("private_key_size", 0),
                comm.get("aggregate_signature_size", 0),
                comm.get("nonce_size", 0),
                comm.get("message_commitment_size", 0),
                comm.get("metadata_per_round", 0),
                f"{comm.get('compression_ratio', 1.0):.2f}"
            ])
        
        comm_headers = ["Algorithm", "Sig (B)", "Pub Key (B)", "Priv Key (B)", 
                       "Agg Sig (B)", "Nonce (B)", "Commit (B)", "Metadata (B)", "Compression"]
        
        if tabulate:
            comm_table = tabulate(comm_table_data, headers=comm_headers, tablefmt="grid")
        else:
            comm_table_lines = [" | ".join(comm_headers)]
            comm_table_lines.append("-" * 100)
            for row in comm_table_data:
                comm_table_lines.append(" | ".join([str(cell) for cell in row]))
            comm_table = "\n".join(comm_table_lines)
        
        # Table 3: Security Summary
        sec_table_data = []
        for algo in algorithms:
            perf = metrics_data[algo]
            sec = perf.get("security", {})
            sec_table_data.append([
                algo,
                sec.get("security_notion", "N/A"),
                "Yes" if sec.get("public_verifiability", False) else "No",
                ", ".join(sec.get("homomorphic_operations", [])),
                sec.get("key_distribution", "N/A"),
                ", ".join(sec.get("limitations", [])) if sec.get("limitations") else "None"
            ])
        
        sec_headers = ["Algorithm", "Security Notion", "Public Verif", "Homomorphic Ops", 
                      "Key Dist", "Limitations"]
        
        if tabulate:
            sec_table = tabulate(sec_table_data, headers=sec_headers, tablefmt="grid")
        else:
            sec_table_lines = [" | ".join(sec_headers)]
            sec_table_lines.append("-" * 100)
            for row in sec_table_data:
                sec_table_lines.append(" | ".join([str(cell) for cell in row]))
            sec_table = "\n".join(sec_table_lines)
        
        # Write comprehensive table
        table_path = os.path.join(self.output_dir, output_file)
        with open(table_path, 'w') as f:
            f.write("=" * 120 + "\n")
            f.write("COMPREHENSIVE BENCHMARK SUMMARY TABLES\n")
            f.write("=" * 120 + "\n\n")
            
            f.write("TABLE 1: PERFORMANCE METRICS\n")
            f.write("-" * 120 + "\n")
            f.write(perf_table)
            f.write("\n\n")
            
            f.write("TABLE 2: COMMUNICATION METRICS\n")
            f.write("-" * 120 + "\n")
            f.write(comm_table)
            f.write("\n\n")
            
            f.write("TABLE 3: SECURITY PROPERTIES\n")
            f.write("-" * 120 + "\n")
            f.write(sec_table)
            f.write("\n\n")
            
            f.write("=" * 120 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 120 + "\n")
        
        # Print all tables to console
        print(f"\n  ✓ {output_file}")
        print("\n" + "=" * 120)
        print("PERFORMANCE METRICS")
        print("=" * 120)
        print(perf_table)
        print("\n" + "=" * 120)
        print("COMMUNICATION METRICS")
        print("=" * 120)
        print(comm_table)
        print("\n" + "=" * 120)
        print("SECURITY PROPERTIES")
        print("=" * 120)
        print(sec_table)
        print("=" * 120)
