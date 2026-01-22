#!/usr/bin/env python3
"""
Complete Benchmark Runner
Run benchmarks, generate plots, and export all results
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.benchmarking.benchmark_runner import BenchmarkRunner
from src.benchmarking.visualization import Visualizer

def main():
    print("=" * 80)
    print("Homomorphic Authentication Benchmark Suite")
    print("=" * 80)
    
    # Create results directory
    output_dir = "results/benchmark"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Absolute path: {os.path.abspath(output_dir)}")
    print("\nStarting benchmarks...\n")
    
    try:
        # Initialize runner
        runner = BenchmarkRunner(config_path="config/benchmark_config.yaml")
        
        # Run benchmarks
        results = runner.run_full_benchmark()
        
        # Save raw results JSON
        raw_path = os.path.join(output_dir, "raw_results.json")
        with open(raw_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Raw results saved to {raw_path}")
        
        # Export all metrics formats
        print("\nExporting metrics...")
        runner.export_results(output_dir=output_dir)
        
        # Generate visualizations
        print("\nGenerating plots...")
        visualizer = Visualizer(output_dir=output_dir)
        visualizer.plot_performance_comparison(results)
        visualizer.plot_scalability(results)
        visualizer.plot_communication_overhead(results)
        visualizer.plot_message_size_impact(results)
        visualizer.plot_fl_benchmarks(results)
        visualizer.create_summary_table(results)
        print("✓ All plots generated")
        
        # List all output files
        print("\n" + "=" * 80)
        print("✅ Benchmark Complete!")
        print("=" * 80)
        print(f"\nResults saved to: {os.path.abspath(output_dir)}")
        print("\nGenerated files:")
        
        expected_files = [
            "raw_results.json",
            "metrics.json",
            "metrics_performance.csv",
            "metrics_communication.csv",
            "metrics_security.csv",
            "metrics.xlsx",
            "metrics.md",
            "metrics.html",
            "summary_table.txt",
            "performance_comparison.png",
            "scalability.png",
            "communication_overhead.png",
            "message_size_impact.png"
        ]
        
        for filename in expected_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"  ✓ {filename} ({size:,} bytes)")
            else:
                print(f"  ✗ {filename} - NOT FOUND")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
