"""
Metrics Collector - Comprehensive Data Collection and Export
Supports multiple export formats: JSON, CSV, Excel, Markdown, HTML, LaTeX
"""

import json
import numpy as np
import psutil
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Performance metrics for cryptographic operations"""
    operation: str
    algorithm: str
    key_gen_time: float = 0.0
    sign_time: float = 0.0
    verify_time: float = 0.0
    aggregate_time: float = 0.0
    aggregate_verify_time: float = 0.0
    encrypt_time: float = 0.0
    decrypt_time: float = 0.0
    total_time: float = 0.0
    memory_peak_mb: float = 0.0
    memory_avg_mb: float = 0.0
    message_size: int = 0
    num_clients: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CommunicationMetrics:
    """Communication overhead metrics"""
    algorithm: str
    signature_size: int = 0
    public_key_size: int = 0
    private_key_size: int = 0
    tag_size: int = 0
    ciphertext_size: int = 0
    total_message_size: int = 0
    aggregate_signature_size: int = 0
    compression_ratio: float = 1.0
    nonce_size: int = 0
    proof_of_possession_size: int = 0
    message_commitment_size: int = 0
    metadata_per_round: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SecurityMetrics:
    """Security properties and characteristics"""
    algorithm: str
    security_notion: str = ""
    public_verifiability: bool = False
    homomorphic_operations: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    key_distribution: str = ""
    post_quantum: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MetricsCollector:
    """Comprehensive metrics collection and export system"""
    
    def __init__(self):
        self.performance_metrics = []
        self.communication_metrics = []
        self.security_metrics = []
        self.process = psutil.Process(os.getpid())
        self.memory_samples = []
    
    def start_memory_monitoring(self):
        """Start monitoring memory usage"""
        self.memory_samples = []
    
    def sample_memory(self):
        """Sample current memory usage"""
        try:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(memory_mb)
        except Exception:
            pass
    
    def stop_memory_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return statistics"""
        if not self.memory_samples:
            return {"peak_mb": 0.0, "avg_mb": 0.0}
        return {
            "peak_mb": max(self.memory_samples),
            "avg_mb": np.mean(self.memory_samples)
        }
    
    def record_performance(self, algorithm: str, operation: str, **kwargs):
        """Record performance metrics"""
        metrics = PerformanceMetrics(
            operation=operation,
            algorithm=algorithm,
            **{k: v for k, v in kwargs.items() if hasattr(PerformanceMetrics, k)}
        )
        self.performance_metrics.append(metrics)
    
    def record_communication(self, algorithm: str, **kwargs):
        """Record communication metrics"""
        metrics = CommunicationMetrics(
            algorithm=algorithm,
            **{k: v for k, v in kwargs.items() if hasattr(CommunicationMetrics, k)}
        )
        self.communication_metrics.append(metrics)
    
    def record_security(self, algorithm: str, **kwargs):
        """Record security metrics"""
        metrics = SecurityMetrics(
            algorithm=algorithm,
            **{k: v for k, v in kwargs.items() if hasattr(SecurityMetrics, k)}
        )
        self.security_metrics.append(metrics)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics"""
        summary = {
            "performance": {},
            "communication": {},
            "security": {},
            "metadata": {
                "total_algorithms": 0,
                "total_operations": 0,
                "collection_timestamp": datetime.now().isoformat()
            }
        }
        
        if self.performance_metrics:
            algorithms = set(m.algorithm for m in self.performance_metrics)
            summary["metadata"]["total_algorithms"] = len(algorithms)
            summary["metadata"]["total_operations"] = len(self.performance_metrics)
            
            for algo in algorithms:
                algo_metrics = [m for m in self.performance_metrics if m.algorithm == algo]
                times = {
                    "key_gen": [m.key_gen_time for m in algo_metrics if m.key_gen_time > 0],
                    "sign": [m.sign_time for m in algo_metrics if m.sign_time > 0],
                    "verify": [m.verify_time for m in algo_metrics if m.verify_time > 0],
                    "aggregate": [m.aggregate_time for m in algo_metrics if m.aggregate_time > 0]
                }
                
                summary["performance"][algo] = {
                    "avg_key_gen_time": np.mean(times["key_gen"]) if times["key_gen"] else 0.0,
                    "min_key_gen_time": np.min(times["key_gen"]) if times["key_gen"] else 0.0,
                    "max_key_gen_time": np.max(times["key_gen"]) if times["key_gen"] else 0.0,
                    "avg_sign_time": np.mean(times["sign"]) if times["sign"] else 0.0,
                    "min_sign_time": np.min(times["sign"]) if times["sign"] else 0.0,
                    "max_sign_time": np.max(times["sign"]) if times["sign"] else 0.0,
                    "avg_verify_time": np.mean(times["verify"]) if times["verify"] else 0.0,
                    "min_verify_time": np.min(times["verify"]) if times["verify"] else 0.0,
                    "max_verify_time": np.max(times["verify"]) if times["verify"] else 0.0,
                    "avg_aggregate_time": np.mean(times["aggregate"]) if times["aggregate"] else 0.0,
                    "avg_memory_peak": np.mean([m.memory_peak_mb for m in algo_metrics if m.memory_peak_mb > 0]) if any(m.memory_peak_mb > 0 for m in algo_metrics) else 0.0,
                    "avg_memory_avg": np.mean([m.memory_avg_mb for m in algo_metrics if m.memory_avg_mb > 0]) if any(m.memory_avg_mb > 0 for m in algo_metrics) else 0.0,
                    "total_measurements": len(algo_metrics)
                }
        
        if self.communication_metrics:
            algorithms = set(m.algorithm for m in self.communication_metrics)
            for algo in algorithms:
                algo_metrics = [m for m in self.communication_metrics if m.algorithm == algo]
                summary["communication"][algo] = {
                    "avg_signature_size": np.mean([m.signature_size for m in algo_metrics if m.signature_size > 0]) if any(m.signature_size > 0 for m in algo_metrics) else 0,
                    "avg_public_key_size": np.mean([m.public_key_size for m in algo_metrics if m.public_key_size > 0]) if any(m.public_key_size > 0 for m in algo_metrics) else 0,
                    "avg_total_message_size": np.mean([m.total_message_size for m in algo_metrics if m.total_message_size > 0]) if any(m.total_message_size > 0 for m in algo_metrics) else 0,
                    "avg_compression_ratio": np.mean([m.compression_ratio for m in algo_metrics if m.compression_ratio > 0]) if any(m.compression_ratio > 0 for m in algo_metrics) else 1.0
                }
        
        if self.security_metrics:
            algorithms = set(m.algorithm for m in self.security_metrics)
            for algo in algorithms:
                algo_metrics = [m for m in self.security_metrics if m.algorithm == algo]
                if algo_metrics:
                    summary["security"][algo] = {
                        "security_notion": algo_metrics[0].security_notion,
                        "public_verifiability": algo_metrics[0].public_verifiability,
                        "post_quantum": algo_metrics[0].post_quantum,
                        "homomorphic_operations": algo_metrics[0].homomorphic_operations,
                        "limitations": algo_metrics[0].limitations
                    }
        
        return summary
    
    def export_to_json(self, filepath: str):
        """Export metrics to JSON file"""
        dir_path = os.path.dirname(filepath) if os.path.dirname(filepath) else '.'
        os.makedirs(dir_path, exist_ok=True)
        
        data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "version": "1.0",
                "total_performance_metrics": len(self.performance_metrics),
                "total_communication_metrics": len(self.communication_metrics),
                "total_security_metrics": len(self.security_metrics)
            },
            "performance": [asdict(m) for m in self.performance_metrics],
            "communication": [asdict(m) for m in self.communication_metrics],
            "security": [asdict(m) for m in self.security_metrics],
            "summary": self.get_summary_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✓ JSON metrics saved to {filepath}")
    
    def export_to_csv(self, filepath_prefix: str):
        """Export metrics to CSV files"""
        import pandas as pd
        
        dir_path = os.path.dirname(filepath_prefix) if os.path.dirname(filepath_prefix) else '.'
        os.makedirs(dir_path, exist_ok=True)
        
        if self.performance_metrics:
            df_perf = pd.DataFrame([asdict(m) for m in self.performance_metrics])
            perf_path = f"{filepath_prefix}_performance.csv"
            df_perf.to_csv(perf_path, index=False)
            print(f"✓ Performance CSV saved to {perf_path}")
        
        if self.communication_metrics:
            df_comm = pd.DataFrame([asdict(m) for m in self.communication_metrics])
            comm_path = f"{filepath_prefix}_communication.csv"
            df_comm.to_csv(comm_path, index=False)
            print(f"✓ Communication CSV saved to {comm_path}")
        
        if self.security_metrics:
            df_sec = pd.DataFrame([asdict(m) for m in self.security_metrics])
            sec_path = f"{filepath_prefix}_security.csv"
            df_sec.to_csv(sec_path, index=False)
            print(f"✓ Security CSV saved to {sec_path}")
    
    def export_to_excel(self, filepath: str):
        """Export metrics to Excel file with multiple sheets"""
        try:
            import pandas as pd
            
            dir_path = os.path.dirname(filepath) if os.path.dirname(filepath) else '.'
            os.makedirs(dir_path, exist_ok=True)
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                if self.performance_metrics:
                    df_perf = pd.DataFrame([asdict(m) for m in self.performance_metrics])
                    df_perf.to_excel(writer, sheet_name='Performance', index=False)
                
                if self.communication_metrics:
                    df_comm = pd.DataFrame([asdict(m) for m in self.communication_metrics])
                    df_comm.to_excel(writer, sheet_name='Communication', index=False)
                
                if self.security_metrics:
                    df_sec = pd.DataFrame([asdict(m) for m in self.security_metrics])
                    df_sec.to_excel(writer, sheet_name='Security', index=False)
                
                # Summary sheet
                summary = self.get_summary_statistics()
                summary_df = pd.DataFrame([{
                    "Category": "Performance",
                    "Algorithms": len(summary.get("performance", {})),
                    "Total Measurements": len(self.performance_metrics)
                }, {
                    "Category": "Communication",
                    "Algorithms": len(summary.get("communication", {})),
                    "Total Measurements": len(self.communication_metrics)
                }, {
                    "Category": "Security",
                    "Algorithms": len(summary.get("security", {})),
                    "Total Measurements": len(self.security_metrics)
                }])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            print(f"✓ Excel file saved to {filepath}")
        except ImportError:
            print("⚠ openpyxl not available, skipping Excel export")
    
    def export_to_markdown(self, filepath: str):
        """Export comprehensive Markdown report"""
        dir_path = os.path.dirname(filepath) if os.path.dirname(filepath) else '.'
        os.makedirs(dir_path, exist_ok=True)
        
        summary = self.get_summary_statistics()
        
        md_content = f"""# Homomorphic Authentication Benchmark Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Total Algorithms Tested:** {summary['metadata']['total_algorithms']}
- **Total Operations:** {summary['metadata']['total_operations']}
- **Performance Measurements:** {len(self.performance_metrics)}
- **Communication Measurements:** {len(self.communication_metrics)}
- **Security Assessments:** {len(self.security_metrics)}

---

## Performance Metrics

### By Algorithm

"""
        
        for algo, stats in summary.get("performance", {}).items():
            md_content += f"""### {algo}

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| Key Generation (s) | {stats['avg_key_gen_time']:.6f} | {stats['min_key_gen_time']:.6f} | {stats['max_key_gen_time']:.6f} |
| Signing (s) | {stats['avg_sign_time']:.6f} | {stats['min_sign_time']:.6f} | {stats['max_sign_time']:.6f} |
| Verification (s) | {stats['avg_verify_time']:.6f} | {stats['min_verify_time']:.6f} | {stats['max_verify_time']:.6f} |
| Aggregation (s) | {stats['avg_aggregate_time']:.6f} | - | - |
| Peak Memory (MB) | {stats['avg_memory_peak']:.2f} | - | - |
| Avg Memory (MB) | {stats['avg_memory_avg']:.2f} | - | - |
| Measurements | {stats['total_measurements']} | - | - |

"""
        
        md_content += "\n## Communication Metrics\n\n"
        md_content += "| Algorithm | Signature Size (B) | Public Key Size (B) | Total Message Size (B) | Compression Ratio |\n"
        md_content += "|-----------|-------------------|---------------------|----------------------|-------------------|\n"
        
        for algo, stats in summary.get("communication", {}).items():
            md_content += f"| {algo} | {stats['avg_signature_size']:.0f} | {stats['avg_public_key_size']:.0f} | {stats['avg_total_message_size']:.0f} | {stats['avg_compression_ratio']:.2f} |\n"
        
        md_content += "\n## Security Properties\n\n"
        md_content += "| Algorithm | Security Notion | Public Verifiable | Post-Quantum | Homomorphic Operations |\n"
        md_content += "|-----------|----------------|-------------------|--------------|----------------------|\n"
        
        for algo, stats in summary.get("security", {}).items():
            ops = ", ".join(stats.get("homomorphic_operations", []))
            md_content += f"| {algo} | {stats.get('security_notion', 'N/A')} | {'Yes' if stats.get('public_verifiability') else 'No'} | {'Yes' if stats.get('post_quantum') else 'No'} | {ops or 'N/A'} |\n"
        
        with open(filepath, 'w') as f:
            f.write(md_content)
        
        print(f"✓ Markdown report saved to {filepath}")
    
    def export_to_html(self, filepath: str):
        """Export interactive HTML report"""
        dir_path = os.path.dirname(filepath) if os.path.dirname(filepath) else '.'
        os.makedirs(dir_path, exist_ok=True)
        
        summary = self.get_summary_statistics()
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Homomorphic Authentication Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; font-weight: bold; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .summary {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ font-size: 14px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Homomorphic Authentication Benchmark Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="metric">
                <div class="metric-value">{summary['metadata']['total_algorithms']}</div>
                <div class="metric-label">Algorithms Tested</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['metadata']['total_operations']}</div>
                <div class="metric-label">Total Operations</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(self.performance_metrics)}</div>
                <div class="metric-label">Performance Measurements</div>
            </div>
        </div>
        
        <h2>Performance Metrics</h2>
        <table>
            <tr>
                <th>Algorithm</th>
                <th>Key Gen (s)</th>
                <th>Sign (s)</th>
                <th>Verify (s)</th>
                <th>Aggregate (s)</th>
                <th>Peak Memory (MB)</th>
            </tr>
"""
        
        for algo, stats in summary.get("performance", {}).items():
            html_content += f"""
            <tr>
                <td><strong>{algo}</strong></td>
                <td>{stats['avg_key_gen_time']:.6f}</td>
                <td>{stats['avg_sign_time']:.6f}</td>
                <td>{stats['avg_verify_time']:.6f}</td>
                <td>{stats['avg_aggregate_time']:.6f}</td>
                <td>{stats['avg_memory_peak']:.2f}</td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <h2>Communication Metrics</h2>
        <table>
            <tr>
                <th>Algorithm</th>
                <th>Signature Size (B)</th>
                <th>Public Key Size (B)</th>
                <th>Total Message Size (B)</th>
                <th>Compression Ratio</th>
            </tr>
"""
        
        for algo, stats in summary.get("communication", {}).items():
            html_content += f"""
            <tr>
                <td><strong>{algo}</strong></td>
                <td>{stats['avg_signature_size']:.0f}</td>
                <td>{stats['avg_public_key_size']:.0f}</td>
                <td>{stats['avg_total_message_size']:.0f}</td>
                <td>{stats['avg_compression_ratio']:.2f}</td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <h2>Security Properties</h2>
        <table>
            <tr>
                <th>Algorithm</th>
                <th>Security Notion</th>
                <th>Public Verifiable</th>
                <th>Post-Quantum</th>
                <th>Homomorphic Operations</th>
            </tr>
"""
        
        for algo, stats in summary.get("security", {}).items():
            ops = ", ".join(stats.get("homomorphic_operations", []))
            html_content += f"""
            <tr>
                <td><strong>{algo}</strong></td>
                <td>{stats.get('security_notion', 'N/A')}</td>
                <td>{'Yes' if stats.get('public_verifiability') else 'No'}</td>
                <td>{'Yes' if stats.get('post_quantum') else 'No'}</td>
                <td>{ops or 'N/A'}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
</body>
</html>
"""
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        print(f"✓ HTML report saved to {filepath}")
    
    def export_all_formats(self, output_dir: str, prefix: str = "metrics"):
        """Export to all available formats"""
        os.makedirs(output_dir, exist_ok=True)
        
        base_path = os.path.join(output_dir, prefix)
        
        print(f"Exporting to: {output_dir}")
        print(f"Base path: {base_path}")
        
        # Core formats
        try:
            self.export_to_json(f"{base_path}.json")
        except Exception as e:
            print(f"Warning: JSON export failed: {e}")
        
        try:
            self.export_to_csv(base_path)
        except Exception as e:
            print(f"Warning: CSV export failed: {e}")
        
        # Additional formats
        try:
            self.export_to_excel(f"{base_path}.xlsx")
        except Exception as e:
            print(f"Warning: Excel export failed: {e}")
        
        try:
            self.export_to_markdown(f"{base_path}.md")
        except Exception as e:
            print(f"Warning: Markdown export failed: {e}")
        
        try:
            self.export_to_html(f"{base_path}.html")
        except Exception as e:
            print(f"Warning: HTML export failed: {e}")
        
        print(f"\n✓ All export formats saved to {output_dir}/")
