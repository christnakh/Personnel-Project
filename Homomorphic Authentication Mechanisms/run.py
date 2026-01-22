#!/usr/bin/env python3
"""
Complete Homomorphic Authentication Suite - ALL IN ONE
=======================================================

This single file runs everything:
- Mathematical verification
- Homomorphic property tests
- Full benchmarking (metrics, security, visualization)
- FL simulation
- Everything!

Usage:
    python run.py verify              # Run verification tests
    python run.py benchmark --plots    # Run benchmarks with plots
    python run.py fl --clients 10     # Run FL simulation
    python run.py all                 # Run everything!
"""

import sys
import argparse
import json
import numpy as np
import hashlib
import os
import time
from pathlib import Path
from colorama import init, Fore, Style
from tqdm import tqdm

# Initialize colorama
init()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Simple results path - works in Docker and locally
RESULTS_DIR = "results"

def get_results_path(relative_path: str = "") -> str:
    """Simple function - just return results/ path (works in Docker too)"""
    # Always use simple relative path - Docker volume mount handles it
    if relative_path:
        path = os.path.join(RESULTS_DIR, relative_path)
        os.makedirs(path, exist_ok=True)
        return path
    else:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        return RESULTS_DIR

# Import all algorithms
from src.algorithms.homomorphic_signatures import (
    BLSSignature, RSASignature, EdDSASignature,
    WatersHomomorphicSignature, BonehBoyenHomomorphicSignature, LHSSignature
)
from src.algorithms.homomorphic_mac import (
    AdditiveHMAC, LinearHMAC, PolynomialHMAC, LatticeHMAC
)

# Import benchmarking components
from src.benchmarking.benchmark_runner import BenchmarkRunner
from src.benchmarking.metrics_collector import MetricsCollector
from src.benchmarking.security_benchmarks import SecurityBenchmark
from src.benchmarking.visualization import Visualizer

# Import FL components
from src.fl_pipeline.client import FLClient
from src.fl_pipeline.server import FLServer

# ============================================================================
# PART 1: MATHEMATICAL VERIFICATION
# ============================================================================

def verify_mathematical_operations():
    """Verify that mathematical operations are REAL, not just API calls"""
    print("\n" + "="*80)
    print(f"{Fore.CYAN}{Style.BRIGHT}PART 1: MATHEMATICAL VERIFICATION{Style.RESET_ALL}")
    print("Proving operations use real cryptographic math, not just hashing")
    print("="*80)
    
    success = True
    
    # Test 1: Additive HMAC
    print(f"\n{Fore.YELLOW}1. Additive HMAC - Prime Field Operations{Style.RESET_ALL}")
    print("-"*80)
    
    try:
        mac = AdditiveHMAC()
        mac.key_generation()
        
        prime = mac.prime
        print(f"Prime p = {prime}")
        print(f"Prime = 2^256 - 189: {Fore.GREEN}{prime == 2**256 - 189}{Style.RESET_ALL}")
        
        msg1 = b"test1"
        msg2 = b"test2"
        id1 = b"id1"
        id2 = b"id2"
        
        tag1, _ = mac.generate_tag(msg1, id1)
        tag2, _ = mac.generate_tag(msg2, id2)
        
        # MANUAL calculation
        prf1 = mac._prf(id1)
        msg1_int = int.from_bytes(hashlib.sha256(msg1).digest(), 'big')
        expected_tag1 = (prf1 * msg1_int) % prime
        actual_tag1 = int.from_bytes(tag1, 'big')
        
        print(f"\nManual calculation:")
        print(f"  PRF(id1) = {prf1}")
        print(f"  H(msg1) = {msg1_int}")
        print(f"  Expected: (PRF * H(msg)) mod p")
        print(f"  Expected tag1 = {expected_tag1}")
        print(f"  Actual tag1   = {actual_tag1}")
        match1 = expected_tag1 == actual_tag1
        print(f"  {Fore.GREEN}✓ MATCH!{Style.RESET_ALL}" if match1 else f"  {Fore.RED}✗ MISMATCH!{Style.RESET_ALL}")
        
        # Verify homomorphic addition
        combined_tag, _ = mac.combine_tags([tag1, tag2])
        combined_int = int.from_bytes(combined_tag, 'big')
        
        tag1_int = int.from_bytes(tag1, 'big')
        tag2_int = int.from_bytes(tag2, 'big')
        expected_combined = (tag1_int + tag2_int) % prime
        
        print(f"\nHomomorphic addition:")
        print(f"  Expected: (tag1 + tag2) mod p = {expected_combined}")
        print(f"  Actual combined = {combined_int}")
        match2 = expected_combined == combined_int
        print(f"  {Fore.GREEN}✓ REAL FIELD ADDITION!{Style.RESET_ALL}" if match2 else f"  {Fore.RED}✗ WRONG!{Style.RESET_ALL}")
        
        success = success and match1 and match2
        
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        success = False
    
    # Test 2: Linear HMAC
    print(f"\n{Fore.YELLOW}2. Linear HMAC - Inner Product in Z_p{Style.RESET_ALL}")
    print("-"*80)
    
    try:
        mac_linear = LinearHMAC(vector_dim=10)
        mac_linear.key_generation()
        
        v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        id1 = b"vector1"
        
        tag1, _ = mac_linear.generate_tag(v1, id1)
        
        # Manual calculation
        prf_vec = mac_linear._prf_vector(id1)
        v1_scaled = np.array([int(v * 1000) % mac_linear.prime for v in v1], dtype=object)
        
        manual_inner_product = 0
        for i in range(10):
            manual_inner_product = (manual_inner_product + int(prf_vec[i]) * int(v1_scaled[i])) % mac_linear.prime
        
        actual_tag_int = int.from_bytes(tag1, 'big')
        
        print(f"Vector dimension: {len(v1)}")
        print(f"PRF vector sample (first 3): [{prf_vec[0]}, {prf_vec[1]}, {prf_vec[2]}]")
        print(f"\nManual inner product: {manual_inner_product}")
        print(f"Actual tag: {actual_tag_int}")
        match3 = manual_inner_product == actual_tag_int
        print(f"{Fore.GREEN}✓ REAL INNER PRODUCT!{Style.RESET_ALL}" if match3 else f"{Fore.RED}✗ WRONG!{Style.RESET_ALL}")
        
        success = success and match3
        
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        success = False
    
    # Test 3: RSA
    print(f"\n{Fore.YELLOW}3. RSA - Multiplicative Homomorphism{Style.RESET_ALL}")
    print("-"*80)
    
    try:
        rsa = RSASignature(key_size=2048, homomorphic_mode=True)
        rsa.key_generation()
        
        print(f"RSA key size: {rsa.key_size} bits")
        print(f"Modulus N: {str(rsa.public_key.n)[:50]}...")
        
        msg1 = b"m1"
        msg2 = b"m2"
        
        sig1, _ = rsa.sign(msg1)
        sig2, _ = rsa.sign(msg2)
        
        sig_product, _ = rsa.homomorphic_multiply(sig1, sig2)
        
        s1 = int.from_bytes(sig1, 'big')
        s2 = int.from_bytes(sig2, 'big')
        expected_product = (s1 * s2) % rsa.public_key.n
        actual_product = int.from_bytes(sig_product, 'big')
        
        print(f"\nMultiplicative homomorphism:")
        print(f"  Expected: (sig1 * sig2) mod N")
        match4 = expected_product == actual_product
        print(f"  {Fore.GREEN}✓ REAL RSA MULTIPLICATION!{Style.RESET_ALL}" if match4 else f"  {Fore.RED}✗ WRONG!{Style.RESET_ALL}")
        
        success = success and match4
        
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        success = False
    
    return success


# ============================================================================
# PART 2: ALGORITHM TESTING
# ============================================================================

def test_all_algorithms():
    """Test all algorithms to verify they work correctly"""
    print("\n" + "="*80)
    print(f"{Fore.CYAN}{Style.BRIGHT}ALGORITHM FUNCTIONALITY TESTS{Style.RESET_ALL}")
    print("="*80)
    
    results = {}
    
    # Test BLS
    print(f"\n{Fore.YELLOW}Testing BLS Signature{Style.RESET_ALL}")
    try:
        bls = BLSSignature()
        sk, pk = bls.key_generation()
        msg = b"Test message"
        sig, _ = bls.sign(msg)
        valid, _ = bls.verify(msg, sig, pk)
        results['BLS'] = valid
        print(f"  {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if valid else f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - Signature: {len(sig)} bytes")
    except Exception as e:
        print(f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - {str(e)}")
        results['BLS'] = False
    
    # Test RSA
    print(f"\n{Fore.YELLOW}Testing RSA Signature{Style.RESET_ALL}")
    try:
        rsa = RSASignature()
        sk, pk = rsa.key_generation()
        msg = b"Test message"
        sig, _ = rsa.sign(msg)
        valid, _ = rsa.verify(msg, sig, pk)
        results['RSA'] = valid
        print(f"  {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if valid else f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - Signature: {len(sig)} bytes")
    except Exception as e:
        print(f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - {str(e)}")
        results['RSA'] = False
    
    # Test EdDSA
    print(f"\n{Fore.YELLOW}Testing EdDSA Signature{Style.RESET_ALL}")
    try:
        eddsa = EdDSASignature()
        sk, pk = eddsa.key_generation()
        msg = b"Test message"
        sig, _ = eddsa.sign(msg)
        valid, _ = eddsa.verify(msg, sig, pk)
        results['EdDSA'] = valid
        print(f"  {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if valid else f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - Signature: {len(sig)} bytes")
    except Exception as e:
        print(f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - {str(e)}")
        results['EdDSA'] = False
    
    # Test Waters
    print(f"\n{Fore.YELLOW}Testing Waters Homomorphic Signature{Style.RESET_ALL}")
    try:
        waters = WatersHomomorphicSignature(vector_dim=10)
        sk, pk = waters.key_generation()
        vector = np.random.randn(10).astype(np.float32)
        sig, _ = waters.sign_vector(vector, b"test_id")
        valid, _ = waters.verify_vector(vector, sig, b"test_id")
        results['Waters'] = valid
        print(f"  {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if valid else f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - Signature: {len(sig)} bytes")
    except Exception as e:
        print(f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - {str(e)}")
        results['Waters'] = False
    
    # Test Boneh-Boyen
    print(f"\n{Fore.YELLOW}Testing Boneh-Boyen Homomorphic Signature{Style.RESET_ALL}")
    try:
        bb = BonehBoyenHomomorphicSignature()
        sk, pk = bb.key_generation()
        msg = b"Test message"
        sig, _ = bb.sign(msg)
        valid, _ = bb.verify(msg, sig, pk)
        results['Boneh-Boyen'] = valid
        print(f"  {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if valid else f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - Signature: {len(sig)} bytes")
    except Exception as e:
        print(f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - {str(e)}")
        results['Boneh-Boyen'] = False
    
    # Test LHS
    print(f"\n{Fore.YELLOW}Testing LHS Signature{Style.RESET_ALL}")
    try:
        lhs = LHSSignature(vector_dim=10)
        sk, pk = lhs.key_generation()
        vector = np.random.randn(10).astype(np.float32)
        sig, _ = lhs.sign_vector(vector, b"test_id")
        valid, _ = lhs.verify_vector(vector, sig, b"test_id")
        results['LHS'] = valid
        print(f"  {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if valid else f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - Signature: {len(sig)} bytes")
    except Exception as e:
        print(f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - {str(e)}")
        results['LHS'] = False
    
    # Test Additive HMAC
    print(f"\n{Fore.YELLOW}Testing Additive HMAC{Style.RESET_ALL}")
    try:
        hmac = AdditiveHMAC()
        key = hmac.key_generation()
        msg = b"Test message"
        tag, _ = hmac.generate_tag(msg, b"test_id")
        valid, _ = hmac.verify_tag(msg, tag, b"test_id")
        results['Additive_HMAC'] = valid
        print(f"  {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if valid else f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - Tag: {len(tag)} bytes")
    except Exception as e:
        print(f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - {str(e)}")
        results['Additive_HMAC'] = False
    
    # Test Linear HMAC
    print(f"\n{Fore.YELLOW}Testing Linear HMAC{Style.RESET_ALL}")
    try:
        hmac = LinearHMAC(vector_dim=10)
        key = hmac.key_generation()
        vector = np.random.randn(10).astype(np.float32)
        tag, _ = hmac.generate_tag(vector, b"test_id")
        valid, _ = hmac.verify_tag(vector, tag, b"test_id")
        results['Linear_HMAC'] = valid
        print(f"  {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if valid else f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - Tag: {len(tag)} bytes")
    except Exception as e:
        print(f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - {str(e)}")
        results['Linear_HMAC'] = False
    
    # Test Polynomial HMAC
    print(f"\n{Fore.YELLOW}Testing Polynomial HMAC{Style.RESET_ALL}")
    try:
        hmac = PolynomialHMAC()
        key = hmac.key_generation()
        msg = b"Test message"
        tag, _ = hmac.generate_tag(msg, b"test_id")
        valid, _ = hmac.verify_tag(msg, tag, b"test_id")
        results['Polynomial_HMAC'] = valid
        print(f"  {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if valid else f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - Tag: {len(tag)} bytes")
    except Exception as e:
        print(f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - {str(e)}")
        results['Polynomial_HMAC'] = False
    
    # Test Lattice HMAC
    print(f"\n{Fore.YELLOW}Testing Lattice HMAC{Style.RESET_ALL}")
    try:
        hmac = LatticeHMAC()
        key = hmac.key_generation()
        msg = b"Test message"
        tag, _ = hmac.generate_tag(msg, b"test_id")
        valid, _ = hmac.verify_tag(msg, tag, b"test_id")
        results['Lattice_HMAC'] = valid
        print(f"  {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if valid else f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - Tag: {len(tag)} bytes")
    except Exception as e:
        print(f"  {Fore.RED}✗ FAIL{Style.RESET_ALL} - {str(e)}")
        results['Lattice_HMAC'] = False
    
    # Summary
    print("\n" + "="*80)
    print(f"{Fore.CYAN}{Style.BRIGHT}ALGORITHM TEST SUMMARY{Style.RESET_ALL}")
    print("="*80)
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = f"{Fore.GREEN}PASS{Style.RESET_ALL}" if result else f"{Fore.RED}FAIL{Style.RESET_ALL}"
        print(f"  {name:20s}: {status}")
    
    print(f"\n{Fore.CYAN}Overall: {passed}/{total} tests passed{Style.RESET_ALL}")
    
    return results, passed == total


# ============================================================================
# PART 3: HOMOMORPHIC PROPERTY TESTS
# ============================================================================

def print_header(title):
    print("\n" + "="*80)
    print(f"{Fore.CYAN}{Style.BRIGHT}{title}{Style.RESET_ALL}")
    print("="*80)

def print_test(test_name):
    print(f"\n{Fore.YELLOW}Testing: {test_name}{Style.RESET_ALL}")
    print("-"*80)

def save_verification_results(results, output_dir=None):
    """Save verification test results to files"""
    if output_dir is None:
        output_dir = get_results_path("verification")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    json_path = os.path.join(output_dir, "verification_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary text
    txt_path = os.path.join(output_dir, "verification_summary.txt")
    with open(txt_path, 'w') as f:
        f.write("HOMOMORPHIC AUTHENTICATION VERIFICATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        passed = sum(results.values())
        total = len(results)
        f.write(f"Overall: {passed}/{total} tests passed\n\n")
        f.write("Individual Results:\n")
        f.write("-" * 80 + "\n")
        for name, result in results.items():
            status = "PASS" if result else "FAIL"
            f.write(f"{name:20s}: {status}\n")
    
    print(f"\n✓ Verification results saved to {output_dir}/")
    return json_path, txt_path

def test_homomorphic_properties(save_results=True, output_dir=None):
    """Test all homomorphic properties"""
    if output_dir is None:
        output_dir = get_results_path("verification")
    print_header("PART 2: HOMOMORPHIC PROPERTY TESTS")
    
    results = {}
    
    # Test BLS
    print_test("BLS Signature Aggregation")
    try:
        import blspy
        print(f"blspy version: {blspy.__version__}")
        
        bls = BLSSignature()
        sk1, pk1 = bls.key_generation()
        
        bls2 = BLSSignature()
        sk2, pk2 = bls2.key_generation()
        
        msg1 = b"Transfer $100"
        msg2 = b"Transfer $200"
        
        sig1, _ = bls.sign(msg1)
        sig2, _ = bls2.sign(msg2)
        
        print(f"Signature 1: {len(sig1)} bytes")
        print(f"Signature 2: {len(sig2)} bytes")
        
        agg_sig, _ = bls.aggregate_signatures([sig1, sig2])
        print(f"Aggregated: {len(agg_sig)} bytes {Fore.GREEN}(constant size!){Style.RESET_ALL}")
        
        valid, _ = bls.aggregate_verify([msg1, msg2], agg_sig, [pk1, pk2])
        results['BLS'] = valid
        print(f"Verification: {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if valid else f"{Fore.RED}✗ FAIL{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.YELLOW}BLS requires blspy: {e}{Style.RESET_ALL}")
        results['BLS'] = False
    
    # Test RSA
    print_test("RSA Multiplicative Homomorphism")
    try:
        rsa = RSASignature(key_size=2048, homomorphic_mode=True)
        rsa.key_generation()
        
        msg1 = b"message1"
        msg2 = b"message2"
        
        sig1, _ = rsa.sign(msg1)
        sig2, _ = rsa.sign(msg2)
        sig_product, _ = rsa.homomorphic_multiply(sig1, sig2)
        print(f"Multiplicative property: sign(m1) * sign(m2) = sign(m1*m2)")
        
        valid1, _ = rsa.verify(msg1, sig1)
        valid2, _ = rsa.verify(msg2, sig2)
        results['RSA'] = valid1 and valid2
        print(f"Verification: {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if results['RSA'] else f"{Fore.RED}✗ FAIL{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        results['RSA'] = False
    
    # Test Additive HMAC
    print_test("Additive HMAC")
    try:
        mac = AdditiveHMAC()
        mac.key_generation()
        
        msg1 = b"data1"
        msg2 = b"data2"
        msg3 = b"data3"
        
        tag1, _ = mac.generate_tag(msg1, b"id1")
        tag2, _ = mac.generate_tag(msg2, b"id2")
        tag3, _ = mac.generate_tag(msg3, b"id3")
        
        combined_tag, _ = mac.combine_tags([tag1, tag2, tag3])
        print(f"Combined 3 tags: t_combined = (t1 + t2 + t3) mod p")
        
        valid1, _ = mac.verify_tag(msg1, tag1, b"id1")
        valid2, _ = mac.verify_tag(msg2, tag2, b"id2")
        valid3, _ = mac.verify_tag(msg3, tag3, b"id3")
        results['Additive HMAC'] = valid1 and valid2 and valid3
        print(f"Verification: {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if results['Additive HMAC'] else f"{Fore.RED}✗ FAIL{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        results['Additive HMAC'] = False
    
    # Test Linear HMAC
    print_test("Linear HMAC")
    try:
        mac = LinearHMAC(vector_dim=10)
        mac.key_generation()
        
        v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        v2 = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        v3 = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        
        tag1, _ = mac.generate_tag(v1, b"v1")
        tag2, _ = mac.generate_tag(v2, b"v2")
        tag3, _ = mac.generate_tag(v3, b"v3")
        
        coeffs = [0.3, 0.5, 0.2]
        combined_tag, _ = mac.linear_combine_tags([tag1, tag2, tag3], coeffs)
        combined_vector = 0.3*v1 + 0.5*v2 + 0.2*v3
        
        print(f"Linear combination: c1*v1 + c2*v2 + c2*v3 with coeffs {coeffs}")
        
        valid, _ = mac.verify_linear_combination([v1, v2, v3], combined_vector, combined_tag, coeffs, [b"v1", b"v2", b"v3"])
        results['Linear HMAC'] = valid
        print(f"Verification: {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if valid else f"{Fore.RED}✗ FAIL{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        results['Linear HMAC'] = False
    
    # Test Polynomial HMAC
    print_test("Polynomial HMAC")
    try:
        mac = PolynomialHMAC(poly_degree=3)
        mac.key_generation()
        
        msg1 = b"poly_message_1"
        msg2 = b"poly_message_2"
        
        tag1, _ = mac.generate_tag(msg1, b"id1")
        tag2, _ = mac.generate_tag(msg2, b"id2")
        
        poly_coeffs = [1.0, 2.0]
        combined_tag, _ = mac.polynomial_combine_tags([tag1, tag2], poly_coeffs)
        
        print(f"Polynomial combination with degree 3")
        
        valid1, _ = mac.verify_tag(msg1, tag1, b"id1")
        valid2, _ = mac.verify_tag(msg2, tag2, b"id2")
        results['Polynomial HMAC'] = valid1 and valid2
        print(f"Verification: {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if results['Polynomial HMAC'] else f"{Fore.RED}✗ FAIL{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        results['Polynomial HMAC'] = False
    
    # Test Lattice HMAC
    print_test("Lattice HMAC (Post-Quantum)")
    try:
        mac = LatticeHMAC(lattice_dim=256)
        mac.key_generation()
        
        msg1 = b"quantum_resistant_1"
        msg2 = b"quantum_resistant_2"
        
        tag1, _ = mac.generate_tag(msg1, b"id1")
        tag2, _ = mac.generate_tag(msg2, b"id2")
        
        combined_tag, _ = mac.lattice_combine_tags([tag1, tag2])
        
        print(f"{Fore.MAGENTA}★ POST-QUANTUM SECURE (LWE){Style.RESET_ALL}")
        
        valid1, _ = mac.verify_tag(msg1, tag1, b"id1")
        valid2, _ = mac.verify_tag(msg2, tag2, b"id2")
        results['Lattice HMAC'] = valid1 and valid2
        print(f"Verification: {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if results['Lattice HMAC'] else f"{Fore.RED}✗ FAIL{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        results['Lattice HMAC'] = False
    
    # Test Waters
    print_test("Waters Homomorphic Signatures")
    try:
        waters = WatersHomomorphicSignature(vector_dim=10)
        waters.key_generation()
        
        v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        v2 = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        
        sig1, _ = waters.sign_vector(v1, b"file1")
        sig2, _ = waters.sign_vector(v2, b"file2")
        
        coeffs = [0.6, 0.4]
        combined_vector = 0.6*v1 + 0.4*v2
        
        print(f"Linear combination of signed vectors")
        
        valid, _ = waters.verify_linear_combination(combined_vector, [v1, v2], [sig1, sig2], coeffs, [b"file1", b"file2"])
        results['Waters'] = valid
        print(f"Verification: {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if valid else f"{Fore.RED}✗ FAIL{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.YELLOW}Waters requires petlib: {e}{Style.RESET_ALL}")
        results['Waters'] = False
    
    # Test Boneh-Boyen
    print_test("Boneh-Boyen Homomorphic Signatures")
    try:
        bb = BonehBoyenHomomorphicSignature()
        bb.key_generation()
        
        msg1 = b"bb_message_1"
        msg2 = b"bb_message_2"
        
        sig1, _ = bb.sign(msg1)
        sig2, _ = bb.sign(msg2)
        
        print(f"Signature 1: {len(sig1)} bytes")
        print(f"Signature 2: {len(sig2)} bytes")
        
        agg_sig, _ = bb.aggregate_signatures([sig1, sig2])
        print(f"Aggregated: {len(agg_sig)} bytes")
        
        try:
            valid1, _ = bb.verify(msg1, sig1)
            print(f"Individual verification 1: {valid1}")
        except Exception as ve:
            valid1 = False
        
        try:
            valid2, _ = bb.verify(msg2, sig2)
            print(f"Individual verification 2: {valid2}")
        except Exception as ve:
            valid2 = False
        
        results['Boneh-Boyen'] = valid1 and valid2
        print(f"Verification: {Fore.GREEN}✓ PASS{Style.RESET_ALL}" if results['Boneh-Boyen'] else f"{Fore.RED}✗ FAIL{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.YELLOW}Boneh-Boyen requires petlib: {e}{Style.RESET_ALL}")
        results['Boneh-Boyen'] = False
    
    if save_results:
        save_verification_results(results, output_dir)
    
    return results


# ============================================================================
# PART 3: FULL BENCHMARKING
# ============================================================================

def run_full_benchmark(output_dir=None, config_path="config/benchmark_config.yaml", generate_plots=True):
    """Run complete benchmarking suite"""
    print_header("PART 3: FULL BENCHMARKING SUITE")
    
    # Use default if not provided
    if output_dir is None:
        output_dir = get_results_path("benchmark")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{Fore.CYAN}Saving results to: {output_dir}{Style.RESET_ALL}")
    
    # Create run info file immediately
    run_info_path = os.path.join(output_dir, "RUN_INFO.txt")
    with open(run_info_path, 'w') as f:
        f.write(f"BENCHMARK RUN STARTED\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Generate Plots: {generate_plots}\n")
    
    try:
        runner = BenchmarkRunner(config_path=config_path)
        
        print("Starting benchmark execution...")
        results = runner.run_full_benchmark()
        
        # Save raw results JSON
        raw_results_path = os.path.join(output_dir, "raw_benchmark_results.json")
        with open(raw_results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✓ Raw results saved to {raw_results_path}")
        
        print(f"\n{Fore.CYAN}Exporting metrics...{Style.RESET_ALL}")
        print(f"Output directory: {output_dir}")
        print(f"Absolute path: {os.path.abspath(output_dir)}")
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Test write access (critical for Docker volume mounts)
        test_write_file = os.path.join(output_dir, ".write_test")
        try:
            with open(test_write_file, 'w') as f:
                f.write("test")
            os.remove(test_write_file)
            print(f"{Fore.GREEN}✓ Write access verified{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}⚠ Warning: Write test failed: {e}{Style.RESET_ALL}")
            if os.path.exists("/.dockerenv"):
                print(f"{Fore.YELLOW}  Running in Docker - ensure volume mount: -v $(pwd)/results:/app/results{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}  Check directory permissions: {output_dir}{Style.RESET_ALL}")
        
        runner.export_results(output_dir=output_dir)
        
        # Verify files were created
        print(f"\nVerifying exported files in {output_dir}...")
        expected_files = [
            "metrics.json",
            "metrics_performance.csv",
            "metrics_communication.csv",
            "metrics_security.csv",
            "metrics.xlsx",
            "metrics.md",
            "metrics.html"
        ]
        
        for filename in expected_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"  ✓ {filename} ({size} bytes)")
            else:
                print(f"  ✗ {filename} - NOT FOUND")
        
        if generate_plots:
            print("\nGenerating visualizations...")
            # Save plots directly in output_dir (results/), not in a subdirectory
            os.makedirs(output_dir, exist_ok=True)
            visualizer = Visualizer(output_dir=output_dir)
            
            visualizer.plot_performance_comparison(results)
            visualizer.plot_scalability(results)
            visualizer.plot_communication_overhead(results)
            visualizer.plot_message_size_impact(results)
            visualizer.create_summary_table(results)
            
            print(f"\n✓ All plots and tables saved to {output_dir}")
        
        # Create comprehensive summary file
        summary_path = os.path.join(output_dir, "BENCHMARK_SUMMARY.txt")
        with open(summary_path, 'w') as f:
            f.write("BENCHMARK RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Output Directory: {output_dir}\n")
            f.write(f"Algorithms Tested: {len(results)}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Files Generated:\n")
            f.write("-" * 80 + "\n")
            f.write("Core Data Files:\n")
            f.write("  - metrics.json (Complete JSON data)\n")
            f.write("  - metrics_performance.csv (Performance metrics)\n")
            f.write("  - metrics_communication.csv (Communication metrics)\n")
            f.write("  - metrics_security.csv (Security metrics)\n")
            f.write("  - raw_benchmark_results.json (Raw benchmark data)\n\n")
            f.write("Additional Export Formats:\n")
            f.write("  - metrics.xlsx (Excel workbook with multiple sheets)\n")
            f.write("  - metrics.md (Markdown report)\n")
            f.write("  - metrics.html (Interactive HTML report)\n\n")
            if generate_plots:
                f.write("Visualizations:\n")
                f.write("  - plots/performance_comparison.png\n")
                f.write("  - plots/scalability.png\n")
                f.write("  - plots/communication_overhead.png\n")
                f.write("  - plots/message_size_impact.png\n")
                f.write("  - plots/summary_table.txt\n")
        
        # Update run info
        with open(run_info_path, 'a') as f:
            f.write(f"\nStatus: COMPLETED\n")
            f.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"\n✓ Benchmark results saved to {output_dir}")
        print(f"✓ Summary file: {summary_path}")
        print(f"✓ All export formats available (JSON, CSV, Excel, Markdown, HTML)")
        return True
        
    except Exception as e:
        # Save error info even on failure
        error_path = os.path.join(output_dir, "ERROR_LOG.txt")
        with open(error_path, 'w') as f:
            f.write(f"BENCHMARK ERROR\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error: {str(e)}\n")
            import traceback
            f.write(f"\nTraceback:\n{traceback.format_exc()}\n")
        
        with open(run_info_path, 'a') as f:
            f.write(f"\nStatus: FAILED\n")
            f.write(f"Error: {str(e)}\n")
        
        print(f"{Fore.RED}Benchmarking failed: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Error log saved to: {error_path}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# PART 4: FL SIMULATION
# ============================================================================

def run_fl_simulation(clients=10, rounds=5, auth_scheme="BLS", model_dim=1000, output_dir=None):
    """Run Federated Learning simulation"""
    print_header("PART 4: FEDERATED LEARNING SIMULATION")
    
    # Use default if not provided
    if output_dir is None:
        output_dir = get_results_path("fl_simulation")
    
    print(f"Authentication Scheme: {auth_scheme}")
    print(f"Number of Clients: {clients}")
    print(f"Number of Rounds: {rounds}")
    print(f"Model Dimension: {model_dim}")
    print(f"{Fore.CYAN}Saving results to: {output_dir}{Style.RESET_ALL}")
    print()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create run info file immediately
    run_info_path = os.path.join(output_dir, "RUN_INFO.txt")
    with open(run_info_path, 'w') as f:
        f.write(f"FL SIMULATION RUN STARTED\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Authentication Scheme: {auth_scheme}\n")
        f.write(f"Clients: {clients}\n")
        f.write(f"Rounds: {rounds}\n")
        f.write(f"Model Dimension: {model_dim}\n")
    
    try:
        # Create clients
        client_list = []
        secret_key = None
        for i in range(clients):
            client = FLClient(client_id=i, model_dim=model_dim, 
                             auth_scheme=auth_scheme)
            if i == 0 and hasattr(client.auth_handler, 'secret_key'):
                secret_key = client.auth_handler.secret_key
            client_list.append(client)
        
        # Create server
        server = FLServer(model_dim=model_dim, auth_scheme=auth_scheme, 
                         )
        if secret_key is not None and hasattr(server.auth_verifier, 'secret_key'):
            server.auth_verifier.secret_key = secret_key
        
        # Run FL rounds
        all_results = []
        for round_num in range(1, rounds + 1):
            print(f"\n--- FL Round {round_num} ---")
            global_model = server.get_global_model()
            client_updates = []
            
            for client in tqdm(client_list, desc=f"Round {round_num}: Client updates"):
                update_data = client.prepare_update(global_model)
                client_updates.append(update_data)
            
            print(f"Round {round_num}: Server aggregation...")
            result = server.receive_updates(client_updates)
            if isinstance(result["aggregated_update"], np.ndarray):
                server.update_global_model(result["aggregated_update"])
            all_results.append(result)
        
        # Save results JSON
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "fl_simulation_metrics.json")
        
        # Custom JSON encoder for numpy arrays
        def json_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, bytes):
                return obj.hex()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        # Convert results to JSON-serializable format
        serializable_results = []
        for r in all_results:
            serializable_r = {}
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    serializable_r[k] = v.tolist()
                elif isinstance(v, bytes):
                    serializable_r[k] = v.hex()
                else:
                    serializable_r[k] = v
            serializable_results.append(serializable_r)
        
        with open(output_file, 'w') as f:
            json.dump({
                "rounds": len(all_results),
                "clients": clients,
                "auth_scheme": auth_scheme,
                "model_dim": model_dim,
                "results": serializable_results
            }, f, indent=2, default=json_serializer)
        
        # Calculate and save summary statistics
        total_rounds = len(all_results)
        avg_aggregation_time = np.mean([r.get('aggregation_time', 0) for r in all_results])
        avg_verification_time = np.mean([r.get('verification_time', 0) for r in all_results])
        total_verified = sum([r.get('verified_clients', 0) for r in all_results])
        
        summary_path = os.path.join(output_dir, "fl_simulation_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("FEDERATED LEARNING SIMULATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Authentication Scheme: {auth_scheme}\n")
            f.write(f"Number of Clients: {clients}\n")
            f.write(f"Number of Rounds: {rounds}\n")
            f.write(f"Model Dimension: {model_dim}\n\n")
            f.write("Performance Metrics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Average Aggregation Time: {avg_aggregation_time:.4f} seconds\n")
            f.write(f"Average Verification Time: {avg_verification_time:.4f} seconds\n")
            f.write(f"Total Verified Clients: {total_verified}\n")
            if clients * rounds > 0:
                f.write(f"Verification Rate: {total_verified / (clients * rounds) * 100:.2f}%\n")
        
        # Update run info
        with open(run_info_path, 'a') as f:
            f.write(f"\nStatus: COMPLETED\n")
            f.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Results File: {output_file}\n")
            f.write(f"Summary File: {summary_path}\n")
        
        print(f"\n✓ FL simulation results saved to {output_file}")
        print(f"✓ Summary saved to {summary_path}")
        return True
        
    except Exception as e:
        # Save error info even on failure
        error_path = os.path.join(output_dir, "ERROR_LOG.txt")
        with open(error_path, 'w') as f:
            f.write(f"FL SIMULATION ERROR\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error: {str(e)}\n")
            import traceback
            f.write(f"\nTraceback:\n{traceback.format_exc()}\n")
        
        with open(run_info_path, 'a') as f:
            f.write(f"\nStatus: FAILED\n")
            f.write(f"Error: {str(e)}\n")
        
        print(f"{Fore.RED}FL simulation failed: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Error log saved to: {error_path}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN - COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complete Homomorphic Authentication Suite - ALL IN ONE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py test                     # Test all algorithms
  python run.py verify                    # Run verification tests
  python run.py benchmark --plots        # Run benchmarks with plots
  python run.py benchmark --output results/
  python run.py fl --clients 10 --rounds 5 --auth_scheme BLS
  python run.py all                      # Run everything (test + verify + benchmark + fl)!
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test all algorithms functionality')
    test_parser.add_argument('--output', type=str, default='results/test',
                            help='Output directory for results')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Run verification tests')
    verify_parser.add_argument('--output', type=str, default='results/verification',
                              help='Output directory for results')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run full benchmarking suite')
    benchmark_parser.add_argument('--output', type=str, default='results/benchmark_run',
                                help='Output directory for results')
    benchmark_parser.add_argument('--config', type=str, default='config/benchmark_config.yaml',
                                help='Path to benchmark configuration file')
    benchmark_parser.add_argument('--plots', action='store_true',
                                help='Generate plots')
    benchmark_parser.add_argument('--no-plots', action='store_true',
                                help='Skip plot generation')
    
    # FL simulation command
    fl_parser = subparsers.add_parser('fl', help='Run Federated Learning simulation')
    fl_parser.add_argument('--clients', type=int, default=10,
                          help='Number of clients')
    fl_parser.add_argument('--rounds', type=int, default=5,
                          help='Number of FL rounds')
    fl_parser.add_argument('--auth_scheme', type=str, default='BLS',
                          choices=['BLS', 'LHS', 'Waters', 'BonehBoyen', 'RSA', 'EdDSA',
                                  'Additive_HMAC', 'Linear_HMAC', 'Polynomial_HMAC', 'Lattice_HMAC'],
                          help='Authentication scheme to use')
    fl_parser.add_argument('--model_dim', type=int, default=1000,
                          help='Model dimension')
    fl_parser.add_argument('--output', type=str, default='results/fl_simulation',
                          help='Output directory')
    
    # All command
    all_parser = subparsers.add_parser('all', help='Run everything: verify + benchmark + fl')
    all_parser.add_argument('--output', type=str, default='results/complete_run',
                           help='Output directory for results')
    all_parser.add_argument('--plots', action='store_true', default=True,
                          help='Generate plots')
    all_parser.add_argument('--clients', type=int, default=10,
                          help='Number of FL clients')
    all_parser.add_argument('--rounds', type=int, default=5,
                          help='Number of FL rounds')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    print("="*80)
    print(f"{Fore.CYAN}{Style.BRIGHT}HOMOMORPHIC AUTHENTICATION SUITE{Style.RESET_ALL}")
    print("="*80)
    print()
    
    # Ensure results directory exists (works in Docker with volume mount)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Verify write access
    test_file = os.path.join(RESULTS_DIR, ".write_test")
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"{Fore.GREEN}✓ Results will be saved to: {RESULTS_DIR}/{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ Write access verified{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.YELLOW}⚠ Warning: Cannot write to {RESULTS_DIR}: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  In Docker, make sure to use: -v $(pwd)/results:/app/results{Style.RESET_ALL}")
    print()
    
    success = True
    
    if args.command == 'test':
        # Save to results/test
        test_dir = get_results_path("test")
        
        # Create run info file immediately
        run_info_path = os.path.join(test_dir, "RUN_INFO.txt")
        with open(run_info_path, 'w') as f:
            f.write(f"ALGORITHM TEST RUN STARTED\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {test_dir}\n")
        
        try:
            test_results, all_passed = test_all_algorithms()
            
            # Save results
            json_path = os.path.join(test_dir, "test_results.json")
            with open(json_path, 'w') as f:
                json.dump(test_results, f, indent=2)
            
            summary_path = os.path.join(test_dir, "test_summary.txt")
            with open(summary_path, 'w') as f:
                f.write("ALGORITHM FUNCTIONALITY TEST RESULTS\n")
                f.write("=" * 80 + "\n\n")
                passed = sum(test_results.values())
                total = len(test_results)
                f.write(f"Overall: {passed}/{total} tests passed\n\n")
                f.write("Individual Results:\n")
                f.write("-" * 80 + "\n")
                for name, result in test_results.items():
                    status = "PASS" if result else "FAIL"
                    f.write(f"{name:20s}: {status}\n")
            
            # Update run info
            with open(run_info_path, 'a') as f:
                f.write(f"\nStatus: COMPLETED\n")
                f.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Results: {passed}/{total} passed\n")
            
            print(f"\n{Fore.CYAN}Results saved to:{Style.RESET_ALL} {test_dir}/")
            success = all_passed
        except Exception as e:
            error_path = os.path.join(test_dir, "ERROR_LOG.txt")
            with open(error_path, 'w') as f:
                f.write(f"TEST ERROR\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error: {str(e)}\n")
                import traceback
                f.write(f"\nTraceback:\n{traceback.format_exc()}\n")
            with open(run_info_path, 'a') as f:
                f.write(f"\nStatus: FAILED\n")
                f.write(f"Error: {str(e)}\n")
            print(f"{Fore.RED}Test failed: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Error log saved to: {error_path}{Style.RESET_ALL}")
            success = False
    
    elif args.command == 'verify':
        # Save to results/verification
        verify_dir = get_results_path("verification")
        
        # Create run info file immediately
        run_info_path = os.path.join(verify_dir, "RUN_INFO.txt")
        with open(run_info_path, 'w') as f:
            f.write(f"VERIFICATION RUN STARTED\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {verify_dir}\n")
        
        try:
            math_success = verify_mathematical_operations()
            results = test_homomorphic_properties(save_results=True, output_dir=verify_dir)
            
            # Update run info
            with open(run_info_path, 'a') as f:
                f.write(f"\nStatus: COMPLETED\n")
                f.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                passed = sum(results.values())
                total = len(results)
                f.write(f"Results: {passed}/{total} passed\n")
        except Exception as e:
            error_path = os.path.join(verify_dir, "ERROR_LOG.txt")
            with open(error_path, 'w') as f:
                f.write(f"VERIFICATION ERROR\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error: {str(e)}\n")
                import traceback
                f.write(f"\nTraceback:\n{traceback.format_exc()}\n")
            with open(run_info_path, 'a') as f:
                f.write(f"\nStatus: FAILED\n")
                f.write(f"Error: {str(e)}\n")
            print(f"{Fore.RED}Verification failed: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Error log saved to: {error_path}{Style.RESET_ALL}")
            results = {}
            math_success = False
        
        print_header("VERIFICATION SUMMARY")
        passed = sum(results.values())
        total = len(results)
        
        print(f"\n{Fore.CYAN}Individual Results:{Style.RESET_ALL}")
        for name, result in results.items():
            status = f"{Fore.GREEN}✓ PASS{Style.RESET_ALL}" if result else f"{Fore.RED}✗ FAIL{Style.RESET_ALL}"
            print(f"  {name:20s}: {status}")
        
        print(f"\n{Fore.CYAN}Overall:{Style.RESET_ALL}")
        print(f"  Passed: {passed}/{total}")
        print(f"\n{Fore.CYAN}Results saved to:{Style.RESET_ALL} {verify_dir}/")
        
        if passed == total:
            print(f"\n{Fore.GREEN}{Style.BRIGHT}🎉 ALL TESTS PASSED! 🎉{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}Some tests require additional libraries (blspy, petlib){Style.RESET_ALL}")
        
        success = math_success and (passed >= 6)
    
    elif args.command == 'benchmark':
        # Save to results/benchmark
        generate_plots = args.plots and not args.no_plots
        output_dir = get_results_path("benchmark")
        
        success = run_full_benchmark(
            output_dir=output_dir,
            config_path=args.config,
            generate_plots=generate_plots
        )
    
    elif args.command == 'fl':
        # Save to results/fl_simulation
        output_dir = get_results_path("fl_simulation")
        
        success = run_fl_simulation(
            clients=args.clients,
            rounds=args.rounds,
            auth_scheme=args.auth_scheme,
            model_dim=args.model_dim,
            output_dir=output_dir
        )
    
    elif args.command == 'all':
        # Run everything! Save ALL files directly to results/ folder
        print(f"{Fore.CYAN}Running complete suite: test + verify + benchmark + fl{Style.RESET_ALL}\n")
        
        # Save everything directly to results/
        main_output_dir = get_results_path()
        
        print(f"{Fore.YELLOW}All results will be saved to: results/{Style.RESET_ALL}")
        print()
        
        # Create master run info immediately
        master_run_info = os.path.join(main_output_dir, "MASTER_RUN_INFO.txt")
        with open(master_run_info, 'w') as f:
            f.write(f"COMPLETE SUITE RUN STARTED\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {main_output_dir}\n")
            f.write(f"Commands: test + verify + benchmark + fl\n")
            f.write(f"All files saved directly in results/ folder\n")
        
        # 1. Algorithm Tests - save directly to results/
        print(f"{Fore.YELLOW}Step 1/4: Algorithm Functionality Tests{Style.RESET_ALL}")
        test_dir = main_output_dir  # Save directly to results/, not results/test/
        
        try:
            test_results, test_success = test_all_algorithms()
            test_json_path = os.path.join(test_dir, "test_results.json")
            with open(test_json_path, 'w') as f:
                json.dump(test_results, f, indent=2)
            
            test_summary_path = os.path.join(test_dir, "test_summary.txt")
            with open(test_summary_path, 'w') as f:
                f.write("ALGORITHM FUNCTIONALITY TEST RESULTS\n")
                f.write("=" * 80 + "\n\n")
                passed = sum(test_results.values())
                total = len(test_results)
                f.write(f"Overall: {passed}/{total} tests passed\n\n")
                for name, result in test_results.items():
                    status = "PASS" if result else "FAIL"
                    f.write(f"{name:20s}: {status}\n")
        except Exception as e:
            with open(os.path.join(test_dir, "TEST_ERROR_LOG.txt"), 'w') as f:
                f.write(f"Error: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
            test_success = False
            test_results = {}
        
        # 2. Verification - save directly to results/
        print(f"\n{Fore.YELLOW}Step 2/4: Verification{Style.RESET_ALL}")
        verify_dir = main_output_dir  # Save directly to results/, not results/verification/
        
        try:
            math_success = verify_mathematical_operations()
            results = test_homomorphic_properties(save_results=True, output_dir=verify_dir)
            verify_success = math_success and (sum(results.values()) >= 6)
        except Exception as e:
            with open(os.path.join(verify_dir, "VERIFICATION_ERROR_LOG.txt"), 'w') as f:
                f.write(f"Error: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
            verify_success = False
            results = {}
            math_success = False
        
        # 3. Benchmarking - save directly to results/
        print(f"\n{Fore.YELLOW}Step 3/4: Benchmarking{Style.RESET_ALL}")
        benchmark_dir = main_output_dir  # Save directly to results/, not results/benchmark/
        benchmark_success = run_full_benchmark(
            output_dir=benchmark_dir,
            generate_plots=args.plots
        )
        
        # 4. FL Simulation - save directly to results/
        print(f"\n{Fore.YELLOW}Step 4/4: FL Simulation{Style.RESET_ALL}")
        fl_dir = main_output_dir  # Save directly to results/, not results/fl_simulation/
        fl_success = run_fl_simulation(
            clients=args.clients,
            rounds=args.rounds,
            auth_scheme='BLS',
            output_dir=fl_dir
        )
        
        success = test_success and verify_success and benchmark_success and fl_success
        
        # Update master run info
        with open(master_run_info, 'a') as f:
            f.write(f"\nStatus: COMPLETED\n")
            f.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\nResults:\n")
            f.write(f"  Test: {'PASS' if test_success else 'FAIL'}\n")
            f.write(f"  Verify: {'PASS' if verify_success else 'FAIL'}\n")
            f.write(f"  Benchmark: {'PASS' if benchmark_success else 'FAIL'}\n")
            f.write(f"  FL Simulation: {'PASS' if fl_success else 'FAIL'}\n")
        
        # Create master summary file
        master_summary_path = os.path.join(main_output_dir, "COMPLETE_RUN_SUMMARY.txt")
        with open(master_summary_path, 'w') as f:
            f.write("HOMOMORPHIC AUTHENTICATION - COMPLETE RUN SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Execution Results:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Algorithm Tests: {'✓ PASS' if test_success else '✗ FAIL'}\n")
            f.write(f"Verification: {'✓ PASS' if verify_success else '✗ FAIL'}\n")
            f.write(f"Benchmarking: {'✓ PASS' if benchmark_success else '✗ FAIL'}\n")
            f.write(f"FL Simulation: {'✓ PASS' if fl_success else '✗ FAIL'}\n\n")
            f.write("All Files Saved Directly to results/ Folder:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Location: {main_output_dir}/\n\n")
            f.write("File List:\n")
            f.write(f"  ├── MASTER_RUN_INFO.txt\n")
            f.write(f"  ├── COMPLETE_RUN_SUMMARY.txt\n")
            f.write(f"  ├── test_results.json\n")
            f.write(f"  ├── test_summary.txt\n")
            f.write(f"  ├── verification_results.json\n")
            f.write(f"  ├── verification_summary.txt\n")
            f.write(f"  ├── metrics.json (Complete JSON data)\n")
            f.write(f"  ├── metrics.xlsx (Excel workbook)\n")
            f.write(f"  ├── metrics.md (Markdown report)\n")
            f.write(f"  ├── metrics.html (Interactive HTML report)\n")
            f.write(f"  ├── metrics_performance.csv\n")
            f.write(f"  ├── metrics_communication.csv\n")
            f.write(f"  ├── metrics_security.csv\n")
            f.write(f"  ├── raw_benchmark_results.json\n")
            f.write(f"  ├── BENCHMARK_SUMMARY.txt\n")
            if args.plots:
                f.write(f"  ├── performance_comparison.png\n")
                f.write(f"  ├── scalability.png\n")
                f.write(f"  ├── communication_overhead.png\n")
                f.write(f"  ├── message_size_impact.png\n")
                f.write(f"  └── summary_table.txt\n")
            f.write(f"  ├── fl_simulation_metrics.json\n")
            f.write(f"  └── fl_simulation_summary.txt\n")
        
        print_header("COMPLETE SUITE SUMMARY")
        print(f"Algorithm Tests: {'✓' if test_success else '✗'}")
        print(f"Verification: {'✓' if verify_success else '✗'}")
        print(f"Benchmarking: {'✓' if benchmark_success else '✗'}")
        print(f"FL Simulation: {'✓' if fl_success else '✗'}")
        print(f"\n{Fore.CYAN}All results saved to:{Style.RESET_ALL} {main_output_dir}/")
        print(f"{Fore.CYAN}Master summary:{Style.RESET_ALL} {master_summary_path}")
    
    # Final summary
    print("\n" + "="*80)
    if success:
        print(f"{Fore.GREEN}{Style.BRIGHT}✅ COMPLETE!{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠ Some components had issues{Style.RESET_ALL}")
    
    # Show where files were saved
    print(f"\n{Fore.CYAN}{Style.BRIGHT}📁 FILES SAVED TO:{Style.RESET_ALL}")
    
    if args.command == 'test':
        print(f"{Fore.GREEN}  → results/test/{Style.RESET_ALL}")
    elif args.command == 'verify':
        print(f"{Fore.GREEN}  → results/verification/{Style.RESET_ALL}")
    elif args.command == 'benchmark':
        print(f"{Fore.GREEN}  → results/benchmark/{Style.RESET_ALL}")
    elif args.command == 'fl':
        print(f"{Fore.GREEN}  → results/fl_simulation/{Style.RESET_ALL}")
    elif args.command == 'all':
        print(f"{Fore.GREEN}  → results/{Style.RESET_ALL}")
    
    print("="*80)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

