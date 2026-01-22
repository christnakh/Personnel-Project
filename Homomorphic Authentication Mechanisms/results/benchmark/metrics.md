# Homomorphic Authentication Benchmark Report

**Generated:** 2025-12-15 21:39:45

## Executive Summary

- **Total Algorithms Tested:** 13
- **Total Operations:** 254
- **Performance Measurements:** 254
- **Communication Measurements:** 10
- **Security Assessments:** 10

---

## Performance Metrics

### By Algorithm

### FL_Linear_HMAC

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| Key Generation (s) | 0.000000 | 0.000000 | 0.000000 |
| Signing (s) | 0.000000 | 0.000000 | 0.000000 |
| Verification (s) | 0.000000 | 0.000000 | 0.000000 |
| Aggregation (s) | 0.000000 | - | - |
| Peak Memory (MB) | 0.00 | - | - |
| Avg Memory (MB) | 0.00 | - | - |
| Measurements | 18 | - | - |

### Waters

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| Key Generation (s) | 0.005509 | 0.005509 | 0.005509 |
| Signing (s) | 0.006147 | 0.006060 | 0.006394 |
| Verification (s) | 0.006644 | 0.006595 | 0.006694 |
| Aggregation (s) | 0.000000 | - | - |
| Peak Memory (MB) | 0.00 | - | - |
| Avg Memory (MB) | 0.00 | - | - |
| Measurements | 20 | - | - |

### BLS

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| Key Generation (s) | 0.000164 | 0.000164 | 0.000164 |
| Signing (s) | 0.000522 | 0.000476 | 0.000688 |
| Verification (s) | 0.001338 | 0.001161 | 0.001992 |
| Aggregation (s) | 0.026742 | - | - |
| Peak Memory (MB) | 0.00 | - | - |
| Avg Memory (MB) | 0.00 | - | - |
| Measurements | 20 | - | - |

### Lattice_HMAC

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| Key Generation (s) | 0.000269 | 0.000269 | 0.000269 |
| Signing (s) | 0.000232 | 0.000210 | 0.000322 |
| Verification (s) | 0.000000 | 0.000000 | 0.000000 |
| Aggregation (s) | 0.000000 | - | - |
| Peak Memory (MB) | 0.00 | - | - |
| Avg Memory (MB) | 0.00 | - | - |
| Measurements | 20 | - | - |

### BonehBoyen

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| Key Generation (s) | 0.000069 | 0.000069 | 0.000069 |
| Signing (s) | 0.000090 | 0.000068 | 0.000180 |
| Verification (s) | 0.000685 | 0.000662 | 0.000752 |
| Aggregation (s) | 0.200791 | - | - |
| Peak Memory (MB) | 0.00 | - | - |
| Avg Memory (MB) | 0.00 | - | - |
| Measurements | 20 | - | - |

### Linear_HMAC

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| Key Generation (s) | 0.000024 | 0.000024 | 0.000024 |
| Signing (s) | 0.000022 | 0.000019 | 0.000030 |
| Verification (s) | 0.000020 | 0.000019 | 0.000021 |
| Aggregation (s) | 0.000000 | - | - |
| Peak Memory (MB) | 0.00 | - | - |
| Avg Memory (MB) | 0.00 | - | - |
| Measurements | 20 | - | - |

### Additive_HMAC

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| Key Generation (s) | 0.000006 | 0.000006 | 0.000006 |
| Signing (s) | 0.000058 | 0.000008 | 0.000191 |
| Verification (s) | 0.000030 | 0.000008 | 0.000123 |
| Aggregation (s) | 0.000048 | - | - |
| Peak Memory (MB) | 0.00 | - | - |
| Avg Memory (MB) | 0.00 | - | - |
| Measurements | 20 | - | - |

### EdDSA

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| Key Generation (s) | 0.000475 | 0.000475 | 0.000475 |
| Signing (s) | 0.000118 | 0.000042 | 0.000440 |
| Verification (s) | 0.000129 | 0.000093 | 0.000283 |
| Aggregation (s) | 0.000007 | - | - |
| Peak Memory (MB) | 0.00 | - | - |
| Avg Memory (MB) | 0.00 | - | - |
| Measurements | 20 | - | - |

### LHS

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| Key Generation (s) | 0.000796 | 0.000796 | 0.000796 |
| Signing (s) | 0.000729 | 0.000687 | 0.000752 |
| Verification (s) | 0.001276 | 0.001248 | 0.001328 |
| Aggregation (s) | 0.000000 | - | - |
| Peak Memory (MB) | 0.00 | - | - |
| Avg Memory (MB) | 0.00 | - | - |
| Measurements | 20 | - | - |

### Polynomial_HMAC

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| Key Generation (s) | 0.000005 | 0.000005 | 0.000005 |
| Signing (s) | 0.000025 | 0.000003 | 0.000118 |
| Verification (s) | 0.000025 | 0.000003 | 0.000118 |
| Aggregation (s) | 0.000000 | - | - |
| Peak Memory (MB) | 0.00 | - | - |
| Avg Memory (MB) | 0.00 | - | - |
| Measurements | 20 | - | - |

### FL_BLS

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| Key Generation (s) | 0.000000 | 0.000000 | 0.000000 |
| Signing (s) | 0.000000 | 0.000000 | 0.000000 |
| Verification (s) | 0.013364 | 0.004724 | 0.024944 |
| Aggregation (s) | 0.000000 | - | - |
| Peak Memory (MB) | 0.00 | - | - |
| Avg Memory (MB) | 0.00 | - | - |
| Measurements | 18 | - | - |

### FL_Additive_HMAC

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| Key Generation (s) | 0.000000 | 0.000000 | 0.000000 |
| Signing (s) | 0.000000 | 0.000000 | 0.000000 |
| Verification (s) | 0.000000 | 0.000000 | 0.000000 |
| Aggregation (s) | 0.000000 | - | - |
| Peak Memory (MB) | 0.00 | - | - |
| Avg Memory (MB) | 0.00 | - | - |
| Measurements | 18 | - | - |

### RSA

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| Key Generation (s) | 0.292958 | 0.292958 | 0.292958 |
| Signing (s) | 0.001383 | 0.001131 | 0.002387 |
| Verification (s) | 0.000497 | 0.000253 | 0.001530 |
| Aggregation (s) | 0.000027 | - | - |
| Peak Memory (MB) | 0.00 | - | - |
| Avg Memory (MB) | 0.00 | - | - |
| Measurements | 20 | - | - |


## Communication Metrics

| Algorithm | Signature Size (B) | Public Key Size (B) | Total Message Size (B) | Compression Ratio |
|-----------|-------------------|---------------------|----------------------|-------------------|
| Waters | 65 | 65 | 0 | 1.00 |
| BLS | 96 | 48 | 0 | 1.00 |
| Lattice_HMAC | 32 | 0 | 0 | 1.00 |
| BonehBoyen | 65 | 65 | 0 | 1.00 |
| Linear_HMAC | 32 | 0 | 0 | 1.00 |
| Additive_HMAC | 32 | 0 | 0 | 1.00 |
| EdDSA | 64 | 32 | 0 | 1.00 |
| LHS | 65 | 65 | 0 | 1.00 |
| Polynomial_HMAC | 32 | 0 | 0 | 1.00 |
| RSA | 256 | 256 | 0 | 1.00 |

## Security Properties

| Algorithm | Security Notion | Public Verifiable | Post-Quantum | Homomorphic Operations |
|-----------|----------------|-------------------|--------------|----------------------|
| Waters | EUF-CMA | Yes | No | N/A |
| BLS | EUF-CMA | Yes | No | N/A |
| Lattice_HMAC | EUF-CMA | No | No | N/A |
| BonehBoyen | EUF-CMA | Yes | No | N/A |
| Linear_HMAC | EUF-CMA | No | No | N/A |
| Additive_HMAC | EUF-CMA | No | No | N/A |
| EdDSA | EUF-CMA | Yes | No | N/A |
| LHS | EUF-CMA | Yes | No | N/A |
| Polynomial_HMAC | EUF-CMA | No | No | N/A |
| RSA | EUF-CMA | Yes | No | N/A |
