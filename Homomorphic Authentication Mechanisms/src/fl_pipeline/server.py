import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import time

from src.algorithms import (
    BLSSignature, LHSSignature, WatersHomomorphicSignature,
    BonehBoyenHomomorphicSignature, RSASignature, EdDSASignature,
    AdditiveHMAC, LinearHMAC, PolynomialHMAC, LatticeHMAC
)
from src.fl_pipeline.aggregation import Aggregator

class FLServer:
    def __init__(self, model_dim: int = 1000, auth_scheme: str = "BLS",
                 aggregation_method: str = "average"):
        self.model_dim = model_dim
        self.auth_scheme = auth_scheme
        self.global_model = np.zeros(model_dim, dtype=np.float32)
        self.auth_verifier = None
        self.client_public_keys = {}
        self.shared_mac_key = None
        self.aggregator = Aggregator(method=aggregation_method)
        self.aggregation_method = aggregation_method
        self._init_auth_scheme()
    
    def _init_auth_scheme(self):
        scheme = self.auth_scheme
        if scheme == "BLS":
            self.auth_verifier = BLSSignature()
        elif scheme == "LHS":
            self.auth_verifier = LHSSignature(vector_dim=self.model_dim)
        elif scheme == "Waters":
            self.auth_verifier = WatersHomomorphicSignature()
        elif scheme == "BonehBoyen":
            self.auth_verifier = BonehBoyenHomomorphicSignature()
        elif scheme == "RSA":
            self.auth_verifier = RSASignature()
        elif scheme == "EdDSA":
            self.auth_verifier = EdDSASignature()
        elif scheme == "Additive_HMAC":
            self.auth_verifier = AdditiveHMAC()
            self.shared_mac_key = self.auth_verifier.key_generation()
        elif scheme == "Linear_HMAC":
            self.auth_verifier = LinearHMAC(vector_dim=self.model_dim)
            self.shared_mac_key = self.auth_verifier.key_generation()
        elif scheme == "Polynomial_HMAC":
            self.auth_verifier = PolynomialHMAC()
            self.shared_mac_key = self.auth_verifier.key_generation()
        elif scheme == "Lattice_HMAC":
            self.auth_verifier = LatticeHMAC()
            self.shared_mac_key = self.auth_verifier.key_generation()
    def register_client_public_key(self, client_id: int, public_key: bytes):
        self.client_public_keys[client_id] = public_key
    def receive_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Receive and aggregate client updates (server-side)"""
        if not client_updates:
            return {"aggregated_update": None, "method": "none"}
        
        updates = [u.get("update", u) if isinstance(u, dict) else u for u in client_updates]
        
        # Verify each client update before aggregation (server-side verification)
        verified_updates = []
        for update_data in client_updates:
            if isinstance(update_data, dict):
                update = update_data.get("update")
                auth_tag = update_data.get("auth_tag")
                auth_metadata = update_data.get("auth_metadata", {})
                
                # Server-side verification
                if update is not None and auth_tag:
                    is_valid, _ = self.verify_client_update(update, auth_tag, auth_metadata)
                    if is_valid:
                        verified_updates.append(update)
        
        if not verified_updates:
            return {"aggregated_update": None, "method": "none", "verified_count": 0}
        
        if isinstance(verified_updates[0], np.ndarray):
            aggregated = self.aggregator.aggregate(verified_updates)
            return {"aggregated_update": aggregated, "method": self.aggregation_method, "verified_count": len(verified_updates)}
        else:
            return {"aggregated_update": verified_updates[0] if verified_updates else None, "method": "none", "verified_count": len(verified_updates)}
    
    def verify_client_update(self, update: Any, auth_tag: bytes, auth_metadata: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Server-side verification of individual client update"""
        if not self.auth_verifier:
            return True, {}
        
        verify_metadata = {}
        if isinstance(update, np.ndarray):
            update_bytes = update.tobytes()
        else:
            update_bytes = str(update).encode()
        
        if hasattr(self.auth_verifier, 'verify'):
            public_key = auth_metadata.get("public_key")
            if public_key:
                valid, verify_time = self.auth_verifier.verify(update_bytes, auth_tag, public_key)
            else:
                valid, verify_time = self.auth_verifier.verify(update_bytes, auth_tag)
            verify_metadata["verify_time"] = verify_time
            verify_metadata["method"] = "signature_verification"
            return valid, verify_metadata
        elif hasattr(self.auth_verifier, 'verify_tag'):
            identifier = auth_metadata.get("identifier", b"")
            valid, verify_time = self.auth_verifier.verify_tag(update_bytes, auth_tag, identifier)
            verify_metadata["verify_time"] = verify_time
            verify_metadata["method"] = "tag_verification"
            return valid, verify_metadata
        
        return True, {}
    def aggregate_auth_tags(self, auth_tags: List[bytes], auth_metadata_list: List[Dict],
                           updates: List[Any], client_ids: List[int]) -> Tuple[bytes, Dict[str, Any]]:
        if not self.auth_verifier:
            return b'', {}
        if isinstance(self.auth_verifier, (BLSSignature, LHSSignature, WatersHomomorphicSignature, BonehBoyenHomomorphicSignature)):
            aggregated_tag, agg_time = self.auth_verifier.aggregate_signatures(auth_tags)
            return aggregated_tag, {"method": "homomorphic_aggregation", "time": agg_time}
        elif isinstance(self.auth_verifier, EdDSASignature):
            aggregated_tag, agg_time = self.auth_verifier.aggregate_signatures(auth_tags)
            return aggregated_tag, {"method": "concatenation", "time": agg_time}
        elif isinstance(self.auth_verifier, RSASignature):
            return auth_tags[0] if auth_tags else b'', {"method": "no_aggregation"}
        elif isinstance(self.auth_verifier, AdditiveHMAC):
            combined_tag, combine_time = self.auth_verifier.combine_tags(auth_tags)
            return combined_tag, {"method": "additive_combination", "time": combine_time}
        elif isinstance(self.auth_verifier, LinearHMAC):
            coefficients = [1.0 / len(updates)] * len(updates)
            combined_tag, combine_time = self.auth_verifier.linear_combine_tags(auth_tags, coefficients)
            return combined_tag, {"method": "linear_combination", "time": combine_time}
        elif isinstance(self.auth_verifier, LatticeHMAC):
            combined_tag, combine_time = self.auth_verifier.lattice_combine_tags(auth_tags)
            return combined_tag, {"method": "lattice_combination", "time": combine_time}
        return auth_tags[0] if auth_tags else b'', {"method": "no_aggregation"}
    def verify_aggregated(self, aggregated_update: Any, aggregated_auth_tag: bytes,
                         individual_updates: List[Any], individual_tags: List[bytes],
                         auth_metadata_list: List[Dict], client_ids: List[int]) -> Tuple[bool, Dict[str, Any]]:
        if not self.auth_verifier:
            return True, {}
        verify_metadata = {}
        if isinstance(self.auth_verifier, (BLSSignature, LHSSignature, WatersHomomorphicSignature, BonehBoyenHomomorphicSignature)):
            messages = []
            public_keys = []
            for update, metadata in zip(individual_updates, auth_metadata_list):
                if isinstance(update, np.ndarray):
                    messages.append(update.tobytes())
                else:
                    messages.append(str(update).encode())
                if "public_key" in metadata:
                    public_keys.append(metadata["public_key"])
            if public_keys:
                valid, verify_time = self.auth_verifier.aggregate_verify(
                    messages, aggregated_auth_tag, public_keys
                )
                verify_metadata["verify_time"] = verify_time
                verify_metadata["method"] = "homomorphic_aggregate"
                return valid, verify_metadata
        elif isinstance(self.auth_verifier, EdDSASignature):
            messages = []
            public_keys = []
            for update, metadata in zip(individual_updates, auth_metadata_list):
                if isinstance(update, np.ndarray):
                    messages.append(update.tobytes())
                else:
                    messages.append(str(update).encode())
                if "public_key" in metadata:
                    public_keys.append(metadata["public_key"])
            if public_keys:
                valid, verify_time = self.auth_verifier.aggregate_verify(
                    messages, aggregated_auth_tag, public_keys
                )
                verify_metadata["verify_time"] = verify_time
                verify_metadata["method"] = "concatenation_individual"
                return valid, verify_metadata
        elif isinstance(self.auth_verifier, RSASignature):
            messages = []
            signatures = []
            public_keys = []
            for update, tag, metadata in zip(individual_updates, individual_tags, auth_metadata_list):
                if isinstance(update, np.ndarray):
                    messages.append(update.tobytes())
                else:
                    messages.append(str(update).encode())
                signatures.append(tag)
                if "public_key" in metadata:
                    public_keys.append(metadata["public_key"])
            if public_keys and signatures:
                valid, verify_time = self.auth_verifier.aggregate_verify(
                    messages, signatures, public_keys
                )
                verify_metadata["verify_time"] = verify_time
                verify_metadata["method"] = "individual_verification"
                return valid, verify_metadata
        elif isinstance(self.auth_verifier, AdditiveHMAC):
            messages = []
            identifiers = []
            for update, metadata in zip(individual_updates, auth_metadata_list):
                if isinstance(update, np.ndarray):
                    messages.append(update.tobytes())
                else:
                    messages.append(str(update).encode())
                if "identifier" in metadata:
                    identifiers.append(metadata["identifier"])
            if messages:
                combined_message = b''.join(messages[:16])
                if len(combined_message) < 16:
                    combined_message = combined_message.ljust(16, b'\0')
            else:
                combined_message = b'\0' * 16
            valid, verify_time = self.auth_verifier.verify_combined(
                messages, combined_message, aggregated_auth_tag, identifiers
            )
            verify_metadata["verify_time"] = verify_time
            verify_metadata["method"] = "additive_combination"
            return valid, verify_metadata
        elif isinstance(self.auth_verifier, LinearHMAC):
            vectors = [u for u in individual_updates if isinstance(u, np.ndarray)]
            if vectors:
                coefficients = [1.0 / len(vectors)] * len(vectors)
                combined_vector = sum(coeff * vec for coeff, vec in zip(coefficients, vectors))
                identifiers = [m.get("identifier", b"") for m in auth_metadata_list]
                valid, verify_time = self.auth_verifier.verify_linear_combination(
                    vectors, combined_vector, aggregated_auth_tag, coefficients, identifiers
                )
                verify_metadata["verify_time"] = verify_time
                verify_metadata["method"] = "linear_combination"
                return valid, verify_metadata
        all_valid = True
        for update, tag, metadata in zip(individual_updates, individual_tags, auth_metadata_list):
            if isinstance(update, np.ndarray):
                update_bytes = update.tobytes()
            else:
                update_bytes = str(update).encode()
            if "identifier" in metadata:
                valid, _ = self.auth_verifier.verify_tag(update_bytes, tag, metadata["identifier"])
            else:
                valid, _ = self.auth_verifier.verify(update_bytes, tag)
            all_valid = all_valid and valid
        return all_valid, verify_metadata
    def update_global_model(self, aggregated_update: np.ndarray, learning_rate: float = 1.0):
        if isinstance(aggregated_update, np.ndarray):
            self.global_model = self.global_model + learning_rate * aggregated_update
        return self.global_model
    def get_global_model(self) -> np.ndarray:
        return self.global_model.copy()