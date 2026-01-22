import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import time

from src.algorithms import (
    BLSSignature, LHSSignature, WatersHomomorphicSignature,
    BonehBoyenHomomorphicSignature, RSASignature, EdDSASignature,
    AdditiveHMAC, LinearHMAC, PolynomialHMAC, LatticeHMAC
)
from src.fl_pipeline.aggregation import Aggregator

class FLClient:
    def __init__(self, client_id: int, model_dim: int = 1000,
                 auth_scheme: str = "BLS"):
        self.client_id = client_id
        self.model_dim = model_dim
        self.auth_scheme = auth_scheme
        self.local_model = np.random.randn(model_dim).astype(np.float32)
        self.auth_handler = None
        self.auth_keys = {}
        self._init_auth_scheme()
    
    def _init_auth_scheme(self):
        scheme = self.auth_scheme
        if scheme == "BLS":
            self.auth_handler = BLSSignature()
            priv_key, pub_key = self.auth_handler.key_generation()
            self.auth_keys = {"private": priv_key, "public": pub_key}
        elif scheme == "LHS":
            self.auth_handler = LHSSignature(vector_dim=self.model_dim)
            priv_key, pub_key = self.auth_handler.key_generation()
            self.auth_keys = {"private": priv_key, "public": pub_key}
        elif scheme == "Waters":
            self.auth_handler = WatersHomomorphicSignature()
            priv_key, pub_key = self.auth_handler.key_generation()
            self.auth_keys = {"private": priv_key, "public": pub_key}
        elif scheme == "BonehBoyen":
            self.auth_handler = BonehBoyenHomomorphicSignature()
            priv_key, pub_key = self.auth_handler.key_generation()
            self.auth_keys = {"private": priv_key, "public": pub_key}
        elif scheme == "RSA":
            self.auth_handler = RSASignature()
            priv_key, pub_key = self.auth_handler.key_generation()
            self.auth_keys = {"private": priv_key, "public": pub_key}
        elif scheme == "EdDSA":
            self.auth_handler = EdDSASignature()
            priv_key, pub_key = self.auth_handler.key_generation()
            self.auth_keys = {"private": priv_key, "public": pub_key}
        elif scheme == "Additive_HMAC":
            self.auth_handler = AdditiveHMAC()
            secret_key = self.auth_handler.key_generation()
            self.auth_keys = {"secret": secret_key}
        elif scheme == "Linear_HMAC":
            self.auth_handler = LinearHMAC(vector_dim=self.model_dim)
            secret_key = self.auth_handler.key_generation()
            self.auth_keys = {"secret": secret_key}
        elif scheme == "Polynomial_HMAC":
            self.auth_handler = PolynomialHMAC()
            secret_key = self.auth_handler.key_generation()
            self.auth_keys = {"secret": secret_key}
        elif scheme == "Lattice_HMAC":
            self.auth_handler = LatticeHMAC()
            secret_key = self.auth_handler.key_generation()
            self.auth_keys = {"secret": secret_key}
    def compute_local_update(self, global_model: Optional[np.ndarray] = None) -> np.ndarray:
        update = np.random.randn(self.model_dim).astype(np.float32) * 0.1
        if global_model is not None:
            update = global_model + update
        return update
    
    def prepare_update(self, global_model: Optional[np.ndarray] = None) -> Dict[str, Any]:
        update = self.compute_local_update(global_model)
        auth_data = self.authenticate_update(update)
        result = {
            "update": update,
            "client_id": self.client_id,
            "auth_tag": auth_data.get("tag"),
            "auth_metadata": auth_data.get("metadata", {})
        }
        return result
    
    def verify_server_response(self, aggregated_update: np.ndarray, aggregated_auth_tag: bytes,
                               server_metadata: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Client-side verification of server's aggregated update"""
        if not self.auth_handler:
            return True, {}
        
        verify_metadata = {}
        update_bytes = aggregated_update.tobytes()
        
        # Verify aggregated authentication tag
        if hasattr(self.auth_handler, 'verify'):
            valid, verify_time = self.auth_handler.verify(update_bytes, aggregated_auth_tag)
            verify_metadata["verify_time"] = verify_time
            verify_metadata["method"] = "signature_verification"
            return valid, verify_metadata
        elif hasattr(self.auth_handler, 'verify_tag'):
            identifier = f"aggregated_round".encode()
            valid, verify_time = self.auth_handler.verify_tag(update_bytes, aggregated_auth_tag, identifier)
            verify_metadata["verify_time"] = verify_time
            verify_metadata["method"] = "tag_verification"
            return valid, verify_metadata
        
        return True, {}
    
    def authenticate_update(self, update: np.ndarray) -> Dict[str, Any]:
        if not self.auth_handler:
            return {"tag": b"", "metadata": {}}
        update_bytes = update.tobytes()
        if hasattr(self.auth_handler, 'sign'):
            sig, sign_time = self.auth_handler.sign(update_bytes)
            metadata = {
                "public_key": self.auth_keys.get("public"),
                "sign_time": sign_time
            }
            return {"tag": sig, "metadata": metadata}
        elif hasattr(self.auth_handler, 'generate_tag'):
            identifier = f"client_{self.client_id}".encode()
            tag, gen_time = self.auth_handler.generate_tag(update_bytes, identifier)
            metadata = {
                "identifier": identifier,
                "gen_time": gen_time
            }
            return {"tag": tag, "metadata": metadata}
        return {"tag": b"", "metadata": {}}