import tenseal as ts
import numpy as np
from typing import List

def get_bfv_context():
    """
    Creates and returns a TenSEAL context configured for the BFV scheme.
    - poly_modulus_degree is set for security and capacity.
    - plain_modulus is set to a specific prime number that is large
      enough to prevent overflow and is compatible with batching.
    """
    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=16384,
        # FIXED: Manually provide a suitable prime number for the plaintext modulus.
        # This prime is chosen to be > 20 bits and compatible with batching for N=8192.
        plain_modulus=786433
    )
    # Batching requires generating Galois keys.
    context.generate_galois_keys()
    return context

def encode(vector: np.ndarray, precision_bits=16) -> List[int]:
    """
    Encodes a vector of floats into a vector of integers for BFV
    by multiplying by a precision factor.
    """
    precision = 1 << precision_bits  # 2**16
    return (vector * precision).astype(int).tolist()

def decode(vector: List[int], precision_bits=16) -> np.ndarray:
    """
    Decodes a vector of integers back into a vector of floats.
    """
    precision = 1 << precision_bits
    return (np.array(vector) / precision).astype(np.float32)
