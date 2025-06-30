# sec_agg/task_bfv.py

import tenseal as ts
import numpy as np

def get_bfv_context():
    """
    Creates and returns a TenSEAL context configured for the BFV scheme.
    BFV works on integers, so we need a plaintext modulus that can hold
    our encoded numbers and a polynomial modulus for security.
    """
    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=4096,
        plain_modulus=1032193  # A prime number large enough for our data
    )
    # Note: Unlike CKKS, BFV doesn't need a scale set in the context.
    # The encoding/decoding is handled manually.
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
    return (np.array(vector) / precision).astype(np.float32)# sec_agg/task_bfv.py

import tenseal as ts
import numpy as np

def get_bfv_context():
    """
    Creates and returns a TenSEAL context configured for the BFV scheme.
    BFV works on integers, so we need a plaintext modulus that can hold
    our encoded numbers and a polynomial modulus for security.
    """
    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=4096,
        plain_modulus=1032193  # A prime number large enough for our data
    )
    # Note: Unlike CKKS, BFV doesn't need a scale set in the context.
    # The encoding/decoding is handled manually.
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