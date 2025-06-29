# sec_agg/task_paillier.py

from phe import paillier
import numpy as np

class PaillierContext:
    """A helper class to manage Paillier keys and fixed-point precision."""
    def __init__(self, key_size=1024, precision_bits=16):
        print(f"Generating Paillier keypair with size {key_size}...")
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_size)
        # The precision factor is used for fixed-point encoding
        self.precision = 2**precision_bits
        print("Paillier keypair generated.")

def encode_weights(weights: np.ndarray, precision: int) -> np.ndarray:
    """Converts a NumPy array of floats to an array of integers for Paillier."""
    return (weights * precision).astype(int)

def decode_weights(encoded_weights: np.ndarray, precision: int) -> np.ndarray:
    """Converts a NumPy array of integers back to floats."""
    return encoded_weights / precision

def encrypt_vector(pub_key, vector: np.ndarray):
    """Encrypts each element of a NumPy array."""
    return np.array([pub_key.encrypt(x) for x in vector])

def decrypt_vector(priv_key, vector):
    """Decrypts each element of a NumPy array of EncryptedNumbers."""
    return np.array([priv_key.decrypt(x) for x in vector])