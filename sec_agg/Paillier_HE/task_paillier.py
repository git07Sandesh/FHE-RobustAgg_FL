# sec_agg/task_paillier.py

from phe import paillier
import numpy as np

class PaillierContext:
    """A helper class to manage Paillier keys and fixed-point precision."""
    def __init__(self, key_size=512):
        # Key size of 512 bits is used for speed in this example.
        # For real-world security, 2048-bit or 3072-bit keys are recommended.
        print(f"Generating Paillier keypair with size {key_size}...")
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_size)
        print("Paillier keypair generated.")

def encrypt_weights(pub_key: paillier.PaillierPublicKey, weights: list) -> list:
    """
    Encrypts a list of NumPy arrays element-wise.
    It explicitly casts each number to a standard Python float before encryption.
    The result is a list of NumPy arrays where each element is an EncryptedNumber.
    """
    encrypted_layers = []
    for layer in weights:
        # Flatten the layer, cast each element to a Python float, then encrypt
        encrypted_flat_layer = [pub_key.encrypt(float(x)) for x in layer.flatten()]
        
        # Reshape it back to the original shape and store it as a NumPy array of EncryptedNumbers
        # This is important for consistent handling (e.g., using np.sum or other np operations later)
        encrypted_layers.append(np.array(encrypted_flat_layer, dtype=object).reshape(layer.shape))
        
    return encrypted_layers

def decrypt_weights(priv_key: paillier.PaillierPrivateKey, encrypted_weights: list) -> list:
    """
    Decrypts a list of NumPy arrays of EncryptedNumbers.
    The result is a list of NumPy arrays of floats.
    """
    decrypted_layers = []
    for layer_encrypted in encrypted_weights:
        # Decrypt each element, which will return a float due to phe's precision handling
        decrypted_flat_layer = [priv_key.decrypt(x) for x in layer_encrypted.flatten()]
        
        # Reshape it back and convert to a standard numpy float32 array
        decrypted_layers.append(np.array(decrypted_flat_layer, dtype=np.float32).reshape(layer_encrypted.shape))
        
    return decrypted_layers

# The encode/decode functions below are for single numbers with explicit precision.
# The encrypt_weights/decrypt_weights functions above handle the model weights directly.
# These specific encode/decode might not be strictly needed for the federated learning flow
# but could be useful for other Paillier operations.
# They explicitly use `precision` argument for `phe.encrypt`, which is a different way
# of handling fixed-point numbers compared to `encrypt_weights` casting to float directly.
# The current `encrypt_weights` implicitly relies on `phe`'s internal float handling.

# def encode(pub_key: paillier.PaillierPublicKey, x, precision=1e-6):
#     """Encodes and encrypts a single number."""
#     return pub_key.encrypt(x, precision=precision)

# def decode(priv_key: paillier.PaillierPrivateKey, x):
#     """Decrypts and decodes a single number."""
#     return priv_key.decrypt(x)
