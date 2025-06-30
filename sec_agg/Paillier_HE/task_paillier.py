# sec_agg/task_paillier.py

from phe import paillier
import numpy as np

class PaillierContext:
    """A helper class to manage Paillier keys and fixed-point precision."""
    def __init__(self, key_size=512):
        print(f"Generating Paillier keypair with size {key_size}...")
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_size)
        print("Paillier keypair generated.")

def encrypt_weights(pub_key: paillier.PaillierPublicKey, weights: list) -> list:
    """
    Encrypts a list of NumPy arrays element-wise.
    It explicitly casts each number to a standard Python float before encryption.
    """
    encrypted_layers = []
    for layer in weights:
        # Flatten the layer, cast each element to a Python float, then encrypt
        # This is the key fix to prevent the TypeError
        encrypted_flat_layer = [pub_key.encrypt(float(x)) for x in layer.flatten()]
        
        # Reshape it back to the original shape and store it
        encrypted_layers.append(np.array(encrypted_flat_layer).reshape(layer.shape))
        
    return encrypted_layers

def decrypt_weights(priv_key: paillier.PaillierPrivateKey, encrypted_weights: list) -> list:
    """Decrypts a list of NumPy arrays of EncryptedNumbers."""
    decrypted_layers = []
    for layer in encrypted_weights:
        # Decrypt each element, which will return a float due to phe's precision handling
        decrypted_flat_layer = [priv_key.decrypt(x) for x in layer.flatten()]
        
        # Reshape it back and convert to a standard numpy float array
        decrypted_layers.append(np.array(decrypted_flat_layer, dtype=np.float32).reshape(layer.shape))
        
    return decrypted_layers

# The other two functions (encode/decode) are not strictly needed with the new `encrypt_weights`
# but can be kept for other potential uses.
def encode(pub_key: paillier.PaillierPublicKey, x):
    """Encodes and encrypts a single number."""
    return pub_key.encrypt(x, precision=1e-6)

def decode(priv_key: paillier.PaillierPrivateKey, x):
    """Decrypts and decodes a single number."""
    return priv_key.decrypt(x)