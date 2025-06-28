# sec_agg/task_he.py

import tenseal as ts

def get_fhe_context():
    """
    Creates and returns a TenSEAL context.
    
    The poly_modulus_degree is increased to 16384 to ensure the entire
    flattened model weight vector fits into a single ciphertext, which is
    crucial for enabling all vector operations and ensuring convergence.
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384, # INCREASED from 8192
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context