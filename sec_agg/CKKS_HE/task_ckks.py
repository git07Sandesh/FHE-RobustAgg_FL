# sec_agg/task_ckks.py

import tenseal as ts

def get_ckks_context():
    """
    Creates and returns a TenSEAL context configured for the CKKS scheme.
    Parameters are chosen to support the multiplicative depth needed for Krum
    and to ensure the SmallNet model fits in a single ciphertext.
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context