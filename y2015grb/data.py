# Source: https://arxiv.org/pdf/1507.06662.pdf
import numpy as np

def get_data():
    legend = 'GRB.2015'

    z_values = np.array([
        1.48,
        1.52,
        1.61,
        1.62,
        1.98,
        2.32,
        2.61,
        2.66,
        2.90,
        3.20,
        3.21,
        3.80,
    ], dtype=np.float64, copy=False)

    distance_modulus_values = np.array([
        45.0990,
        44.8357,
        45.5094,
        45.6337,
        46.6219,
        46.0496,
        47.4273,
        46.8414,
        47.1522,
        46.5993,
        45.4623,
        50.9127,
    ], dtype=np.float64, copy=False)

    sigma_distance_moduli = np.array([
        0.4684,
        0.5404,
        0.6244,
        0.4921,
        0.6476,
        0.9760,
        0.7877,
        0.8608,
        0.6272,
        0.6032,
        0.6342,
        1.0548
    ], dtype=np.float64, copy=False)

    return (legend, z_values, distance_modulus_values, sigma_distance_moduli)
