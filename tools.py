import numpy as np

def random_point_in_triangle(vertices):
    """
    Generates a random dot into a triangle defined by vertex in (RA, Dec).

    :param vertex: List of tuples [(ra1, dec1), (ra2, dec2), (ra3, dec3)] which defines the triangle.
    :return: A tuple (ra, dec) which represents a dot into the triangle.
    """
    v0 = np.array(vertices[0])
    v1 = np.array(vertices[1])
    v2 = np.array(vertices[2])

    # Generates random dots
    r1 = np.sqrt(np.random.uniform(0, 1))  # sqrt to assure uniformity
    r2 = np.random.uniform(0, 1)

    # Baricentric coordinates
    point = (1 - r1) * v0 + r1 * (1 - r2) * v1 + r1 * r2 * v2
    return tuple(point)