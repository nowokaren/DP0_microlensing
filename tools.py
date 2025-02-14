import numpy as np
import random

import sys

def tri_sample(vertex, edge_threshold=0.01):
    """
    Generates a random dot into a triangle defined by vertex in (RA, Dec), avoiding edges.
    vertex: List of tuples [(ra1, dec1), (ra2, dec2), (ra3, dec3)] which defines the triangle.
    :param edge_threshold: Minimum distance from edges (as a fraction of triangle size)
    :return: A tuple (ra, dec) which represents a dot into the triangle.
    """
    v0 = np.array(vertex[0])
    v1 = np.array(vertex[1])
    v2 = np.array(vertex[2])
    
    while True:
        r1 = np.sqrt(np.random.uniform(0, 1))  # sqrt to assure uniformity
        r2 = np.random.uniform(0, 1)
        
        # Baricentric coordinates
        w1 = 1 - r1
        w2 = r1 * (1 - r2)
        w3 = r1 * r2
        
        # Check if point is too close to edges
        if min(w1, w2, w3) > edge_threshold:
            point = w1 * v0 + w2 * v1 + w3 * v2
            return tuple(point)


def distance(p1, p2):
    '''Return distance between two points
    p: list of coordinates (x,y)'''
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def triangle_min_height(vertex):
    """
    Returns the smallest height of the triangle using the vertexs.

    Args:
        vertex (list of tuples): list of coordinates of the vertexs.
    """
    
    v0 = np.array(vertex[0])
    v1 = np.array(vertex[1])
    v2 = np.array(vertex[2])
    a = distance(v1, v2)  
    b = distance(v0, v2) 
    c = distance(v0, v1)  
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    h1 = (2 * area) / a  
    h2 = (2 * area) / b  
    h3 = (2 * area) / c  
    return min(h1, h2, h3)

def circ_sample(x, y, radius, margin=0.05):
    while True:
        angle = random.uniform(0, 2 * np.pi)
        r = random.uniform(0, radius)
        dx = r * np.cos(angle)
        dy = r * np.sin(angle)
        distance = np.sqrt(dx**2 + dy**2)
        if distance <= radius * (1 - margin):
            return (x + dx, y + dy)

def latlon_to_radians(lat, lon):
    """Convierte las coordenadas de latitud y longitud a radianes"""
    return np.radians(lat), np.radians(lon)

def spherical_distance(lat1, lon1, lat2, lon2):
    """Calcula la distancia esférica entre dos puntos dados en latitud y longitud"""
    lat1, lon1 = latlon_to_radians(lat1, lon1)
    lat2, lon2 = latlon_to_radians(lat2, lon2)
    delta_lon = lon2 - lon1
    return np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(delta_lon))

def spherical_area(vertices):
    """Calcula el área de un triángulo esférico dado por tres vértices en coordenadas de latitud y longitud"""
    lat1, lon1 = vertices[0]
    lat2, lon2 = vertices[1]
    lat3, lon3 = vertices[2]
    lat1, lon1 = latlon_to_radians(lat1, lon1)
    lat2, lon2 = latlon_to_radians(lat2, lon2)
    lat3, lon3 = latlon_to_radians(lat3, lon3)
    d1 = spherical_distance(lat1, lon1, lat2, lon2)
    d2 = spherical_distance(lat2, lon2, lat3, lon3)
    d3 = spherical_distance(lat3, lon3, lat1, lon1)

    s = (d1 + d2 + d3) / 2
    area_steradians = 4 * np.arctan(np.sqrt(np.tan(s / 2) * np.tan((s - d1) / 2) * np.tan((s - d2) / 2) * np.tan((s - d3) / 2)))
    area_degrees_squared = area_steradians * (180/np.pi)**2
    return area_degrees_squared



def get_object_size(obj):
    size = sys.getsizeof(obj) 
    name = []; memory = []
    print(f"Objeto: {obj}, Tamaño: {size} bytes")
    if hasattr(obj, '__dict__'):
        for attr_name, attr_value in obj.__dict__.items():
            attr_size = sys.getsizeof(attr_value)
            print(f"Atributo: {attr_name}, Tamaño: {attr_size} bytes")

            if isinstance(attr_value, object):
                memory.append(get_object_size(attr_value))
                name.append(attr_value)
    return name, memory



    
