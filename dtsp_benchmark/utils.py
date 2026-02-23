import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List

def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Calculate the 2D Euclidean distance matrix for a given set of coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, 2) containing the (x, y) coordinates of N cities.

    Returns
    -------
    np.ndarray
        Array of shape (N, N) where the element at (i, j) is the
        Euclidean distance between city i and city j.
    """
    diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    return distances

def plot_tour(coords: np.ndarray, tour: List[int], ax: Axes, title: str = "TSP Tour"):
    """
    Plot the current points and the tour taken so far.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, 2) containing the (x, y) coordinates.
    tour : List[int]
        List of city indices representing the tour. 
        It can be a partial tour or a complete one.
    ax : matplotlib.axes.Axes
        The matplotlib axis to draw on.
    title : str
        Title of the plot.
    """
    ax.clear()
    ax.set_title(title)
    
    # Plot all cities
    ax.scatter(coords[:, 0], coords[:, 1], c='blue', marker='o', s=50, label='Cities')
    
    # Draw tour edges
    if len(tour) > 1:
        tour_coords = coords[tour]
        ax.plot(tour_coords[:, 0], tour_coords[:, 1], c='red', linestyle='-', linewidth=2, label='Tour')
        
        # Highlight start city
        ax.scatter(coords[tour[0], 0], coords[tour[0], 1], c='green', marker='s', s=100, label='Start')
        
        # Highlight current city if not at the start
        if len(tour) > 1 and tour[-1] != tour[0]:
            ax.scatter(coords[tour[-1], 0], coords[tour[-1], 1], c='orange', marker='*', s=150, label='Current')

    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
