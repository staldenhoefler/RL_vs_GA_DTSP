from typing import List, Union
import numpy as np
from .simulator import TDTSPSimulator

def evaluate_tour(simulator: TDTSPSimulator, permutation: Union[List[int], np.ndarray]) -> float:
    """
    Evaluates a single sequence of node visits mapping directly to the underlying simulation mechanics.
    """
    return simulator.evaluate_tour(permutation)

def evaluate_population(simulator: TDTSPSimulator, population: List[Union[List[int], np.ndarray]]) -> List[float]:
    """
    Evaluates a population block sequentially, extracting the objective deterministically 
    for evaluating offspring / chromosomes in a GA context.
    """
    costs = []
    for perm in population:
        costs.append(evaluate_tour(simulator, perm))
    return costs
