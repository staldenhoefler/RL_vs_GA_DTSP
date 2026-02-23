import numpy as np
from typing import Tuple, List

class GASolver:
    """
    Genetic Algorithm solver for the Traveling Salesman Problem.
    Stateless with respect to the environment; it computes the TSP route
    for a given static distance matrix.
    """
    def __init__(self, pop_size: int = 100, n_generations: int = 500,
                 mutation_rate: float = 0.01, elite_size: int = 10,
                 tournament_size: int = 5):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size

    def solve(self, distance_matrix: np.ndarray) -> Tuple[List[int], float, List[float]]:
        """
        Runs the GA to find the optimal route.
        
        Parameters
        ----------
        distance_matrix : np.ndarray
            A 2D square array containing the distances between cities.
            
        Returns
        -------
        best_route : List[int]
            The best sequence of cities visited (starts and ends at 0).
        best_distance : float
            Total distance of the best_route.
        history : List[float]
            List of the best fitness (minimum distance) at each generation.
        """
        n_cities = distance_matrix.shape[0]
        # Population consists of permutations of cities 1 to n_cities-1.
        # City 0 is always the start/end point.
        pop = [np.random.permutation(np.arange(1, n_cities)).tolist() for _ in range(self.pop_size)]
        
        history = []
        best_overall_route = None
        best_overall_distance = float('inf')
        
        for generation in range(self.n_generations):
            # Calculate fitness
            fitnesses = [self._calculate_distance(ind, distance_matrix) for ind in pop]
            
            # Record best of generation
            min_dist = min(fitnesses)
            best_idx = fitnesses.index(min_dist)
            history.append(min_dist)
            
            if min_dist < best_overall_distance:
                best_overall_distance = min_dist
                best_overall_route = [0] + pop[best_idx] + [0]
                
            # Create next generation
            next_pop = []
            
            # Elitism
            elite_indices = np.argsort(fitnesses)[:self.elite_size]
            for idx in elite_indices:
                next_pop.append(pop[idx].copy())
                
            # Selection, Crossover, Mutation
            while len(next_pop) < self.pop_size:
                parent1 = self._tournament_selection(pop, fitnesses)
                parent2 = self._tournament_selection(pop, fitnesses)
                
                child = self._order_crossover(parent1, parent2)
                
                if np.random.rand() < self.mutation_rate:
                    self._swap_mutation(child)
                    
                next_pop.append(child)
                
            pop = next_pop

        return best_overall_route, best_overall_distance, history

    def _calculate_distance(self, route: List[int], distance_matrix: np.ndarray) -> float:
        """Calculate total distance of a route starting and ending at 0."""
        dist = 0.0
        curr = 0
        for nxt in route:
            dist += distance_matrix[curr, nxt]
            curr = nxt
        dist += distance_matrix[curr, 0]
        return dist

    def _tournament_selection(self, pop: List[List[int]], fitnesses: List[float]) -> List[int]:
        """Select best individual from a random subset."""
        indices = np.random.choice(len(pop), self.tournament_size, replace=False)
        best_idx = indices[np.argmin([fitnesses[i] for i in indices])]
        return pop[best_idx]

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order Crossover (OX1)."""
        size = len(parent1)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        
        child = [-1] * size
        # Copy subset from parent1
        child[start:end+1] = parent1[start:end+1]
        
        # Fill remaining elements from parent2
        p2_idx = 0
        for i in range(size):
            if child[i] == -1:
                while parent2[p2_idx] in child:
                    p2_idx += 1
                child[i] = parent2[p2_idx]
                p2_idx += 1
                
        return child

    def _swap_mutation(self, route: List[int]):
        """Swap two random positions."""
        idx1, idx2 = np.random.choice(len(route), 2, replace=False)
        route[idx1], route[idx2] = route[idx2], route[idx1]
