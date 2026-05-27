import numpy as np

class GeneticAlgorithm:
    def __init__(self, pop_size=100, elite_size=10, mutation_rate=0.1, max_generations=200):
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        
    def _create_initial_population(self, n, depot=0):
        population = []
        nodes = [i for i in range(n) if i != depot]
        for _ in range(self.pop_size):
            perm = np.random.permutation(nodes).tolist()
            population.append(perm)
        return population

    def _crossover_pmx(self, p1, p2):
        # Partially Mapped Crossover (PMX) minimal implementation
        size = len(p1)
        p1, p2 = p1.copy(), p2.copy()
        cx1, cx2 = sorted(np.random.choice(size, 2, replace=False))
        
        c1, c2 = [-1]*size, [-1]*size
        c1[cx1:cx2+1] = p1[cx1:cx2+1]
        c2[cx1:cx2+1] = p2[cx1:cx2+1]
        
        def _fill(c, p, start, end):
            p_slice = p[start:end+1]
            c_slice = c[start:end+1]
            for i in range(size):
                if start <= i <= end: continue
                val = p[i]
                while val in c_slice:
                    val = p_slice[c_slice.index(val)]
                c[i] = val
        
        _fill(c1, p2, cx1, cx2)
        _fill(c2, p1, cx1, cx2)
        return c1, c2
        
    def _mutate_swap(self, perm):
        if np.random.rand() < self.mutation_rate:
            idx1, idx2 = np.random.choice(len(perm), 2, replace=False)
            perm[idx1], perm[idx2] = perm[idx2], perm[idx1]
        return perm

    def solve(self, simulator, evaluator_fn, use_wandb=False, initial_population=None):
        n = simulator.inst.coords.shape[0]
        depot = simulator.inst.depot
        
        if initial_population is not None:
            population = [list(p) for p in initial_population]
            # Fill remaining spots with random if given population is too small
            if len(population) < self.pop_size:
                needed = self.pop_size - len(population)
                filler = self._create_initial_population(n, depot)
                population.extend(filler[:needed])
            # Truncate if too large
            elif len(population) > self.pop_size:
                population = population[:self.pop_size]
        else:
            population = self._create_initial_population(n, depot)
        
        best_cost = float('inf')
        best_tour = None
        
        for generation in range(self.max_generations):
            costs = evaluator_fn(simulator, population)
            
            # Find best
            min_idx = np.argmin(costs)
            if costs[min_idx] < best_cost:
                best_cost = costs[min_idx]
                best_tour = population[min_idx]
                full_current_tour = [depot] + best_tour + [depot]
                print(f"[GA] New cost minima achieved: {best_cost:.2f} | Tour list: {full_current_tour}")
                if use_wandb:
                    import wandb
                    wandb.log({
                        "new_minima/cost": best_cost,
                        "new_minima/tour_list": str(full_current_tour)
                    }, commit=False)
                
            if use_wandb:
                import wandb
                wandb.log({
                    "generation": generation,
                    "best_cost": best_cost,
                    "gen_min_cost": costs[min_idx],
                    "gen_avg_cost": np.mean(costs)
                })
                
            # Selection (Tournament-like sorting)
            sorted_indices = np.argsort(costs)
            elites = [population[i] for i in sorted_indices[:self.elite_size]]
            
            new_population = elites.copy()
            # Crossover
            while len(new_population) < self.pop_size:
                p1_idx = np.random.choice(sorted_indices[:self.pop_size//2])
                p2_idx = np.random.choice(sorted_indices[:self.pop_size//2])
                
                c1, c2 = self._crossover_pmx(population[p1_idx], population[p2_idx])
                
                c1 = self._mutate_swap(c1)
                c2 = self._mutate_swap(c2)
                
                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)
                    
            population = new_population
            
        # Final eval pass to ensure cost is accurately updated for last generation
        costs = evaluator_fn(simulator, population)
        min_idx = np.argmin(costs)
        if costs[min_idx] < best_cost:
            best_cost = costs[min_idx]
            best_tour = population[min_idx]
            full_current_tour = [depot] + best_tour + [depot]
            print(f"[GA] Final evaluation new cost minima achieved: {best_cost:.2f} | Tour list: {full_current_tour}")
            if use_wandb:
                import wandb
                wandb.log({
                    "new_minima/cost": best_cost,
                    "new_minima/tour_list": str(full_current_tour)
                }, commit=False)

        full_best_tour = [depot] + best_tour + [depot]
        return full_best_tour, best_cost
