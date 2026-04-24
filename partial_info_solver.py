import numpy as np


# PARTIAL INFORMATION SOLVER  

class PartialInfoSolver:
    def __init__(self, game):
        self.game = game
        self.n_actions = len(game.schedules)
        self.cumulative_regret   = np.zeros(self.n_actions)
        self.cumulative_strategy = np.zeros(self.n_actions)
        self.probe_strategies = self._find_probe_strategies()

    def _find_probe_strategies(self):
        k = self.game.k
        probe_strategies = []
        for j in range(k):
            best_idx, best_uniqueness, best_target = None, -1, 0
            for idx, schedule in enumerate(self.game.schedules):
                j_target = self.game.attacker_best_response(j, schedule)
                uniqueness = sum(
                    1 for j2 in range(k)
                    if j2 != j and
                    self.game.attacker_best_response(j2, schedule) != j_target
                )
                if uniqueness > best_uniqueness:
                    best_uniqueness, best_idx, best_target = uniqueness, idx, j_target
            probe_strategies.append((best_idx, best_target))
        return probe_strategies

    def get_strategy(self):
        pos = np.maximum(self.cumulative_regret, 0)
        total = pos.sum()
        return pos / total if total > 0 else np.ones(self.n_actions) / self.n_actions

    def get_average_strategy(self):
        total = self.cumulative_strategy.sum()
        return (self.cumulative_strategy / total
                if total > 0 else np.ones(self.n_actions) / self.n_actions)

    def solve(self, T, attacker_sequence=None, verbose=True):
        n, k = self.game.n, self.game.k
        if attacker_sequence is None:
            attacker_sequence = [t % k for t in range(T)]

        log_nk = max(1.0, np.log(n * k))
        Z_float = n * (T**2 * log_nk) ** (1.0 / 3.0)
        Z = max(1, min(int(Z_float), T // k))
        block_size = max(k, T // Z)
        Z = T // block_size

        if verbose:
            print(f"  [Partial Info] T={T}, Z={Z} blocks, block_size={block_size}")

        utility_history = []
        regret_history = []
        cumulative_utility = 0.0
        log_interval = max(1, Z // 20)

        for tau in range(Z):
            block_start = tau * block_size
            block_end = min(block_start + block_size, T)
            block_T = block_end - block_start
            if block_T <= 0:
                break

            strategy = self.get_strategy()
            explore_set = set(np.random.choice(block_T, size=k, replace=False))
            probe_perm = np.random.permutation(k)
            explore_list = sorted(explore_set)
            probe_obs = {}

            for t_local in range(block_T):
                t_abs = block_start + t_local
                if t_abs >= T:
                    break
                attacker = attacker_sequence[t_abs]

                if t_local in explore_set:
                    probe_rank = explore_list.index(t_local)
                    probe_j = probe_perm[probe_rank % k]
                    sched_idx, watch_target = self.probe_strategies[probe_j]
                    p_probe = self.game.schedules[sched_idx]
                    attacked_target = self.game.attacker_best_response(
                        attacker, p_probe)
                    probe_obs[probe_j] = 1 if attacked_target == watch_target else 0
                    util = self.game.defender_utility(attacked_target, p_probe)
                else:
                    
                    i_star = self.game.committed_best_response(attacker, strategy)
                    played_a = np.random.choice(self.n_actions, p=strategy)
                    util = self.game.defender_utility(
                        i_star, self.game.schedules[played_a])

                cumulative_utility += util

            freq_estimates = np.array([probe_obs.get(j, 0) * block_T
                                       for j in range(k)], dtype=float)
            freq_total = freq_estimates.sum()
            est_dist = (freq_estimates / freq_total
                        if freq_total > 0 else np.ones(k) / k)

            estimated_losses = np.zeros(self.n_actions)
            for j in range(k):
                i_star_j = self.game.committed_best_response(j, strategy)
                for a in range(self.n_actions):
                    u = self.game.defender_utility(i_star_j, self.game.schedules[a])
                    estimated_losses[a] += -est_dist[j] * u

            expected_loss = strategy @ estimated_losses
            self.cumulative_regret += -(estimated_losses - expected_loss)
            self.cumulative_strategy += strategy

            t_done = (tau + 1) * block_size
            avg_regret = np.max(np.maximum(self.cumulative_regret, 0)) / (tau + 1)
            regret_history.append((t_done, avg_regret))

            if verbose and (tau + 1) % log_interval == 0:
                avg_utility = cumulative_utility / max(1, t_done)
                utility_history.append((t_done, avg_utility))
                print(f"  Block {tau+1:>5d}/{Z} | T={t_done:>7d} | "
                      f"Avg utility: {avg_utility:>8.5f}")

        return self.get_average_strategy(), regret_history, utility_history
