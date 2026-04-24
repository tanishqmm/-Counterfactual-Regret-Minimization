import numpy as np


# CFR SOLVER  

class CFRSolver:
    
    def __init__(self, game):
        self.game = game
        self.n_actions = len(game.schedules)
        self.cumulative_regret = np.zeros(self.n_actions)
        self.cumulative_strategy = np.zeros(self.n_actions)

    def get_strategy(self):
        positive_regret = np.maximum(self.cumulative_regret, 0)
        total = positive_regret.sum()
        if total > 0:
            return positive_regret / total
        return np.ones(self.n_actions) / self.n_actions

    def get_average_strategy(self):
        total = self.cumulative_strategy.sum()
        if total > 0:
            return self.cumulative_strategy / total
        return np.ones(self.n_actions) / self.n_actions

    def update(self, attacker_type):
        strategy = self.get_strategy()

        i_star = self.game.committed_best_response(attacker_type, strategy)
        action_utils = np.array([
            self.game.defender_utility(i_star, self.game.schedules[a])
            for a in range(self.n_actions)
        ])

        expected_utility = float(strategy @ action_utils)
        self.cumulative_regret += action_utils - expected_utility
        self.cumulative_strategy += strategy

    def solve(self, T, attacker_sequence=None, verbose=True):
        if attacker_sequence is None:
            attacker_sequence = [t % self.game.k for t in range(T)]

        regret_history = []
        utility_history = []
        cumulative_utility = 0.0

        for t in range(T):
            attacker = attacker_sequence[t]
            strategy = self.get_strategy()

            i_star = self.game.committed_best_response(attacker, strategy)
            played_a = np.random.choice(self.n_actions, p=strategy)
            cumulative_utility += self.game.defender_utility(
                i_star, self.game.schedules[played_a])

            self.update(attacker)

            avg_regret = np.max(np.maximum(self.cumulative_regret, 0)) / (t + 1)
            regret_history.append((t + 1, avg_regret))

            if (t + 1) % max(1, T // 20) == 0:
                avg_utility = cumulative_utility / (t + 1)
                utility_history.append((t + 1, avg_utility))
                if verbose:
                    print(f"  Iter {t+1:>7d}/{T} | "
                        f"Avg utility: {avg_utility:>8.5f}")

        return self.get_average_strategy(), regret_history, utility_history
