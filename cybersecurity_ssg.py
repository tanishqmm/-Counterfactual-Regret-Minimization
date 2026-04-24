import numpy as np
from itertools import product
from game_model import (DEFENDER_RESOURCES, N_TARGETS, N_ATTACKER_TYPES)
from utilities import compute_utilities


# GAME MECHANICS  

class CybersecuritySSG:
    """
    Stackelberg Security Game for Cybersecurity.
    """

    def __init__(self):
        self.n = N_TARGETS
        self.k = N_ATTACKER_TYPES
        (self.u_d_c, self.u_d_u,
         self.u_a_c, self.u_a_u) = compute_utilities()
        self.schedules = self._build_schedules()

    def _build_schedules(self):
        resource_options = []
        for r_id, res in DEFENDER_RESOURCES.items():
            options = res["can_cover"] + [-1]
            resource_options.append(options)

        schedules = []
        raw_assignments = []
        for assignment in product(*resource_options):
            coverage = np.zeros(self.n)
            for r_id, target_idx in enumerate(assignment):
                if target_idx >= 0:
                    quality = DEFENDER_RESOURCES[r_id]["coverage_quality"]
                    coverage[target_idx] = 1 - ((1 - coverage[target_idx])
                                                * (1 - quality))
            schedules.append(coverage)
            raw_assignments.append(assignment)

        unique = []
        unique_assignments = []
        seen = set()
        for s, a in zip(schedules, raw_assignments):
            key = tuple(np.round(s, 4))
            if key not in seen:
                seen.add(key)
                unique.append(s)
                unique_assignments.append(a)

        self.assignments = unique_assignments
        return np.array(unique)

    def defender_utility(self, target_i, p):
        return self.u_d_c[target_i] * p[target_i] + self.u_d_u[target_i] * (1 - p[target_i])

    def attacker_utility(self, attacker_j, target_i, p):
        return self.u_a_c[attacker_j, target_i] * p[target_i] + self.u_a_u[attacker_j, target_i] * (1 - p[target_i])

    def attacker_best_response(self, attacker_j, p):
        utilities = [self.attacker_utility(attacker_j, i, p) for i in range(self.n)]
        return int(np.argmax(utilities))

    def expected_coverage(self, mixed_strategy):
        """p_σ = Σ σ_a · schedule_a   (coverage induced by mixed strategy)."""
        return mixed_strategy @ self.schedules

    def committed_best_response(self, attacker_j, mixed_strategy):
        """Attacker's best target given the COMMITTED mixed strategy."""
        p_sigma = self.expected_coverage(mixed_strategy)
        return self.attacker_best_response(attacker_j, p_sigma)

    def defender_payoff_vs_committed(self, schedule_a_idx, attacker_j,
                                     mixed_strategy):
        i_star = self.committed_best_response(attacker_j, mixed_strategy)
        p_a = self.schedules[schedule_a_idx]
        return self.defender_utility(i_star, p_a)

    def defender_payoff_pure(self, p, attacker_j):
        target = self.attacker_best_response(attacker_j, p)
        return self.defender_utility(target, p)
