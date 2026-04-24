import numpy as np
from game_model import (TARGETS, ATTACKER_TYPES, N_TARGETS, N_ATTACKER_TYPES)


# UTILITY COMPUTATION  

def compute_utilities():
    cvss_map = {"High": 0.56, "Low": 0.22, "None": 0.0}
    max_value = max(t["asset_value_millions"] for t in TARGETS.values())

    motivation_weights = {
        "notoriety":  {"C": 0.2, "I": 0.3, "A": 0.5},
        "espionage":  {"C": 0.8, "I": 0.1, "A": 0.1},
        "financial":  {"C": 0.6, "I": 0.2, "A": 0.2},
        "ransom":     {"C": 0.1, "I": 0.2, "A": 0.7},
        "disruption": {"C": 0.1, "I": 0.3, "A": 0.6},
        "data_theft": {"C": 0.7, "I": 0.2, "A": 0.1},
    }

    u_d_covered = np.zeros(N_TARGETS)
    u_d_uncovered = np.zeros(N_TARGETS)
    for i, target in TARGETS.items():
        detection_cost = 0.1 + 0.05 * (target["asset_value_millions"] / max_value)
        u_d_covered[i] = -detection_cost
        u_d_uncovered[i] = -(target["asset_value_millions"] / max_value)

    u_a_covered = np.zeros((N_ATTACKER_TYPES, N_TARGETS))
    u_a_uncovered = np.zeros((N_ATTACKER_TYPES, N_TARGETS))
    for j, attacker in ATTACKER_TYPES.items():
        mw = motivation_weights[attacker["motivation"]]
        for i, target in TARGETS.items():
            c_val = cvss_map[target["cvss_confidentiality"]]
            i_val = cvss_map[target["cvss_integrity"]]
            a_val = cvss_map[target["cvss_availability"]]
            weighted_impact = mw["C"]*c_val + mw["I"]*i_val + mw["A"]*a_val
            target_attractiveness = (weighted_impact * attacker["capability"]
                                     * (target["asset_value_millions"] / max_value))
            u_a_uncovered[j, i] = target_attractiveness
            risk_cost = (1 - attacker["risk_tolerance"]) * attacker["capability"]
            u_a_covered[j, i] = -risk_cost * (0.3 + 0.2 * weighted_impact)

    return u_d_covered, u_d_uncovered, u_a_covered, u_a_uncovered
