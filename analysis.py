import numpy as np
import time
import matplotlib.pyplot as plt
from game_model import (TARGETS, DEFENDER_RESOURCES, N_TARGETS, N_RESOURCES)
from cfr_solver import CFRSolver
from partial_info_solver import PartialInfoSolver


# ANALYSIS HELPERS 

def compute_exploitability(game, mixed_strategy, attacker_probs=None):
    if attacker_probs is None:
        attacker_probs = np.ones(game.k) / game.k

    total = 0.0
    for j in range(game.k):
        i_star = game.committed_best_response(j, mixed_strategy)
        u = sum(mixed_strategy[a]
                * game.defender_utility(i_star, game.schedules[a])
                for a in range(len(game.schedules)))
        total += attacker_probs[j] * u
    return total


def compute_resource_probabilities(game, mixed_strategy):
    resource_probs = {r: {t: 0.0 for t in list(range(game.n)) + [-1]}
                      for r in range(N_RESOURCES)}
    for s_idx, (x_s, assignment) in enumerate(
            zip(mixed_strategy, game.assignments)):
        if x_s < 1e-9:
            continue
        for r_id, target_idx in enumerate(assignment):
            resource_probs[r_id][target_idx] += x_s
    coverage = np.zeros(game.n)
    for s_idx in range(len(game.schedules)):
        coverage += mixed_strategy[s_idx] * game.schedules[s_idx]
    return resource_probs, coverage


def print_resource_analysis(game, mixed_strategy, solver_name=""):
    resource_probs, coverage = compute_resource_probabilities(game, mixed_strategy)

    print(f"\n{'='*70}")
    print(f"RESOURCE DEPLOYMENT ANALYSIS — {solver_name}")

    for r_id, res in DEFENDER_RESOURCES.items():
        p_idle = resource_probs[r_id].get(-1, 0.0)
        p_deploy = 1.0 - p_idle
        print(f"\n  Resource {r_id}: {res['name']}")
        print(f"    Coverage quality : {res['coverage_quality']:.0%}")
        print(f"    P(deployed)      : {p_deploy:.3f}   |   P(idle): {p_idle:.3f}")
        print(f"    ── Per-target deployment probability ──")
        for t in res["can_cover"]:
            p_t = resource_probs[r_id].get(t, 0.0)
            if p_t < 1e-5:
                continue
            contrib = p_t * res["coverage_quality"]
            bar = "█" * int(p_t * 20) + "░" * (20 - int(p_t * 20))
            print(f"      → t{t} {TARGETS[t]['name'][:32]:<32s} "
                  f"deploy prob {p_t:.3f}  [{bar}]")
            print(f"           coverage contribution ≈ {contrib:.3f}")

    print(f"\n  ── FINAL TARGET COVERAGE PROBABILITIES (P(target t defended)) ──")
    for t in range(N_TARGETS):
        bar = "█" * int(coverage[t] * 20) + "░" * (20 - int(coverage[t] * 20))
        print(f"    t{t} [{bar}] {coverage[t]:.3f}  {TARGETS[t]['name']}")


def evaluate_and_compare(game):
    print(f"\nTotal pure strategies (schedules): {len(game.schedules)}")

    T_runs = 5000
    np.random.seed(42)

    solvers_to_run = [
        ("CFRSolver",         CFRSolver(game)),
        ("PartialInfoSolver", PartialInfoSolver(game)),
    ]

    results = []
    regret_data = {}
    for name, solver in solvers_to_run:
        print(f"\n{'='*70}\nSOLVER: {name}\n{'='*70}")
        t0 = time.time()
        strat, regret_hist, _ = solver.solve(T_runs, verbose=False)
        elapsed = time.time() - t0
        util = compute_exploitability(game, strat)
        print(f"  Regret convergence: {regret_hist[-1][1]:.6f}")
        print(f"  Runtime: {elapsed:.3f}s | Defender utility: {util:+.5f}")
        print_resource_analysis(game, strat, name)
        results.append((name, strat, util, elapsed))
        regret_data[name] = regret_hist

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    for ax, (name, hist) in zip(axes, regret_data.items()):
        if len(hist) == 0:
            continue
        iters, regrets = zip(*hist)
        ax.plot(iters, regrets, linewidth=1.5)
        ax.set_title(f"{name} — Avg Regret Convergence")
        ax.set_xlabel("Iteration T")
        ax.set_ylabel("Avg Regret")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("regret_convergence.png", dpi=120)
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for name, hist in regret_data.items():
        if len(hist) == 0:
            continue
        iters, regrets = zip(*hist)
        ax2.plot(iters, regrets, linewidth=1.5, label=name)
    ax2.set_title("Regret Convergence Comparison")
    ax2.set_xlabel("Iteration T")
    ax2.set_ylabel("Avg Regret")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.savefig("regret_comparison.png", dpi=120)
    plt.show()
