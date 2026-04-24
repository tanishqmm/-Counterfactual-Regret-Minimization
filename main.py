from cybersecurity_ssg import CybersecuritySSG
from analysis import evaluate_and_compare


if __name__ == "__main__":
    print("Initializing Cybersecurity SSG (with corrected Stackelberg payoff)...")
    game = CybersecuritySSG()
    evaluate_and_compare(game)
