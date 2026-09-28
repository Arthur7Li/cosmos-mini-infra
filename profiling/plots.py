"""
Generates plots for the benchmarking report.
"""
import os

import matplotlib.pyplot as plt
import numpy as np


def generate_throughput_plot(output_dir: str) -> None:
    labels = ["Sync (Blocking)", "Async (Decoupled)"]
    throughput = [59.95, 63.98]

    _fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, throughput, color=["#1f77b4", "#2ca02c"])

    ax.set_ylabel("Throughput (Samples/sec)")
    ax.set_title("RL Synchronization Throughput (Mac M2 CPU)")
    ax.set_ylim(0, 80)

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 1,
            f"{yval:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput.png"), dpi=300)
    plt.close()


def generate_staleness_plot(output_dir: str) -> None:
    # Dummy simulated data for demonstration
    np.random.seed(42)
    steps = np.arange(0, 1000, 50)
    
    # Sync implies 0 staleness always, reward goes up steadily
    sync_reward = 100 * (1 - np.exp(-steps / 300))
    
    # Async implies some staleness, reward goes up slightly slower or has more variance initially
    async_reward = 100 * (1 - np.exp(-steps / 350)) + np.random.normal(0, 2, size=len(steps))

    _fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(steps, sync_reward, label="Sync (0 staleness)", color="#1f77b4", linewidth=2)
    ax.plot(
        steps, async_reward, label="Async (N steps stale)", color="#2ca02c", linewidth=2, linestyle="--"
    )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Dummy Reward")
    ax.set_title("Simulated Convergence: Sync vs Async")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "staleness_vs_reward.png"), dpi=300)
    plt.close()


def main() -> None:
    os.makedirs("docs/assets", exist_ok=True)
    generate_throughput_plot("docs/assets")
    generate_staleness_plot("docs/assets")
    print("Plots generated in docs/assets/")


if __name__ == "__main__":
    main()
