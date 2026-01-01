import numpy as np


def generate_dataset(
    seed: int = 9001,
    n: int = 365,
    low: int = -10,
    high: int = 35,
    optimal: float = 16.7,
    noise_scale: float = 4,
) -> np.ndarray:
    np.random.seed(seed)
    temperatures = np.random.uniform(low, high, n)
    energy = (
        50 + 2 * (temperatures - optimal) ** 2 + np.random.normal(0, noise_scale, n)
    )
    return np.stack([temperatures, energy])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x, y = generate_dataset(n=365)

    plt.scatter(x, y, marker="x")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Energy (MWh)")
    plt.title("Energy vs. Temperature")
    plt.grid(True)
    plt.show()
