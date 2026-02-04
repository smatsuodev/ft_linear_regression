import sys
import numpy as np
import matplotlib.pyplot as plt


def estimate_price(theta0, theta1, mileage):
    return theta0 + theta1 * mileage


def plot(t0, t1, **kwargs):
    x = np.linspace(0, 250000, 100)
    y = estimate_price(t0, t1, x)
    plt.plot(
        x,
        y,
        color=kwargs["color"] if "color" in kwargs else "blue",
        label=kwargs["label"] if "label" in kwargs else None,
    )


def main():
    args = sys.argv
    if len(args) not in [2, 3]:
        print("Usage: python learn.py <training_data_file> [<model_file>]")
        return 1

    model_path = args[2] if len(args) == 3 else ".model"

    try:
        with open(model_path) as f:
            t0, t1 = list(map(float, f.read().split(",")))
    except FileNotFoundError:
        print("Model file not found. Creating a new one with default parameters.")
        t0, t1 = 0.0, 0.0
        with open(model_path, "w") as f:
            f.write("0,0")
    except PermissionError:
        print("Cannot read model file.")
        return 1
    except ValueError:
        print("Invalid model.")
        return 1

    training_data_path = args[1]
    try:
        with open(training_data_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Training data file '{training_data_path}' not found.")
        return 1

    try:
        if len(lines) == 0 or lines[0] != "km,price\n":
            print("Invalid training data header.")
            return 1

        training_data = np.array(
            list(list(map(float, line.strip().split(","))) for line in lines[1:])
        )
    except ValueError:
        print("Invalid training data format.")
        return 1

    mean = training_data.mean(axis=0)  # [mean_mileage, mean_price]
    std = training_data.std(axis=0)  # [std_mileage, std_price]
    mean_x, mean_y = mean[0], mean[1]
    std_x, std_y = std[0], std[1]

    normalized_data = (training_data - mean) / std
    learning_rate = 0.1
    n_milages, n_prices = np.split(normalized_data, 2, axis=1)
    m = float(len(normalized_data))
    threshold = 0.0001

    while True:
        g0 = np.sum(estimate_price(t0, t1, n_milages) - n_prices) / m
        g1 = (
            np.sum(np.multiply(estimate_price(t0, t1, n_milages) - n_prices, n_milages))
            / m
        )

        if (
            np.isinf(g0)
            or np.isinf(g1)
            or np.isneginf(g0)
            or np.isneginf(g1)
            or np.isnan(g0)
            or np.isnan(g1)
        ):
            print("Gradient computation resulted in ambiguous.")
            return 1

        if abs(g0) < threshold and abs(g1) < threshold:
            break

        tmp0 = learning_rate * g0
        tmp1 = learning_rate * g1

        t0, t1 = t0 - tmp0, t1 - tmp1

    t1 = t1 * (std_y / std_x)
    t0 = (t0 * std_y + mean_y) - (t1 * mean_x)
    milages, prices = np.split(training_data, 2, axis=1)

    plot(t0, t1, color="green", label="Trained model")
    plt.scatter(
        training_data[:, 0],
        training_data[:, 1],
        color="black",
        label="Training data",
    )
    plt.xlabel("mileage")
    plt.ylabel("price")
    plt.legend()
    plt.show()

    try:
        with open(model_path, "w") as f:
            f.write(f"{t0},{t1}")
    except PermissionError:
        print("Cannot write to model file.")
        return 1

    errors = np.abs(estimate_price(t0, t1, milages) - prices)
    precision = np.sum(1 - errors / prices) / m
    print(f"precision: {precision * 100}%")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
