import sys


def main():
    args = sys.argv

    if len(args) not in [1, 2]:
        print("Usage: python predict.py [<model_file>]")
        return 1

    model_path = args[1] if len(args) == 2 else ".model"

    try:
        with open(model_path) as f:
            theta0, theta1 = map(float, f.read().split(","))

    except FileNotFoundError:
        print("Model file not found.")
        return 1

    except ValueError:
        print("Invalid model.")
        return 1

    try:
        mileage = float(input("mileage: "))
    except ValueError:
        print("Invalid mileage.")
        return 1

    price = theta0 + theta1 * mileage
    print(f"price: {price}")


if __name__ == "__main__":
    main()
