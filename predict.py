def main():
    try:
        with open(".model") as f:
            theta0,theta1 = map(float,f.read().split(','))

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
