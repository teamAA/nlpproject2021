import pandas as pd
import argparse

def main(n):
    base_data = pd.read_csv("../data-registry/train.csv")
    new_data = pd.read_csv(f"../data/train_{n}.csv")
    print(f"Old data shape: {base_data.shape}")
    print(f"New data shape: {new_data.shape}")

    print("Concatenating...")
    pd.concat([base_data, new_data]).to_csv("../data-registry/train.csv")
    print("New data has been written succesfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type = int)
    args = parser.parse_args()

    main(args.chunk)

