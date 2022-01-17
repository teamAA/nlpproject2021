import pandas as pd

def main():
    data = pd.read_csv('../data-registry/train.csv')

    chunk_length = 5000
    n_chunks = (len(data) // 5000) + 1

    for chunk in range(n_chunks):
        start = chunk*chunk_length
        end = start + chunk_length
        data_chunk = data.loc[start:end]

        data_chunk.to_csv(f"../data/train_{chunk}.csv", index = False)

if __name__ == "__main__":
    main()