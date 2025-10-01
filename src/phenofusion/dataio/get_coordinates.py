import pandas as pd


if __name__ == "__main__":
    directory_path = "/burg/glab/users/al4385/data/CSIFMETEO/"
    # Assuming 'filename.parquet' is the path to your Parquet file
    filename = directory_path + "merged_BDT_1982_2021.parquet"
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(filename)
    df = df[["location", "latitude", "longitude"]]
    df.drop_duplicates()
    df.to_parquet("/burg/glab/users/al4385/data/coordinates/merged_BDT.parquet")
