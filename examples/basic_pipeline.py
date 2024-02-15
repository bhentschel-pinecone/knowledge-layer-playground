from kl_ingest.data_ingest import DataLoader

if __name__ == "__main__":
    data_loader = DataLoader()
    df_files = data_loader.parquet_folder_to_pandas("/Users/brianhentschel/data/miracl/corpus-parquet/")
    print(df_files[1].head())

