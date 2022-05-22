class Config:
    DATA_PATH = "app/data/new_data.csv"
    
    SHIFT = 60
    DATE_COL = "Date"
    DATE_FORMAT = "%d/%m/%Y"
    
    RESPONSE_VAR = "Harga"
    BUY_MODE = "beli"
    SELL_MODE = "jual"

    PARAMS_KEY = ["dataset_type", "test_size", "n_gen", "size", "cr", "mr"]
    PARAMS_LABLE = [
        "Tipe Dataset", 
        "Ukuran Data Test", 
        "Jumlah Generasi", 
        "Ukuran Populasi", 
        "Crossover Rate", 
        "Mutation Rate"
    ]