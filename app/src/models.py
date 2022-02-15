from sklearn.metrics import r2_score, mean_squared_error
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


session = st.session_state
prediction_columns = ["Y_test", "MLR Without Genetic", "MLR With Genetic"]
error_columns = ["Error MSE MLR", "Error MSE MLR+Genetic", "Error RMSE MLR", "Error RMSE MLR+Genetic"]


def create_population(size, n_feat):
    population = np.random.uniform(low=-1., high=1., size=(size, n_feat + 1))
    return population


def get_fitness(population, X_train, y_train):
    constant = np.ones(shape=(X_train.shape[0], 1))
    X_train = np.concatenate((constant, X_train), axis=1)
    
    # Lakukan prediksi dengan melakukan perkalian matriks antara koefisien dan data
    predictions = np.matmul(X_train, population.T)
    
    # Hitung nilai fitness
    mse = np.mean((predictions - np.array(y_train))**2, axis=0)
    fitness = 1 / mse

    # urutkan dari terbesar -> terkecil
    inds = np.argsort(fitness)[::-1]  

    return population[inds], fitness[inds]


def selection_pair(population, fitness):
    length = len(population)
    probabilities = fitness / np.sum(fitness)
    chromosome_index = np.arange(length)
    selection_index = np.random.choice(chromosome_index, size=2, p=probabilities)
    return population[selection_index]


def crossover(chrom1, chrom2, alpha=0.4, cr=0.9):
    odds = np.random.random()
        
    if odds < cr:
        chrom1 = alpha * chrom1 + (1 - alpha) * chrom2
        chrom2 =  alpha * chrom2 + (1 - alpha) * chrom1

    return chrom1, chrom2


def mutation(chrom, mutation_rate=0.9):
    length = len(chrom)
    mutation_size = int(mutation_rate * length)
    
    random_gene = np.random.randint(0, length, size=mutation_size)
    mutated_gene = np.random.uniform(-1., 1., size=mutation_size)
    chrom[random_gene] = mutated_gene

    return chrom


@st.cache(suppress_st_warning=True)
def gen_algo(size, n_gen, X_train, y_train, cr=0.9, mr=0.5, mode=None):
    print("START")
    # Hitung jumlah fitur
    n_feat = X_train.shape[1]

    # Inisiasi model regresi
    linreg = LinearRegression()

    # Bangkitkan kromosom secara acak
    population = create_population(size, n_feat)

    # Sematkan progress bar
    st.write("Regresi Linier GA pada harga {}".format(mode))
    train_bar = st.progress(0.0)

    for iter_ in range(n_gen):
        # Hitung skor fitness masing-masing kromosom
        population, fitness = get_fitness(population, X_train, y_train)

        # Catat populasi dan fitness setiap 10 iterasi
        if (iter_ + 1) % 10 == 0:
            print(f"Iteration {iter_ + 1}")
            print("=" * 20)
            print(f"Best chromosome:\n{population[0]}")
            print(f"Best fitness:\n{fitness[0]}")
            print()

        # Simpan 2 kromosom terbaik untuk generasi berikutnya
        next_gen = list(population[:2])

        for _ in range( int(size / 2) - 1 ):
            
            # Seleksi 2 kromosom secara acak
            parents_a, parents_b = selection_pair(population, fitness)

            # Kawin silang pada 2 induk dan menghasilkan 2 keturunan
            offspring_a, offspring_b = crossover(parents_a, parents_b, cr)

            # Mutasi pada hasil keturunan
            offspring_a = mutation(offspring_a, mr)
            offspring_b = mutation(offspring_b, mr)

            # Ikut sertakan 2 keturunan tadi ke generasi berikutnya
            next_gen.append(offspring_a)
            next_gen.append(offspring_b)

        # Perbarui populasi lama dengan populasi baru
        population = np.array(next_gen)

        # Perbarui progress bar
        train_bar.progress( (iter_ + 1) / n_gen )

    # Optimasi parameter regresi
    linreg.intercept_ = population[0][0:1]
    linreg.coef_ = population[0][1:].reshape(1, -1)

    return population, fitness, linreg


def evaluate(model, mode, ga=False):
    predictions = model.predict(session[mode]["X_test"])
    true = session[mode]["y_test"]

    r2 = r2_score(true, predictions)
    mse = mean_squared_error(true, predictions)
    rmse = mean_squared_error(true, predictions, squared=False)

    return {"r2": r2, "mse": mse, "rmse": rmse}





def predict_future(period, X, model, colname):
    predictions = []
    dates = []

    # Ambil indeks terakhir pada data
    feature = X.iloc[-1:].copy()
    for _ in range(period):
        # Prediksi pada masa depan
        prediction = model.predict(feature)[0][0]
        
        # Simpan tanggal dan hasil prediksi
        feature.index += pd.Timedelta(days=1)
        predictions.append(prediction)
        dates.append(feature.index[0])

        # Ganti fitur data dengan hasil prediksi sebelumnya
        feature = feature.shift(axis=1).replace(np.nan, prediction)

    return pd.DataFrame(predictions, columns=[colname], index=dates)


def combine_predictions(period, X_test, rekap, model, model_ga):
    
    rekap = rekap[prediction_columns + error_columns]

    # Dapatkan hasil prediksi pada masa depan
    prediksi_lanjut = predict_future(period=period, 
                                     X=X_test, 
                                     model=model,
                                     colname="MLR Without Genetic")

    prediksi_lanjut_ga = predict_future(period=period, 
                                        X=X_test, 
                                        model=model_ga,
                                        colname="MLR With Genetic")
    
    # Gabungkan hasil prediksi
    prediksi_lanjut_gabungan = pd.concat([prediksi_lanjut, prediksi_lanjut_ga], axis=1)
    dates_str = [date.strftime("%Y-%m-%d") for date in prediksi_lanjut_gabungan.index]
    prediksi_lanjut_gabungan.index = dates_str
    prediksi_df = rekap.append(prediksi_lanjut_gabungan)
    
    # DataFrame pada waktu tertentu
    prediksi_tertentu_df = prediksi_df.iloc[-period * 2:]

    return prediksi_tertentu_df


def prediction_date_based(date, X, y, model, model_ga):
    
    # Copy Dataframe
    pd_date = pd.to_datetime(date, format="%Y-%m-%d")
    X = X.copy()
    
    # Jika dalam jangkauan data
    if pd_date <= X.index[-1]:
        # Fitur data
        feature = X.loc[pd_date: pd_date + pd.Timedelta(days=1)]

        # Prediksi
        predictions = y.copy().loc[pd_date: pd_date + pd.Timedelta(days=0)]
        predictions.columns = ["Y_actual"]
        predictions["MLR Without GA"] = model.predict(feature)[0][0]
        predictions["MLR With Genetic"] = model_ga.predict(feature)[0][0]

    # Jika pada masa depan
    else:
        steps = pd_date - X.index[-1]
        steps = steps.days
        predictions = predict_future(steps, X, model, "MLR Without GA")
        predictions_ga = predict_future(steps, X, model_ga, "MLR With Genetic")
        predictions = pd.concat([predictions, predictions_ga], axis=1).iloc[-1:]

    predictions.index = [date.strftime("%Y-%m-%d") for date in predictions.index]

    return predictions