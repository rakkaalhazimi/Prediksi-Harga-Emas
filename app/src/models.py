import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

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
def gen_algo(size, n_gen, X_train, y_train, cr=0.9, mr=0.5):
    print("START")
    # Hitung jumlah fitur
    n_feat = X_train.shape[1]

    # Inisiasi model regresi
    linreg = LinearRegression()

    # Bangkitkan kromosom secara acak
    population = create_population(size, n_feat)

    for i in range(n_gen):
        # Hitung skor fitness masing-masing kromosom
        population, fitness = get_fitness(population, X_train, y_train)

        # Catat populasi dan fitness setiap 10 iterasi
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}")
            print("=" * 20)
            print(f"Best chromosome:\n{population[0]}")
            print(f"Best fitness:\n{fitness[0]}")
            print()

        # Simpan 2 kromosom terbaik untuk generasi berikutnya
        next_gen = list(population[:2])

        for i in range( int(size / 2) - 1 ):
            
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

    # Optimasi parameter regresi
    linreg.intercept_ = population[0][0:1]
    linreg.coef_ = population[0][1:].reshape(1, -1)

    return population, fitness, linreg