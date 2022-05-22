import streamlit as st
import pandas as pd

def verify_data(df):
    # Check Column Names
    try:
        assert "HargaBeli" in df.columns
        assert "HargaJual" in df.columns
        assert "Date" in df.columns
    except AssertionError:
        st.warning("""
        Tidak ada kolom HargaBeli, HargaJual atau Date, pastikan nama kolom sesuai dengan yang diberitahukan. 
        Nama kolom saat ini {}
        """.format(",".join(df.columns)))
        return False

    # Check Data Types
    try:
        df["HargaBeli"].astype(float)
        df["HargaJual"].astype(float)
    except ValueError:
        st.warning("Tipe data tidak sesuai, pastikan data dalam bentuk angka.")
        return False

    # Check Date Formats
    try:
        pd.to_datetime(df["Date"])
    except ValueError:
        st.warning("""
        Format tanggal dalam kolom 'Date' salah, seharusnya DD/MM/YYYY, namun yang didapat {}
        """.format(df["Date"].iloc[0])
        )
        return False

    # Check Data Length
    length = len(df)
    if length < 20:
        st.warning("""
        Dataset kurang dari 20 sampel, dataset saat ini berjumlah {} sampel.
        """.format(length)
        )
        return False
        
    st.info("Data berhasil diproses")
    return True