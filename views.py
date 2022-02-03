import streamlit as st


class Component:
    def __init__(self, comp, *args, **kwargs) -> None:
        self.comp = comp
        self.args = args
        self.kwargs = kwargs

    def show(self) -> None:
        return self.comp(*self.args, **self.kwargs)


class Header():
    def __init__(self) -> None:
        self.title = Component(st.title, body="Prediksi Harga Emas")
        self.desc = Component(st.write, "Aplikasi untuk melakukan prediksi pada harga beli atau harga jual emas")
        self.comps = [self.title, self.desc]

    def build(self) -> None:
        for comp in self.comps:
            comp.show()
        

class Sidebar:
    def __init__(self) -> None:
        self.title = Component(st.subheader, body="Parameter")
        self.generation = Component(st.number_input, label="Jumlah Generasi", min_value=1, max_value=100, step=10)
        self.size = Component(st.number_input, label="Ukuran Populasi", min_value=10, max_value=1500, step=100)
        self.cr = Component(st.number_input, label="Crossover Rate", min_value=0.0, max_value=1.0, step=0.1)
        self.mr = Component(st.number_input, label="Mutation Rate", min_value=0.0, max_value=1.0, step=0.1)
        self.submit = Component(st.form_submit_button, label="Submit")

        self.comps = [self.generation, self.size, self.cr, self.mr]


    def build(self) -> dict:
        values = []
        with st.sidebar.form("Parameter"):
            self.title.show()
            for comp in self.comps:
                val = comp.show()
                values.append(val)
        
            is_submit = self.submit.show()

        if is_submit:
            st.write(values)