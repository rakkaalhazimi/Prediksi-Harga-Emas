from abc import abstractmethod
import json
import streamlit as st


class ViewElement:

    @abstractmethod
    def build(self):
        ...


class Component:
    def __init__(self, comp, *args, **kwargs) -> None:
        self.comp = comp
        self.args = args
        self.kwargs = kwargs

    def show(self) -> None:
        return self.comp(*self.args, **self.kwargs)


class Header(ViewElement):
    def __init__(self, title, desc) -> None:
        self.title = Component(st.title, body=title)
        self.desc = Component(st.write, desc)
        self.comps = [self.title, self.desc]

    def build(self) -> None:
        for comp in self.comps:
            comp.show()


class InfoBoard(ViewElement):
    def __init__(self, title, desc) -> None:
        self.title = Component(st.subheader, body=title)
        self.desc = Component(st.write, desc)
        self.comps = [self.title, self.desc]

    def build(self) -> None:
        st.markdown("#")
        for comp in self.comps:
            comp.show()
        st.markdown("---")


class InfoBoardWithButton(ViewElement):
    def __init__(self, title, desc, btn_label) -> None:
        self.title = Component(st.subheader, body=title)
        self.desc = Component(st.write, desc)
        self.button = Component(st.button, btn_label)
        self.comps = [self.title, self.desc]

    def build(self) -> None:
        st.markdown("#")
        for comp in self.comps:
            comp.show()

        is_train = self.button.show()
        st.markdown("---")
        return is_train


class PreParam(ViewElement):
    def __init__(self) -> None:
        self.title = Component(st.subheader, body="Setelan Jumlah Data Test")
        self.desc = Component(st.write, 
        """Tentukan proporsi antara jumlah data latih dan data test.
        """
        )
        self.test_size = Component(st.number_input, label="Ukuran Data Test", min_value=0.1, max_value=0.5, step=0.05)
        self.submit = Component(st.form_submit_button, label="Konfirmasi")

        self.comps = [self.test_size]
        self.pnames = ["test_size"]

    def build(self) -> dict:
        st.markdown("#")
        self.title.show()
        self.desc.show()

        params = {}
        with st.form("ParameterTest"):
            for key, comp in zip(self.pnames, self.comps):
                val = comp.show()
                params[key] = val
        
            is_submit = self.submit.show()

        if is_submit:
            return params


class GAParam(ViewElement):
    def __init__(self) -> None:
        self.title = Component(st.subheader, body="Setelan Algoritma Genetika")
        self.desc = Component(st.write, 
        """Tentukan nilai parameter untuk melatih model 
        menggunakan metode regresi linier dengan optimalisasi algoritma genetika.
        Setelah parameter ditentukan, klik 'Konfirmasi'.
        """
        )
        self.generation = Component(st.number_input, label="Jumlah Generasi", min_value=10, max_value=100, step=10)
        self.size = Component(st.number_input, label="Ukuran Populasi", min_value=100, max_value=1500, step=100)
        self.cr = Component(st.number_input, label="Crossover Rate", min_value=0.0, max_value=1.0, step=0.1)
        self.mr = Component(st.number_input, label="Mutation Rate", min_value=0.0, max_value=1.0, step=0.1)
        self.submit = Component(st.form_submit_button, label="Konfirmasi")

        self.comps = [self.generation, self.size, self.cr, self.mr]
        self.pnames = ["n_gen", "size", "cr", "mr"]

    def build(self) -> dict:
        st.markdown("#")
        self.title.show()
        self.desc.show()

        params = {}
        with st.form("ParameterGA"):
            for key, comp in zip(self.pnames, self.comps):
                val = comp.show()
                params[key] = val
        
            is_submit = self.submit.show()

        if is_submit:
            return params