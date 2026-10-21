import streamlit as st
import joblib
import numpy as np


model = joblib.load('immo_modell.pkl')

st.title("🏠 KI-Immobilien-Preisschätzer")
st.write("Gib die Eigenschaften der Immobilie ein, um eine Schätzung zu erhalten:")


qualitaet = st.slider("Gesamtzustand / Qualität (1 = schlecht, 10 = Luxus)", 1, 10, 5)
flaeche = st.number_input("Wohnfläche in qm / sqft", value=150)
zimmer = st.number_input("Anzahl Schlafzimmer", value=3)
baeder = st.number_input("Anzahl Badezimmer", value=2)


if st.button("Preis schätzen"):
    eingabe = np.array([[flaeche, zimmer, baeder, qualitaet]])
    vorhersage = model.predict(eingabe)[0]
    st.success(f"💰 Geschätzter Wert: ${vorhersage:,.2f}")
