# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

pclass_d = {0:"Pierwsza",1:"Druga", 2:"Trzecia"}
embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
sex_d = {0:"Female", 1:"Male"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

	st.set_page_config(page_title="PUM5")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://th.bing.com/th/id/R.1f6fb205e3ca4554cd1fc2a473b17564?rik=n6MCX%2f%2f%2bhIfW2w&riu=http%3a%2f%2fwww.zastavki.com%2fpictures%2foriginals%2f2013%2fDrawn_wallpapers___Paintings___Titanic_on_the_berth_042610_.jpg&ehk=iGchnjY911Ei%2fq2F6OWO%2f9zWfYApBZPc%2bvvr5tK%2fjz0%3d&risl=&pid=ImgRaw&r=0")

	with overview:
		st.title("BUY A TICKET")

	with left:
		sex_radio = st.radio( "Płeć", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
		embarked_radio = st.radio( "Port zaokrętowania", list(embarked_d.keys()), index=2, format_func= lambda x: embarked_d[x] )
		pclass_radio = st.radio( "Klasa", list(pclass_d.keys()), format_func= lambda x: pclass_d[x])

	with right:
		age_slider = st.slider("Wiek", value=1, min_value=1, max_value=80)
		sibsp_slider = st.slider("Liczba rodzeństwa i/lub partnera", min_value=0, max_value=8)
		parch_slider = st.slider("Liczba rodziców i/lub dzieci", min_value=0, max_value=8)
		fare_slider = st.slider("Cena biletu", min_value=0, max_value=512, step=1)

	data = [[pclass_radio, sex_radio,  age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba przeżyłaby katastrofę?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
