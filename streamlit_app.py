import streamlit as st
import sentencepiece as spm
from fairseq.models.bart import BARTModel

#===========================================#
#                Load Model                 #
#===========================================#

@st.cache
def get_model():
    BASEDIR = './model'
    bart = BARTModel.from_pretrained(
            BASEDIR,
            checkpoint_file='checkpoint_best.pt',
            bpe='sentencepiece',
            sentencepiece_model=BASEDIR + '/sentence.bpe.model')
    bart.eval()
    return bart

#===========================================#
#                 Run Inference             #
#===========================================#

def summarize(full_text):
    summarized_text = get_model().sample([full_text], beam=7, lenpen=2.0, max_len_b=140, min_len=40, no_repeat_ngram_size=3)
    return summarized_text

#===========================================#
#              Streamlit Code               #
#===========================================#
desc = "Ši programa gali apibendrinti jūsų pateiktą lietuvišką tekstą."
authr = "Autorius: Robert Tarasevič, 2021"
uni = "VGTU FMF katedros studentas, grupė: TSF-17"

st.title('Lietuviško teksto automatinis apibendrinimas')
st.write(desc)
st.write(authr)
st.write(uni)
initial_inp = """Ryškiausia Paryžiaus įžymybė – 1889 metais pastatytas Eifelio bokštas, esantis Champ de Mars (Marso laukai) parke. 324 metrų aukščio metalinė konstrukcija yra vienas lankomiausių mokamų objektų pasaulyje, kiekvienais metais pritraukiantis milijonus turistų.
Eifelio bokštas pavadintas inžinieriaus Gustavo Eifelio garbei, kurio įmonė suprojektavo ir pastatė bokštą 1889 metais rengtai Pasaulio parodai. Iki pat 1930 metų tai buvo didžiausias statinys pasaulyje, kol jį nukarūnavo Niujorke pastatytas Chrysler Building dangoraižis. Tuo metu bokšto niekas nenorėjo pripažinti, buvo teigiama, kad jis bjaurojo Paryžiaus veidą, tačiau jo taip ir niekas nenugriovė, o laikui bėgant tapo ne tik miesto, bet ir visos šalies simboliu.

Įspūdingo bokšto statymo darbai truko tik 2 metus, 2 mėnesius ir 5 dienas. Jam prireikė 2,5 milijono kniedžių, daugiau nei 18 tūkstančių geležies gabalų ir apie 60 tonų dažų. Nors Eifelio bokštas atrodo vienspalvis, tačiau iš tikrųjų padengtas trijų skirtingų spalvų dažais: šviesiausia spalva dažoma viršūnė, o tamsiausia – apačia. Prieš daugiau nei 100 metų Paryžių papuošęs bokštas suprojektuotas pagal tuo metu pažangiausius inžinerinius sprendimus, todėl net pučiant stipriems vėjams, metalinė konstrukcija nesvyruoja daugiau nei 12 cm. 
Vieniems „Geležinė ledi“ tik eilinis metalo gabalas, kitiems – architektūrinis šedevras. Tačiau turbūt nėra nei vieno žmogaus, kuris atvykęs į Paryžių nepamatytų šio miesto simboliu tapusio bokšto.

Eifelyje įrengti 1665 laipteliai ir trys platformos su apžvalginėmis aikštelėmis bei kavinėmis. Pirmoji aikštelė yra 57, antroji – 115, o trečioji – 276 metrų aukštyje. Į pirmąjį ir antrąjį aukštus patenkama lipant laiptais arba kylant liftu, o norint pasiekti viršūnę – tik liftu. Iš bokšto atsiveria nepakartojama miesto panorama, o esant geram orui matomumas būna net 60 km atstumu.
Nuo 1889 iki 1930 metų Eifelio bokštas buvo aukščiausias statinys pasaulyje, kol jį pralenkė Niujorke pastatytas Chrysler Building dangoraižis. Tačiau 1957 metais ant bokšto viršaus pastačius anteną Chrysler Building pastatas tapo 5,2 metrais už Eifelį mažesnis. Vis dėlto, aukščiausio statinio karūna Eifelio bokšui nebuvo gražinta, nes 1931 metais Niujorke pastatytas dar vienas dangoraižis buvo aukštesnis ir už Eifelį, ir už Chrysler Building.

1925-1934 metais Prancūzijos automobilių kompanija Eifelio bokštą naudojo kaip reklaminį stendą. Tai vienintelis prekinis ženklas geležinę konstrukciją naudojęs reklamos tikslais.

Nepaisant didelio aukščio, montavimo metu žuvo tik vienas žmogus.

1912 metais austrų kilmės siuvėjas bei parašiutizmo pradininkas Franz Reichelt norėdamas išbandyti savo sukurtą parašiutą nušoko nuo Eifelio bokšto pirmojo aukšto. Deja, jo bandymas buvo nesėkmingas – išradėjas per 5 sekundes nukrito ant žemės ir užsimušė (pagal kitus šaltinius – mirties priežastis buvo širdies smūgis kritimo eigoje)."""

summarized_text = """Paryžiaus simboliu tapęs Eifelio bokštas, esantis Champ de Mars (Marso laukai) parke, yra ne tik vienas lankomiausių objektų pasaulyje, bet ir simbolinis simbolis, pritraukiantis milijonus turistų iš viso pasaulio, rašoma pranešime spaudai."""

#user_input = st.text_input('Jūsų tekstas:')
user_input = st.text_area(label='Jūsų tekstas:', value=initial_inp, height=200)
inp = " ".join(user_input.split(" ")[:550])

if st.button('Apibendrinti'):
    summarized_text = summarize(inp)[0][:-7]
#    st.write(summarized_text)
    st.text_area(label='Įvesto teksto santrauka:', value=summarized_text, height=100)
