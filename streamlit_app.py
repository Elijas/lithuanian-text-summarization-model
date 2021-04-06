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
    summarized_text = get_model().sample([full_text], beam=5, lenpen=2.0, max_len_b=140, min_len=30, no_repeat_ngram_size=3)
    return summarized_text

#===========================================#
#              Streamlit Code               #
#===========================================#
desc = "Ši programa gali apibendrinti jūsų pateiktą lietuvišką tekstą."

st.title('Lietuviško teksto automatinis apibendrinimas')
st.write(desc)

#user_input = st.text_input('Jūsų tekstas:')
user_input = st.text_area(label='Jūsų tekstas:', height=200)

if st.button('Apibendrinti'):
    summarized_text = summarize(user_input)[0][:-7]
#    st.write(summarized_text)
    st.text_area(label='Įvesto teksto santrauka:', value=summarized_text, height=100)
