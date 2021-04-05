import streamlit as st
import sentencepiece as spm
from fairseq.models.bart import BARTModel

#===========================================#
#                Load Model                 #
#===========================================#

BASEDIR = './model'
bart = BARTModel.from_pretrained(
        BASEDIR,
        checkpoint_file='checkpoint_best.pt',
        bpe='sentencepiece',
        sentencepiece_model=BASEDIR + '/sentence.bpe.model')
bart.eval()
#===========================================#
#                 Run Inference             #
#===========================================#

def summarize(full_text):
    summarized_text = bart.sample([full_text], beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
    return summarized_text

#===========================================#
#              Streamlit Code               #
#===========================================#
desc = "Ši programa gali apibendrinti jūsų pateiktą lietuvišką tekstą."

st.title('Lietuviško teksto automatinis apibendrinimas')
st.write(desc)

user_input = st.text_input('Jūsų tekstas:')

if st.button('Apibendrinti'):
    summarized_text = summarize(user_input)
    st.write(summarized_text)
