import streamlit as st

#===========================================#
#                Load Model                 #
#===========================================#

model = ...

#===========================================#
#                 Run Inference             #
#===========================================#

def summarize(full_text):
    # summarized_text = model.run(full_text)
    summarized_text = "Apibendrintas tekstas"
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