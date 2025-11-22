import streamlit as st


st.set_page_config(page_title="Prediction Page", layout="wide",page_icon="📈")

left, center, right = st.columns([1,4,1])
with center:
    st.title("Citizen Issue Prediction Page")
    st.write("")


if st.session_state.get('Citizen'):
    name = st.session_state['Citizen'][0]
    email = st.session_state['Citizen'][1]
    phone = st.session_state['Citizen'][2]
    comment = st.session_state['Citizen'][3]


    with st.popover("Citizen Details",width="stretch"):
        col1, col2 = st.columns([1,3])
        with col1:
            st.write(f"**Name:** {name}")
            st.write(f"**Email:** {email}")
        with col2:
            st.write(f"**Phone:** {phone}")
        st.write(f"**Comment:** {comment}")


if st.session_state.get('Prediction'):
    st.markdown(f"## Prediction Results")
    col1, col2, col3 = st.columns([2,3,1])
    with col1:
        st.markdown(f"#### Category:",)
        st.markdown(f"#### Sub-Category:")
    with col2:
        st.markdown(f"#### {st.session_state['Prediction'][0].lower()}")
        st.markdown(f"#### {st.session_state['Prediction'][1].lower()}")
    with col3:
        st.metric(label="Confidence", value=f"{st.session_state['Prediction'][2]:.2f}%",delta="4%")
else:
    st.markdown(st.session_state.get('Prediction', "No prediction available. Please submit the form first."))