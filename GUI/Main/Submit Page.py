import streamlit as st

# Set page configuration and settings
st.set_page_config(page_title="Citizens Issues Submission ", layout="wide",page_icon="📝")

# Background image with animation witth CSS
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://www.tripsavvy.com/thmb/-tzBp8Gy4A2v7-XK07TYecZXWfk=/2286x1311/filters:fill(auto,1)/GettyImages-200478089-001-06db86e7b540494a807a46af6c6c7f11.jpg");
    background-size: cover;
    animation: moveBackground 120s linear infinite alternate;
}

@keyframes moveBackground {
    0% { background-position: 0 0; }
    100% { background-position: 100% 0; }
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# This creates the form with the title
st.title("Citizen Issues Submission ")
with st.form("client_form"):
    name = st.text_input("Client Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone Number")
    comment = st.text_area("Describe the problem in detail")

    submitted = st.form_submit_button("Submit")

if submitted:
    # This line of code saves the data in session state {its like a temporary storage} and then we
    #     recall it in the Prediction Page
    st.session_state['Citizen'] = [name,email,phone,comment]
    st.toast("Form submitted successfully!", icon="✅")

# We can use this to send the data to the model and get the prediction

#if submitted:
#    category, sub_category, confidence = predict_category(comment)
#st.session_state['Prediction'] = ["category", "sub_category", 91.569]

# Feedback section
col1, col2, col3 = st.columns([2,2,1])
with col2 :
    st.markdown("###### was this submission helpful?")
col1, col2, col3 ,g,t= st.columns([2,3,1,3,2])
with col3 :
    selected = st.feedback("thumbs")