import streamlit as st



st.markdown(
        """
        <style>
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% {
                transform: translateY(0);
            }
            40% {
                transform: translateY(-10px);
            }
            60% {
                transform: translateY(-5px);
            }
        }
        .animated-title {
            font-size: 55px;
            font-weight: 800;
            text-align: center;
            animation: bounce 2s infinite;
        }
        </style>
        <div class="animated-title">Welcome to ShelfMate</div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
        """
        <style>
        .gradient-text {
            font-size: 25px;
            font-weight: 800;
            background: linear-gradient(50deg, #740001, #c8a67f);  /* Gradient colors */
            -webkit-background-clip: text;  /* Clip gradient to text */
            -webkit-text-fill-color: transparent;  /* Hide original text */
            text-align: center;
        }
        </style>
        <div class="gradient-text">Book suggestion chatbot!</div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)
st.write("With ShelfMate you will be able to receive book suggestions based on your favourite authors, books and genres. You will also be able to create a weekly, monthly and yearly reading plan.")
st.write("ShelfMate was created in 2024, with the mission of grow the love for reading by helping readers discover books according to their unique personalities. We want to become the world’s most trusted and intuitive platform for book discovery, redefining how readers explore and connect with literature.")
st.write("If you want to know more about us, click on the following button!")

if st.button("About Us"):
    
    st.switch_page("pages/About.py")


st.write("Register now to unlock your personalized book recommendations and start your reading journey!")
if st.button("Register Now"):
    
    st.switch_page("pages/Register.py")
    

st.subheader("Our Mood Board")  
st.image("images/background1.png", caption="Mood Board", use_container_width=True)
