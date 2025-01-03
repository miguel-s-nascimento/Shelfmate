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
            font-size: 45px;
            font-weight: 800;
            text-align: center;
            animation: bounce 2s infinite;
        }
        </style>
        <div class="animated-title">About Us</div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
        """
        <style>
        .justified-text {
            text-align: justify;
        }
        </style>
        <div class="justified-text">
            <b>Created in 2024</b>, <b>ShelfMate</b> is an <b>innovative book recommendation platform</b> created to <b>promote enthusiasm for reading</b> by providing <b>incredibly tailored recommendations</b> based on <b>individual preferences, passions, and moods</b>.
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .custom-subheader {
        color: #740001;  /* Change to desired color */
        font-size: 28px;
        font-weight: 700;
        text-align: left;
    }
    </style>
    <div class="custom-subheader">Our Mission</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
        """
        <div class="justified-text">
            <b>ShelfMate's mission</b> is to <b>grow the love for reading</b> by helping readers <b>discover books according to their unique personalities</b>. It believes that the <b>right book at the right time</b> can <b>inspire, entertain, and transform lives</b>. Through an <b>intelligent recommendation platform</b>, it aims to connect readers with stories that <b>resonate</b>, from <b>classics to undiscovered masterpieces</b>. Whether you are an <b>active reader</b> or just <b>starting your journey</b>, ShelfMate is here to be your <b>trusted guide</b>, making book discovery an <b>effortless, enjoyable, and personalized experience</b> for everyone.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .custom-subheader {
        color: #740001;  /* Change to desired color */
        font-size: 28px;
        font-weight: 700;
        text-align: left;
    }
    </style>
    <div class="custom-subheader">Our Vision</div>
    """,
    unsafe_allow_html=True
)
st.markdown(
        """
        <div class="justified-text">
            <b>ShelfMate's vision</b> is to become the <b>world’s most trusted and intuitive platform</b> for <b>book discovery</b>, redefining how readers <b>explore and connect with literature</b>. It aspires to create a space where <b>individuals from all backgrounds</b> can seamlessly find books that align with their <b>tastes, interests, and emotions</b>, fostering a <b>deeper appreciation</b> for stories across <b>diverse genres, cultures, and languages</b>. By blending <b>modern technology</b> with a <b>passion for literature</b>, the platform seeks to <b>revolutionize the book discovery experience</b>, making it <b>personalized, inclusive, and enriching</b> for readers at every stage of their <b>literary journey</b>.
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .custom-subheader {
        color: #740001;  /* Change to desired color */
        font-size: 28px;
        font-weight: 700;
        text-align: left;
    }
    </style>
    <div class="custom-subheader">Our Values</div>
    """,
    unsafe_allow_html=True
)


# Add custom CSS to style the smaller squares/rectangles
st.markdown("""
    <style>
    .value-box {
        display: inline-block;
        width: 150px;
        height: 70px;
        margin: 5px;
        padding: 15px;
        background-color: #f1f1f1;
        text-align: center;
        font-size: 16px;
        font-weight: bold;
        border: 2px solid #740001;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Add the squares/rectangles with the values inside in a single row
values = ['Transparency', 'Integrity', 'Excellence', 'Personalization']

# Use columns to layout the boxes in a row
cols = st.columns(4)

for idx, value in enumerate(values):
    with cols[idx]:
        st.markdown(f'<div class="value-box">{value}</div>', unsafe_allow_html=True)
        

st.markdown("<br>", unsafe_allow_html=True)
        