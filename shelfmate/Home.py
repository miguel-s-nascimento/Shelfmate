import streamlit as st

st.image("images/logo2.png", width=150, use_container_width=False)


st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #4a2d1c;
        }
        [data-testid="stSidebar"] * {
            color: #fff3da;
        }
        
    </style>
    """,
    unsafe_allow_html=True
)

# Define pages, including standalone pages and grouped pages
pages = {
    
    "ShelfMate": [
        st.Page("pages/Welcome.py", title="Home Page", icon="🏠"),
        st.Page("pages/About.py", title="About Us", icon="👥"),
        st.Page("pages/Chatbot.py", title="Chatbot", icon="🤖"),
        st.Page("pages/ReadList.py", title="Read List", icon="📖")
        
    ],
    
    
    "Your account": [
        st.Page("pages/Login.py", title="Log in",icon="🔓"),
        st.Page("pages/Register.py", title="Register", icon="📝")
        
    ]
    
    
}

# Initialize the navigation
pg = st.navigation(pages)

# Run the navigation component
pg.run()