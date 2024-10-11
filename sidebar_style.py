import streamlit as st
import io
from PIL import Image
import base64

def apply_sidebar_style():

    st.logo("images/lsu_tiger.png")
    st.set_page_config(layout="wide",
               page_title="GeauxTune",
               page_icon="images/lsu_tiger.png",
               )
    
    file = open("images/logo_image.png", "rb")
    contents = file.read()
    file.close()

    # Encode the image to base64 for embedding in CSS
    img_str = base64.b64encode(contents).decode("utf-8")
    buffer = io.BytesIO()
    img_data = base64.b64decode(img_str)
    
    # Resize the image to your desired dimensions
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize((300, 200))  # Increase size here (width, height)
    resized_img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # CSS to embed the image at the top of the sidebar with the increased size
    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url('data:image/png;base64,{img_b64}');
                background-repeat: no-repeat;
                background-position: center 0px;  /* Center horizontally, adjust vertical position */
                background-size: 280px 120px;  /* Adjust the size of the image (increased size) */
                padding-top: 100px;  /* Keep padding consistent */
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )