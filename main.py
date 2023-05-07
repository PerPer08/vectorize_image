import streamlit as st
import numpy as np
import cv2
import svgwrite
import io


@st.cache_data
def get_img(uploaded_img):
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
   
    return img


@st.cache_data
def kmeans_clustering(k, img):
    # if(len(img.shape) < 3):
    #     z = img.reshape((-1,1))
    # elif(len(img.shape)== 3):
    z = img.reshape((-1,3))
    K = k
    attempts = 10
    z = np.float32(z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(z, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    print(center)
    res = center[label.flatten()]
    print(res)
    result_image = res.reshape((img.shape))

    return result_image

def dl_png(compressed_image):
    # Convert the compressed image back to uint8 format
    compressed_image = compressed_image.astype(np.uint8)
    compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)
    # Convert the compressed image to PNG format
    _, compressed_png = cv2.imencode(".png", compressed_image)
    # Get the binary data of the compressed image
    dl_img = compressed_png.tobytes()

    return dl_img

@st.cache_data
def vectorize_img(compressed_image):
    # Convert compressed image to vector image
    h, w, _ = compressed_image.shape
    img = svgwrite.Drawing(size=(w, h))
    for row in range(h):
        for col in range(w):
            r, g, b = compressed_image[row, col]
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            img.add(img.rect((col, row), (1, 1), fill=hex_color))
    # Output vector image as a file download
    output = io.BytesIO()
    img.write(output)
    data = output.getvalue()
    return data


def compress():
    st.session_state['compress'] = False
    
    return


if 'init' not in st.session_state:
    st.session_state['init'] = True


if 'compress' not in st.session_state:
    st.session_state['compress'] = False

if 'img_up' not in st.session_state:
    st.session_state['img_up'] = False





def main():
    st.set_page_config(page_title="Img ComVec")

    st.markdown("<h3 style='text-align: center; color: White;'>Compress and Vectorize Img</h3>", unsafe_allow_html=True)


    with st.sidebar:
        st.write("Upload an image and choose the number of colors to reduce it to using KMeans clustering:")
    
        uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"] )
        k_limit = 56
        k = st.slider("Number of colors:", 1, k_limit)


        if st.button("Compress Img"):
            st.session_state['compress'] = True

        

    col1,col2 = st.columns(2)
    if uploaded_img is not None:
        
        original_image = get_img(uploaded_img)
        with col1:
            st.image(original_image, caption="Original Image")

            
        

        if st.session_state['compress'] == True:
            
           
            compressed_image = kmeans_clustering(k, original_image)
            
            
            if 'comp_img' in st.session_state:
                st.session_state['comp_img'] = compressed_image
            st.session_state['comp_img'] = compressed_image
            st.session_state['img_up'] = True
            st.session_state['compress'] = False


        

    
        if st.session_state['img_up']:
            with col2:

                st.image(st.session_state['comp_img'], caption=f"Compressed Image (K={k})")

                st.download_button(
                    label="Download compressed image",
                    data = dl_png(st.session_state['comp_img']),
                    file_name="compressed.png",
                    mime="image/png"
                )

                if st.button("vectorize img"):
                    st.download_button(
                        label="Download vectorized image",
                        data=vector_img(st.session_state['comp_img']),
                        file_name="vectorized.svg",
                        mime="image/svg+xml"
                    )


if __name__ == "__main__":
    main()