import streamlit as st
import numpy as np
import cv2
#import svgwrite


@st.cache_data
def get_img(uploaded_img):
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
   
    return img


@st.cache_data
def kmeans_clustering(k, img):

    if(len(img.shape) < 3):
        z = img.reshape((-1,1))
    elif(len(img.shape)== 3):
        z = img.reshape((-1,3))

    K = k
    attempts = 10
    z = np.float32(z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    
    ret, label, center = cv2.kmeans(z, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    print(ret)
    return result_image


# @st.cache_data
# def vector_img(compressed_image):
#     # Convert compressed image to vector image
#     h, w, _ = compressed_image.shape
#     dwg = svgwrite.Drawing('vectorized.svg', size=(w, h))
#     for row in range(h):
#         for col in range(w):
#             r, g, b = compressed_image[row, col]
#             hex_color = f'#{r:02x}{g:02x}{b:02x}'
#             dwg.add(dwg.rect((col, row), (1, 1), fill=hex_color))
#     # Output vector image as a file download
#     with open('vectorized.svg', 'rb') as f:
#         data = f.read()

#     return data

if 'init' not in st.session_state:
    st.session_state['init'] = True
if 'upload' not in st.session_state:
    st.session_state['upload'] = "not done"






def main():
    st.set_page_config(page_title="Image Compression and Vectorization App")

    st.title("Image Compression and Vectorization App")
    st.write("Upload an image and choose the number of colors to reduce it to using KMeans clustering:")
    
    uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], on_change= st.session_state['init'] = True)
    k_limit = 56
    if st.session_state['init']:
        st.session_state['init'] = 'false'
        if uploaded_img is not None:
            for i in range(k_limit)
            original_image = get_img(uploaded_img)
            compressed_image = kmeans_clustering(k, original_image)
        

    col1,col2 = st.columns(2)
    if uploaded_img is not None:
        original_image = get_img(uploaded_img)
        with col1:
            st.image(original_image, caption="Original Image")

            k = st.slider("Number of colors:", 1, k_limit)
        
        if st.button("Compress Img"):
            compressed_image = kmeans_clustering(k, original_image)
            with col2:
                st.image(compressed_image, caption=f"Compressed Image (K={k})")
                # Convert the compressed image back to uint8 format
                compressed_image = compressed_image.astype(np.uint8)
                compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)

                # Convert the compressed image to PNG format
                _, compressed_png = cv2.imencode(".png", compressed_image)

                # Get the binary data of the compressed image
                dl_img = compressed_png.tobytes()
                
                st.download_button(
                    label="Download compressed image",
                    data=dl_img,
                    file_name="compressed.png",
                    mime="image/png"
                )
                # # if st.button("vectorize Img"):
                #     # with col3:
                
                # data = vector_img(compressed_image)
                # st.image(data, caption=f"Vectorized Image (K={k})")
                # st.download_button(
                #     label="Download image",
                #     data=data,
                #     file_name="vectorized.svg",
                #     mime="image/svg+xml"
                # )


if __name__ == "__main__":
    main()