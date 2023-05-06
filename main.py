import streamlit as st
import numpy as np
import cv2
import svgwrite


@st.cache_data
def get_img(uploaded_img):
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    return img


@st.cache_data
def kmeans_clustering(k, img):
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = k
    attempts = 10
    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    return result_image

@st.cache_data
def vector_img(compressed_image):
    # Convert compressed image to vector image
    h, w, _ = compressed_image.shape
    dwg = svgwrite.Drawing('vectorized.svg', size=(w, h))
    for row in range(h):
        for col in range(w):
            r, g, b = compressed_image[row, col]
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            dwg.add(dwg.rect((col, row), (1, 1), fill=hex_color))
    # Output vector image as a file download
    with open('vectorized.svg', 'rb') as f:
        data = f.read()

    return data



def main():
    st.set_page_config(page_title="Image Compression and Vectorization App")

    st.title("Image Compression and Vectorization App")
    st.write("Upload an image and choose the number of colors to reduce it to using KMeans clustering:")

    uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    col1,col2,col3 = st.columns(3)
    if uploaded_img is not None:
        original_image = get_img(uploaded_img)
        with col1:
            st.image(original_image, caption="Original Image")

            k = st.slider("Number of colors:", 2, 100)
        compressed_image = kmeans_clustering(k, original_image)
        if st.button("Compress Img"):
            with col2:
                st.image(compressed_image, caption=f"Compressed Image (K={k})")
                st.download_button(
                    label="Download compressed image",
                    data=compressed_image,
                    file_name="compressed.png",
                    mime="image/png"
                )
                # if st.button("vectorize Img"):
                    # with col3:
                
                data = vector_img(compressed_image)
                st.image(data, caption=f"Vectorized Image (K={k})")
                st.download_button(
                    label="Download image",
                    data=data,
                    file_name="vectorized.svg",
                    mime="image/svg+xml"
                )


if __name__ == "__main__":
    main()