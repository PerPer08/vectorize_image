import streamlit as st
import numpy as np
import cv2
# import svgwrite
# import io
# from potrace import Bitmap, Path
# from PIL import Image

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
    result_image = res.reshape((img.shape))
    print(result_image)
    return result_image

def dl_jpg(compressed_image):
    # Convert the compressed image back to uint8 format
    compressed_image = compressed_image.astype(np.uint8)
    compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)
    # Convert the compressed image to PNG format
    _, compressed_jpg = cv2.imencode(".jpg", compressed_image)
    # Get the binary data of the compressed image
    dl_img = compressed_jpg.tobytes()

    return dl_img

# @st.cache_data
# def vectorize_img(ndarray):
#     # Convert ndarray to grayscale
#     gray = cv2.cvtColor(ndarray, cv2.COLOR_RGB2GRAY)

#     # Create a bitmap object from the grayscale image
#     bmp = Bitmap(gray.tolist())

#     # Trace the bitmap to get a path object
#     path = bmp.trace()

#     # Create an SVG object and add the path
#     dwg = svgwrite.Drawing('vectorized.svg', profile='tiny')
#     path_data = Path(dwg).from_potrace(path).d()
#     dwg.add(dwg.path(d=path_data, fill='none', stroke='black'))

#     # Save the SVG file and return the binary data
#     buffer = io.BytesIO()
#     dwg.write(buffer)
#     buffer.seek(0)
#     svg_data = buffer.getvalue()
#     return svg_data




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

    st.markdown("<h3 style='text-align: center; color: White;'>Compress image using kmeans</h3>", unsafe_allow_html=True)


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
                    data = dl_jpg(st.session_state['comp_img']),
                    file_name="compressed.jpg",
                    mime="image/jpg"
                )

                # if st.button("vectorize img"):
                #     st.download_button(
                #         label="Download vectorized image",
                #         data=vectorize_img(st.session_state['comp_img']),
                #         file_name="vectorized.svg",
                #         mime="image/svg+xml"
                #     )


if __name__ == "__main__":
    main()