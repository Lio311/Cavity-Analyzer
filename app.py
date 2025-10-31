import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Streamlit general settings
st.set_page_config(page_title="X-Ray Hole Detection (Fourier)", layout="wide")
st.title("🔬 X-Ray Hole Detection using Fourier Transform")
st.markdown("A classic image processing tool for defect detection, leveraging the frequency domain (high frequencies = edges/defects).")

# -----------------------------------------------------------------
# Fourier Transform Based Image Processing Function
# -----------------------------------------------------------------

def fourier_defect_detection(image_array, cutoff_radius, threshold):
    # Step A: Preprocessing
    # Convert image to grayscale and float32
    img_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    img_float = np.float32(img_gray)
    
    # Step B: Moving to the Frequency Domain (DFT)
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Calculate the magnitude spectrum (for visualization)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
    
    # Step C: Creating a High Pass Filter (HPF)
    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2  # Center of the spectrum
    
    # Create the filter mask (Ideal HPF for simplicity)
    mask = np.ones((rows, cols, 2), np.uint8)
    
    # Create the LPF mask and then HPF by subtracting from 1
    center = [crow, ccol]
    for i in range(rows):
        for j in range(cols):
            # Distance from the center
            distance = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            
            # If the distance is smaller than the cutoff radius, it's a low frequency (block it)
            if distance < cutoff_radius:
                mask[i, j] = 0
                
    # Step D: Applying the Filter
    fshift = dft_shift * mask
    
    # Step E: Returning to the Spatial Domain (IDFT)
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    
    # Highlight defects by normalizing the values
    # Defects (holes) will appear as bright spots
    defect_highlight = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Step F: Segmentation and Detection
    _, binary_image = cv2.threshold(defect_highlight, threshold, 255, cv2.THRESH_BINARY)
    
    # Finding contours for hole identification (Connected Components)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Drawing contours on the original image
    img_result = image_array.copy()
    num_defects = 0
    for cnt in contours:
        # Noise filtering: ignore very small spots (you can tune the value 200)
        if cv2.contourArea(cnt) > 200:
            x, y, w, h = cv2.boundingRect(cnt)
            # Draw a red rectangle around the detected defect
            cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            num_defects += 1
            
    return img_result, magnitude_spectrum, defect_highlight, binary_image, num_defects

# -----------------------------------------------------------------
# Streamlit User Interface
# -----------------------------------------------------------------

# Sidebar for parameters
with st.sidebar:
    st.header("⚙️ Processing Parameters")
    # File upload
    uploaded_file = st.file_uploader("Select an X-ray image to upload (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    # Parameters for Fourier Transform (HPF Cutoff Radius)
    st.subheader("Fourier Transform (HPF)")
    cutoff_radius = st.slider(
        "Cutoff Radius (Low Frequency Suppression):",
        min_value=5, max_value=100, value=30, step=5,
        help="Higher value means smaller details are emphasized and the background is suppressed more."
    )
    
    # Parameters for Segmentation (Threshold)
    st.subheader("Segmentation and Detection")
    threshold = st.slider(
        "Detection Threshold:",
        min_value=10, max_value=250, value=150, step=10,
        help="The brightness threshold for binary conversion. Higher value = fewer defects will be detected."
    )
    
    st.markdown("---")
    st.info("💡 Defects appear as sharp changes and are therefore highlighted by high frequencies.")

if uploaded_file is not None:
    # Load the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Perform detection
    result_img, magnitude, defect_highlight, binary_img, num_defects = fourier_defect_detection(img, cutoff_radius, threshold)

    st.success(f"✅ Found **{num_defects}** potential defects.")

    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Source Image with Defect Marking")
        st.image(result_img, channels="BGR", caption="Defects marked with a red rectangle", use_column_width=True)
        
    with col2:
        st.subheader("Fourier Spectrum (Demonstration)")
        # Display the magnitude spectrum using Matplotlib
        fig, ax = plt.subplots()
        ax.imshow(magnitude, cmap='gray')
        ax.title.set_text("Shifted Magnitude Spectrum (FFT Shift)")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Additional Processing Steps")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.text("Image After High Pass Filter (HPF)")
        st.image(defect_highlight, caption="Highlighting defects and suppressing the background", use_column_width=True)
        
    with col4:
        st.text("Binary Segmentation (After Applying Threshold)")
        st.image(binary_img, caption=f"Pixels above threshold {threshold} turn white", use_column_width=True)
        

else:
    st.info("Please upload an X-ray image to start the detection.")
