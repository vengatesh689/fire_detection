import streamlit as st
import cv2
import numpy as np
import av
import torch
import tempfile
from PIL import Image
import geocoder
from geopy.geocoders import Nominatim
import requests

# Set the background color
background_color = "#00ff00"  # blue color, you can use any other color code

# Set the page configuration
st.set_page_config(page_title="Fire Detection", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

# Apply the background color using the theme configuration
st.markdown(f"""
    <style>
        body {{
            background-color: {background_color};
        }}
    </style>
""", unsafe_allow_html=True)


def get_location():
    location = st.session_state.get("location")
    if location:
        return location
    else:
        # Default to Madurai's coordinates
        return {'city': 'Madurai', 'latitude': 9.9252, 'longitude': 78.1198}

def get_weather(latitude, longitude):
    # Replace 'YOUR_API_KEY' with your actual API key
    api_key = '19b1d383b69e6c595c4e1fb3e5191c96'
    api_url = f'http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric'
    response = requests.get(api_url)
    if response.status_code == 200:
        weather_data = response.json()
        temperature = weather_data['main']['temp']
        weather_description = weather_data['weather'][0]['description']
        return temperature, weather_description
    else:
        return None, None


def send_email_with_attachment(sender_email, password, department_emails, subject, body, attachment_path,
                               original_attachment_path, location, camera_location, weather_data):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    for department, receiver_email in department_emails.items():
        # Create the email message
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = receiver_email
        message['Subject'] = subject

        # Construct the email body
        body_with_location = f"{body}\n\nLocation: {location['city']}, Latitude: {location['latitude']}, Longitude: {location['longitude']}\nWeather: {weather_data[1]}, Temperature: {weather_data[0]} °C"
        message.attach(MIMEText(body_with_location, "plain"))

        # Attach the cropped image
        attachment = open(attachment_path, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename= {attachment_path}")
        message.attach(part)

        # Attach the original image
        original_attachment = open(original_attachment_path, "rb")
        original_part = MIMEBase('application', 'octet-stream')
        original_part.set_payload((original_attachment).read())
        encoders.encode_base64(original_part)
        original_part.add_header('Content-Disposition', f"attachment; filename= {original_attachment_path}")
        message.attach(original_part)

        # Send the email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()


@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="weights/best.pt", force_reload=True)
    return model

# Load the YOLOv5 model
model = load_model()


demo_img = "fire.9.png"
demo_video = "Fire_Video.mp4"

st.title('Fire Detection')
st.sidebar.title('App Mode')

app_mode = st.sidebar.selectbox('Choose the App Mode',
                                ['About App', 'Detect on Image', 'Detect on Video', 'Detect on WebCam'])

if app_mode == 'About App':
    st.subheader("About")
    st.markdown(' <h5>🔥 WildfireEye: YOLO-Based Forest Fire Detection and Alert System</h5>', unsafe_allow_html=True)

    st.markdown("- <h5>Forest Fire</h5>", unsafe_allow_html=True)
    st.image("Images/Forest-Fire-Protection.jpg")
    st.markdown("- <h5>Detection system on YOLO</h5>", unsafe_allow_html=True)
    st.image("Images/10-Figure11-1.png")

    st.markdown("""
                ## Features
- Detect on Image
- Detect on Videos
- Live Detection
""")

if app_mode == 'Detect on Image':
    st.subheader("Detected Fire:")
    text = st.markdown("")

    st.sidebar.markdown("---")
    # Input for Image
    img_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if img_file:
        image = np.array(Image.open(img_file))
    else:
        image = np.array(Image.open(demo_img))

    st.sidebar.markdown("---")
    st.sidebar.markdown("Original Image")
    st.sidebar.image(image)

    # predict the image
    model = load_model()
    results = model(image)
    length = len(results.xyxy[0])
    output = np.squeeze(results.render())
    text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>", unsafe_allow_html=True)
    st.subheader("Output Image")
    st.image(output, use_column_width=True)

if app_mode == 'Detect on Video':
    st.subheader("Detected Fire:")
    text = st.markdown("")

    st.sidebar.markdown("---")

    st.subheader("Output")
    stframe = st.empty()

    # Input for Video
    video_file = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    st.sidebar.markdown("---")
    tffile = tempfile.NamedTemporaryFile(delete=False)

    if not video_file:
        vid = cv2.VideoCapture(demo_video)
        tffile.name = demo_video
    else:
        tffile.write(video_file.read())
        vid = cv2.VideoCapture(tffile.name)

    st.sidebar.markdown("Input Video")
    st.sidebar.video(tffile.name)

    # predict the video
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model = load_model()
        results = model(frame)
        detections = results.xyxy[0]
        length = sum(1 for d in detections if d[4] >= 0.7)  # Count detections with confidence >= 0.7
        output = np.squeeze(results.render())
        text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>", unsafe_allow_html=True)
        stframe.image(output)

if app_mode == 'Detect on WebCam':
    st.subheader("Detected Fire:")
    text = st.markdown("")

    st.sidebar.markdown("---")

    st.sidebar.subheader("Settings")
    threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)

    st.subheader("Output")
    stframe = st.empty()

    run = st.sidebar.button("Start")
    stop = st.sidebar.button("Stop")
    st.sidebar.markdown("---")

    cam = cv2.VideoCapture(0)

    # Set frame width, height, and FPS
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)

    # Load the model outside the loop
    model = load_model()

    if run:
        while True:
            if stop:
                break
            ret, frame = cam.read()

            # Check if the frame is not empty
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame)

            # Filter out detections below confidence threshold
            detections = [d for d in results.xyxy[0] if d[4] >= threshold]
            length = len(detections)
            output = np.squeeze(results.render())

            # Crop and save the detected object if it has high confidence
            if length > 0:
                detected_object = frame[int(detections[0][1]):int(detections[0][3]),
                                  int(detections[0][0]):int(detections[0][2])]
                cv2.imwrite('detected_object.jpg', cv2.cvtColor(detected_object, cv2.COLOR_RGB2BGR))
                # Save the original image
                cv2.imwrite('original_image.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                # Send email with the cropped and original images as attachments
                sender_email = "firedetection60@gmail.com"
                password = "pkdt wqvy bzlr heqj"
                department_emails = {
                    "Fire Department": "firedetection60@gmail.com"
                    # "Forest Department": "f07387005@gmail.com",
                    # "Ambulance Department": "f07387005@gmail.com"
                }
                subject = "Emergency Alert!"
                body = "Fire detected, please respond immediately."
                attachment_path = 'detected_object.jpg'

                # Use geocoder to get the location
                location = get_location()
                latitude = location['latitude']
                longitude = location['longitude']

                # Get the camera location
                camera_location = location['city'] + f'. [Lat: {latitude}, Lng:{longitude}]'

                # Get weather data
                temperature, weather_description = get_weather(latitude, longitude)
                weather_data = (temperature, weather_description)

                # Path to the original image
                original_attachment_path = 'original_image.jpg'

                send_email_with_attachment(sender_email, password, department_emails, subject, body, attachment_path,
                                           original_attachment_path, location, camera_location, weather_data)
                st.write("Fire detected and Alert sent!")

            text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>", unsafe_allow_html=True)
            stframe.image(output)

    cam.release()