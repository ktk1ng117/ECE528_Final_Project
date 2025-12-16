# When RPi is triggered, run this script which grabs a frame from the camera
# converts it to greyscale and runs efficientnet on it to classify image
import os
import logging
import time
import numpy as np
import RPi.GPIO as GPIO
from datetime import datetime
from PIL import Image
import serial
import argparse
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

# Suppress Python logging from picamera2 module
logging.getLogger("picamera2").setLevel(logging.CRITICAL)

HIGH_RES = (3280, 2464)
LOW_RES = (224, 224)
SIGNAL_PIN = 17
CLASS_DEF = ["Black Bear", "Brown Bear", "Cougar", "Coyote", "Human", "Non-Human", "Polar Bear", "Wolf"]

parser = argparse.ArgumentParser(description='Motion Activated Harmful Animal Detection and Deterrence System using TFLite Models')
parser.add_argument('--res_mode', type=str, choices=['high', 'low'], default='low',
                    help='Resolution mode for image capture: "high" for high-resolution, "low" for low-resolution (default: low)')
parser.add_argument('--greyscale', action='store_true',
                    help='Use greyscale TFLite model (default: False)')
parser.add_argument('--quant', action='store_true',
                    help='Use quantized TFLite model (default: False)')
parser.add_argument('--base_model', type=str, choices=['efficientnet', 'mobilenet', 'resnet50', 'yolov11'], default='efficientnet',
                    help='Base model type to use: "efficientnet", "mobilenet", "resnet50", or "yolov11" (default: efficientnet)')
parser.add_argument('--continuous', action='store_true',
                    help='Enable continuous monitoring mode without Arduino trigger (default: False)')
parser.add_argument('--burst', action='store_true',
                    help='Enable burst mode: capture 25 images with 5 second intervals (default: False)')
parser.add_argument('--interval', type=float, default=5,
                    help='Interval in seconds between captures in continuous mode (default: 5)')
args = parser.parse_args()
res_mode = args.res_mode
greyscale = args.greyscale
quant = args.quant
base_model = args.base_model
continuous_mode = args.continuous
burst_mode = args.burst
capture_interval = args.interval

# Configure logging
log_file = "mahadd_predictions.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def setup_gpio(pin_number):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin_number, GPIO.IN)

def evaluate_tflite_model(interpreter, image_path):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess the image
    img = Image.open(image_path)
    img = np.array(img, dtype=np.float32)

    # If the image is greyscale, expand dimensions to match model input
    if len(img.shape) == 2:  # Greyscale image has shape (height, width)
        img = np.expand_dims(img, axis=-1)  # Add channel dimension

    # Ensure the image has 3 channels if the model expects RGB
    if input_details[0]['shape'][3] == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)  # Repeat the single channel 3 times

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run the inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data, axis=1)[0]

    return predicted_class

def capture_image_raw():
    """Capture a raw image from the camera and return as PIL Image."""
    camera = Picamera2()
    
    # Configure camera to use full resolution
    config = camera.create_still_configuration(main={"size": HIGH_RES})
    camera.configure(config)
    
    camera.start()
    time.sleep(0.1)  # Allow camera to warm up
    img_array = camera.capture_array()
    camera.stop()
    camera.close()
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Convert RGBA to RGB if necessary (JPEG doesn't support alpha channel)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    return img

def save_image(img, filepath, target_resolution=LOW_RES):
    """Save an image to the specified filepath with optional resizing and greyscale conversion."""
    # Resize image
    img_resized = img.resize(target_resolution, Image.LANCZOS)
    
    # Convert to greyscale if specified
    if greyscale:
        img_resized = img_resized.convert('L')
    
    # Save image
    img_resized.save(filepath)
    
    return filepath

def capture_and_save_image():
    """Capture image and save to both high-res and low-res directories."""
    # Capture raw image
    img = capture_image_raw()
    
    # Save the file with the timestamp in the name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    high_res_filepath = f'/home/admin/mahadd/images/highres/image_{timestamp}.jpg'
    low_res_filepath = f'/home/admin/mahadd/images/lowres/image_{timestamp}.jpg'
    
    # Save high-res image
    img.save(high_res_filepath)
    
    # Save low-res image
    save_image(img, low_res_filepath, LOW_RES)
    
    if res_mode == 'high':
        return high_res_filepath
    else:
        return low_res_filepath

def capture_image():
    """Capture image and save to temporary directory for inference."""
    # Capture raw image
    img = capture_image_raw()
    
    # Save to temporary file for inference
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tmp_dir = '/tmp/mahadd_captures'
    os.makedirs(tmp_dir, exist_ok=True)
    img_filepath = f'{tmp_dir}/temp_image_{timestamp}.jpg'
    
    # Save image with proper formatting
    save_image(img, img_filepath, LOW_RES)
    
    return img_filepath

def get_model_filepath(base_model, quant, greyscale):
    base_model_folder = f"/home/admin/mahadd/models/{base_model}/"
    if greyscale:
        if quant:
            return base_model_folder + f"{base_model}_model_greyscale_quantized.tflite"
        else:
            return base_model_folder + f"{base_model}_model_greyscale.tflite"
    else:
        if quant:
            return base_model_folder + f"{base_model}_model_quantized.tflite"
        else:
            return base_model_folder + f"{base_model}_model.tflite"

def get_model_name(base_model, quant, greyscale):
    name = base_model
    if greyscale:
        name += "_greyscale"
    if quant:
        name += "_quantized"
    return name

# load model in keras
# model = tf.keras.models.load_model('model_EfficientNetb0.keras')

# Initialize GPIO, serial connection, and a few other things
setup_gpio(SIGNAL_PIN)
# Initialize serial communication with Arduino
arduino_serial = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

if (res_mode == 'high'):
    # Error because HIGH_RES not supported for any of our models currently
    raise ValueError("High-resolution mode is not supported for our TFLite model. Please use low-resolution mode.")
else:
    print("Using low-resolution images for inference.")
    TARGET_RES = LOW_RES

print(f"Using base model: {base_model}, Greyscale: {greyscale}, Quantized: {quant}")
# Log model details at the start
logging.info("======= Starting Script Using " + get_model_name(base_model, quant, greyscale) + " =======")

try:
    print("Waiting for motion...")
    while True:
        if continuous_mode:
            #print("Continuous mode enabled. Capturing image and running inference...")
            #img_filepath = capture_and_save_image()
            img_filepath = capture_image()

            model_filepath = get_model_filepath(base_model, quant, greyscale)
            model_name = get_model_name(base_model, quant, greyscale)
            interpreter = tflite.Interpreter(model_path=model_filepath)
            interpreter.allocate_tensors()

            # Measure inference time
            start_time = time.time()
            predicted_class = evaluate_tflite_model(interpreter, img_filepath)
            inference_time = time.time() - start_time

            # Log prediction details
            #model_size = os.path.getsize(model_filepath) / float(2**20)
            #logging.info(
            #    f"Model Invoked={model_name}, "
             #   f"Motion Detected: Class={CLASS_DEF[predicted_class]}, "
             #   f"Model Size={model_size:.4f}MB, " f"Inference Time={inference_time:.4f}s, "
            #    f"Captured Image={img_filepath}"
            #)

            #print("Prediction complete!")
            print("Predicted Class:", CLASS_DEF[predicted_class])
            #print(f"{model_name} TFLite Model Size (MB):", model_size)
            print("Inference Time (s):", inference_time)

            # Send the detected class name to Arduino if the class is not "Non-Human"
            #class_name = CLASS_DEF[predicted_class]
            #print(f"Sending class '{class_name}' to Arduino.")
            #arduino_serial.write(f"{class_name}\n".encode())

            # Wait for Arduino response
            #while True:
            #    if arduino_serial.in_waiting > 0:
            #        arduino_response = arduino_serial.readline().decode().strip()
            #        print(f"Arduino echo: {arduino_response}")
            #        break
            if (capture_interval > 0):
                time.sleep(capture_interval)  # Interval between captures in continuous mode
        elif burst_mode:
            print("Burst mode enabled. Capturing 25 images with 5 second intervals...")
            for i in range(25):
                print(f"Burst capture {i+1}/25...")
                img_filepath = capture_and_save_image()

                model_filepath = get_model_filepath(base_model, quant, greyscale)
                model_name = get_model_name(base_model, quant, greyscale)
                interpreter = tflite.Interpreter(model_path=model_filepath)
                interpreter.allocate_tensors()

                # Measure inference time
                start_time = time.time()
                predicted_class = evaluate_tflite_model(interpreter, img_filepath)
                inference_time = time.time() - start_time

                # Log prediction details
                model_size = os.path.getsize(model_filepath) / float(2**20)
                logging.info(
                    f"Model Invoked={model_name}, "
                    f"Burst Mode ({i+1}/25): Class={CLASS_DEF[predicted_class]}, "
                    f"Model Size={model_size:.4f}MB, Inference Time={inference_time:.4f}s, "
                    f"Captured Image={img_filepath}"
                )

                print("Prediction complete!")
                print("Predicted Class:", CLASS_DEF[predicted_class])
                print(f"{model_name} TFLite Model Size (MB):", model_size)
                print("Inference Time (s):", inference_time)

                if i < 24:  # Don't sleep after the last capture
                    time.sleep(5)  # 5 seconds between captures
            
            print("Burst mode complete! 25 images captured.")
            break  # Exit after capturing 25 images
        else:
            if GPIO.input(SIGNAL_PIN) == GPIO.HIGH:
                print("Trigger detected! Capturing image and running inference...")
                img_filepath = capture_and_save_image()

                model_filepath = get_model_filepath(base_model, quant, greyscale)
                model_name = get_model_name(base_model, quant, greyscale)
                interpreter = tflite.Interpreter(model_path=model_filepath)
                #interpreter = tf.lite.Interpreter(model_path=model_filepath)
                interpreter.allocate_tensors()

                # Measure inference time
                start_time = time.time()
                predicted_class = evaluate_tflite_model(interpreter, img_filepath)
                inference_time = time.time() - start_time

                # Log prediction details
                model_size = os.path.getsize(model_filepath) / float(2**20)
                logging.info(
                    f"Model Invoked={model_name}, "
                    f"Motion Detected: Class={CLASS_DEF[predicted_class]}, "
                    f"Model Size={model_size:.4f}MB, " f"Inference Time={inference_time:.4f}s, "
                    f"Captured Image={img_filepath}"
                )

                print("Prediction complete!")
                print("Predicted Class:", CLASS_DEF[predicted_class])
                print(f"{model_name} TFLite Model Size (MB):", model_size)
                print("Inference Time (s):", inference_time)

                # Send the detected class name to Arduino if the class is not "Non-Human"
                class_name = CLASS_DEF[predicted_class]
                print(f"Sending class '{class_name}' to Arduino.")
                arduino_serial.write(f"{class_name}\n".encode())

                # Wait for Arduino response
                while True:
                    if arduino_serial.in_waiting > 0:
                        arduino_response = arduino_serial.readline().decode().strip()
                        print(f"Arduino echo: {arduino_response}")
                        break

                time.sleep(10)  # Debounce delay
                print("Waiting for motion...")
            time.sleep(0.1)  # Polling delay
except KeyboardInterrupt:
    print("Exiting program.")
finally:
    GPIO.cleanup()
    arduino_serial.close()
