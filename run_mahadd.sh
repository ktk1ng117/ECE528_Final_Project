#!/bin/bash

# === Configuration ===
VENV_PATH="$HOME/mahadd/venv"
PYTHON_SCRIPT="$HOME/mahadd/python/mahadd.py"
SKETCH_PATH="$HOME/mahadd/arduino/arduino.ino"
BOARD="arduino:avr:nano"
PORT="/dev/ttyUSB0"

# === Activate virtual environment ===
echo "Activating Python virtual environment..."
source "$VENV_PATH/bin/activate"

# === Compile and upload Arduino sketch ===
echo "Compiling Arduino sketch..."
arduino-cli compile --fqbn $BOARD $SKETCH_PATH
if [ $? -ne 0 ]; then
    echo "Arduino compilation failed!"
    exit 1
fi

echo "Uploading Arduino sketch to $PORT..."
arduino-cli upload -p $PORT --fqbn $BOARD $SKETCH_PATH
if [ $? -ne 0 ]; then
    echo "Arduino upload failed!"
    exit 1
fi
echo "Arduino upload successful!"

# === Set environment variables to suppress libcamera/TFLite logs ===
export LIBCAMERA_LOG_LEVELS=1
export LIBCAMERA_LOG_LEVEL=1
export LIBCAMERA_LOG_LEVEL_CAMERA=1
export LIBCAMERA_LOG_LEVEL_RPI=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

# === Run Python script ===
echo "Running Python script..."
python3 "$PYTHON_SCRIPT" "$@"

# === Deactivate venv when done ===
deactivate
