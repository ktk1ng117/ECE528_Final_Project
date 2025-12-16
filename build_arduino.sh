SKETCH_PATH="$HOME/mahadd/arduino/arduino.ino"
BOARD="arduino:avr:nano"
PORT="/dev/ttyUSB0"

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