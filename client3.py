from socketio import Client
import pyaudio
import wave
import os
import time
from datetime import datetime

# Connect to the server
sio = Client()
sio.connect('http://localhost:4020')

def send_audio(client_name, x, y):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5  # Adjust the recording duration as needed

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    return {
        'audio_data': b''.join(frames),
        'client_name': client_name,
        'x': x,  # Client's x-coordinate on the radar
        'y': y   # Client's y-coordinate on the radar
    }

client_name = "Audio Sensor 3"  # Get the client's name
x_coordinate = 290  # Specify the client's x-coordinate on the radar
y_coordinate = 175  # Specify the client's y-coordinate on the radar

while True:
    # Capture audio every 10 seconds and send it to the server
    audio_info = send_audio(client_name, x_coordinate, y_coordinate)
    sio.emit('audio_from_client', audio_info)

    time.sleep(2)  # Wait for 2 seconds before capturing again

# Disconnect from the server
sio.disconnect()
