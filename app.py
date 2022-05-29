from main import generator, dct, hifigan, infer, output_sampling_rate
from io import StringIO
from flask import Flask, make_response, request
import io
from scipy.io.wavfile import write
import base64
import torch

app = Flask(__name__)

@app.route('/speak', methods=['POST'])
def speak():

    data = request.get_json()
    input_text = data["text"]

    # generate_wav_file should take a file as parameter and write a wav in it
    y = infer(input_text, generator, dct)

    with torch.no_grad():
        audio = hifigan.forward(y).cpu().squeeze().clamp(-1, 1).detach().numpy()

    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, output_sampling_rate, audio)
    wav_bytes = byte_io.read()

    audio_data = base64.b64encode(wav_bytes).decode('UTF-8')

    response = make_response({"speech": audio_data})

    return response