from main import generator, dct, hifigan, infer, output_sampling_rate
from io import StringIO
from flask import Flask, make_response, request
import soundfile as sf
import torch

app = Flask(__name__)

@app.route('/speak', methods=['POST'])
def speak():

    buf = StringIO()
    data = request.get_json()
    input_text = data["text"]

    # generate_wav_file should take a file as parameter and write a wav in it
    y = infer(input_text, generator, dct)

    with torch.no_grad():
        audio = hifigan.forward(y).cpu().squeeze().clamp(-1, 1)

    sf.write(buf, audio, output_sampling_rate)

    response = make_response(buf.getvalue())
    buf.close()

    response.headers['Content-Type'] = 'audio/wav'
    response.headers['Content-Disposition'] = 'attachment; filename=sound.wav'

    return response