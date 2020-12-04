import gradio
import warnings
warnings.simplefilter('ignore')
import nussl
import matplotlib.pyplot as plt
import numpy as np
import librosa
# from common import viz

import base64
from io import BytesIO

from functools import *

import argparse


separators = {

    "duet": nussl.separation.spatial.Duet,
    "highpassf": nussl.separation.benchmark.HighLowPassFilter,
    "timberc": nussl.separation.primitive.TimbreClustering,
    "wienerf": nussl.separation.benchmark.WienerFilter,
    "hpss": nussl.separation.primitive.HPSS

}


def gradio_fn(audio_file, result_mode, separator="hpss"):

    mix = nussl.AudioSignal(audio_file.name)
    audio_signal = mix

    sep = separators[separator](audio_signal)

    estimates = sep()

    estimates = {f'Estimate {i}': s for i, s in enumerate(estimates)}

    if result_mode == "Separated Audios":
        html = nussl.play_utils.multitrack(
            estimates, ext='.mp3', display=False)
        return html

    elif result_mode == "Overlaid Spectrogram":
        tmpfile = BytesIO()
        plt.figure()
        nussl.utils.visualize_sources_as_masks(estimates,
                                               y_axis='mel', db_cutoff=-60, alpha_amount=2.0)

        plt.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

        html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
        return html

    else:

        tmpfile = BytesIO()
        plt.figure()
        nussl.utils.visualize_sources_as_waveform(
            estimates, show_legend=False)
        plt.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

        html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
        return html


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='host gradio backend for source separation')
    parser.add_argument(
        '--algo', type=str, help='name of the source separation algorithm', required=True)

    args = parser.parse_args()

    audio_in = gradio.inputs.Audio(source="upload", type="file", label="Input")

    func = partial(gradio_fn, separator=args.algo)

    gradio.Interface(
        fn=gradio_fn,
        inputs=[
            audio_in,
            gradio.inputs.Radio(
                ["Separated Audios", "Overlaid Spectrogram", "Overlaid Waveforms"]),
        ],
        outputs=[gradio.outputs.HTML()],
        examples=[["example_audios/testaudio.mp3", "audio"]]
    ).launch(share=True)
