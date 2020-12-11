import gradio
import torch
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


def highpassf_gradio_fn(audio_file, result_mode):

    mix = nussl.AudioSignal(audio_file.name)
    audio_signal = mix

    sep = nussl.separation.benchmark.HighLowPassFilter(
        audio_signal, high_pass_cutoff_hz=300)

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


def hpss_gradio_fn(audio_file, result_mode):

    mix = nussl.AudioSignal(audio_file.name)
    audio_signal = mix

    sep = nussl.separation.primitive.HPSS(audio_signal)

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


def repet_gradio_fn(audio_file, result_mode):

    mix = nussl.AudioSignal(audio_file.name)
    audio_signal = mix

    sep = nussl.separation.primitive.Repet(
        audio_signal, mask_type='binary')

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


def timber_gradio_fn(audio_file, result_mode):

    mix = nussl.AudioSignal(audio_file.name)
    audio_signal = mix

    # model_path = nussl.efz_utils.download_trained_model(
    #     'dpcl-wsj2mix-model.pth')

    # model = torch.load(model_path, map_location=torch.device('cpu'))
    # model["nussl_version"] = "1.0.0"

    # torch.save(model, model_path)

    sep = nussl.separation.primitive.TimbreClustering(
        audio_signal, 2, 50, mask_type='binary')
    print("hi")
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


gradio_fns = {

    "repet": repet_gradio_fn,
    "highpass": highpassf_gradio_fn,
    "timber": timber_gradio_fn,
    "hpss": hpss_gradio_fn

}

examples = {

    "hpss": [
        ["example_audios/simple_sa_with_clap.mp3", "audio"],
        ["example_audios/sa_with_teen_taal.mp3", "audio"],
        ["example_audios/sargam_with_clap.mp3", "audio"],
        ["example_audios/sargam_with_teen_taal.mp3", "audio"]
    ],
    "highpass": [
        ["example_audios/hpassdefault.mp3", "audio"],
        ["example_audios/hpassftest.wav", "audio"]
    ],
    "repet": [
        ["example_audios/raag_bhairavi_by_kaushiki.m4a", "audio"],
    ],
    "timber": [
        ["example_audios/timber_test.m4a", "audio"],
        ["example_audios/marimba_timbre.mp3", "audio"]
    ]

}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='host gradio backend for source separation')
    parser.add_argument(
        '--algo', type=str, help='name of the source separation algorithm', required=True)

    args = parser.parse_args()

    audio_in = gradio.inputs.Audio(source="upload", type="file", label="Input")

    func = gradio_fns[args.algo]

    gradio.Interface(
        fn=func,
        inputs=[
            audio_in,
            gradio.inputs.Radio(
                ["Separated Audios", "Overlaid Spectrogram", "Overlaid Waveforms"]),
        ],
        outputs=[gradio.outputs.HTML()],
        examples=examples[args.algo],
        server_name="127.0.0.1",
        allow_flagging=False
    ).launch(share=True)
