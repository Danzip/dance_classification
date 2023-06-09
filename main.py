import streamlit as st
from separate_classes.ModelInference import ModelInference
import time
from separate_classes.VideoLoader import VideoLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import torch
import numpy as np
import tempfile
import seaborn as sns
import os

from utils import KINETICS_CLASSES, DANCE_CLASSES_5, DANCE_CLASSES_18

plt.rcParams["figure.dpi"] = 200
plt.rcParams.update({'font.size': 15})

loader = VideoLoader()

# get params for modules
st.header('Model hyperparams')
# batch_size = st.number_input('Batch size',value=5, step=1)
# n_frames_skip = st.number_input('Number of frames to skip', value=6, step=1)
# reset_buffer_every_n_seconds = st.number_input('Reset buffer every N seconds',value=3, step=1)
# label_granularity = st.number_input('Select granularity',value=, step=1)

label_granularity = st.radio(f'Select Classes amount',
                             [f"{KINETICS_CLASSES}", f"{DANCE_CLASSES_5}", f"{DANCE_CLASSES_18}"],
                             index=0)
model_id = st.radio('Select model,a0 - a5', ["a0", "a1", "a2", "a3", "a4", "a5"], index=0)
upload_bool = st.radio('Upload or read local video',
                       ['Upload', 'Read file from disk'], index=0)


# st.write(label_granularity, type(label_granularity))

def file_selector(folder_path='example_vids', key=1):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames, key=key)
    return os.path.join(folder_path, selected_filename)


def run_inference():
    if upload_bool == 'Read file from disk':
        col1, col2 = st.columns(2)
        fname = file_selector()
        reset_buffer_every_n_seconds = st.number_input('Reset buffer every N seconds', value=50, step=1)
        with col1:
            st.video(fname, format="video")
        video, fps = loader.load_video_for_classification(fname)
        batch_size = 5
        n_frames_skip = round(fps / 5)
        m = ModelInference(batch_size=batch_size, n_frames_skip=n_frames_skip,
                           reset_buffer_every_n_batches=reset_buffer_every_n_seconds,
                           label_granularity=label_granularity, model_id=model_id)
        start = time.time()
        df_out = m.analyse_vid(video, int(fps), return_df=True, show_plot=False)
        end = time.time()
        st.header(f'Inference time: {end - start:.2f} seconds')
        st.write(f'Video length: {video.shape[1] / fps:.2f} seconds')
        st.write(f'Time ratio: {(end - start) / (video.shape[1] / fps):.2f} seconds')
        fig, ax = plt.subplots()
        sns.lineplot(x='index', y="Probabilities", ax=ax, hue='cols', data=df_out, alpha=0.6, legend=False)
        sns.scatterplot(x='index', y="Probabilities", ax=ax, hue='cols', data=df_out)
        plt.xlabel('Time in seconds')
        # plt.legend()
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        # with col2:
        st.pyplot(fig)
    elif upload_bool == 'Upload':
        uploaded_file = st.file_uploader(label='Pick a video to test', type=['mp4'])
        reset_buffer_every_n_seconds = st.number_input('Reset buffer every N seconds', value=50, step=1)

        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            video_data = uploaded_file.getvalue()
            with col1:
                st.video(video_data, format="video")
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_data)
            video, fps = loader.load_video_for_classification(tfile.name)
            batch_size = 5
            n_frames_skip = round(fps / 5)
            m = ModelInference(batch_size=batch_size, n_frames_skip=n_frames_skip,
                               reset_buffer_every_n_batches=reset_buffer_every_n_seconds,
                               label_granularity=label_granularity, model_id=model_id)
            start = time.time()
            df_out = m.analyse_vid(video, int(fps), return_df=True, show_plot=False)
            end = time.time()
            st.header(f'Inference time: {end - start:.2f} seconds')
            st.write(f'Video length: {video.shape[1] / fps:.2f} seconds')
            st.write(f'Time ratio: {(end - start) / (video.shape[1] / fps):.2f} seconds')
            fig, ax = plt.subplots()
            sns.lineplot(x='index', y="Probabilities", ax=ax, hue='cols', data=df_out, alpha=0.6, legend=False)
            sns.scatterplot(x='index', y="Probabilities", ax=ax, hue='cols', data=df_out)
            plt.xlabel('Time in seconds')
            # plt.legend()
            plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
            with col2:
                st.pyplot(fig)


def main():
    st.title('Inference demo')
    run_inference()


if __name__ == '__main__':
    main()
