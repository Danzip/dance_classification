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

plt.rcParams["figure.dpi"] = 200
plt.rcParams.update({'font.size': 15})



# get params for modules
st.header('Model hyperparams')
#batch_size = st.number_input('Batch size',value=5, step=1)
#n_frames_skip = st.number_input('Number of frames to skip', value=6, step=1)
reset_buffer_every_n_seconds = st.number_input('Reset buffer every N batches',value=3, step=1)



## initialize all modules
loader = VideoLoader()

def load_video():
    uploaded_file = st.file_uploader(label='Pick an video to test',type = ['mp4'])
    if uploaded_file is not None:
        video_data = uploaded_file.getvalue()
        st.video(video_data,format="video")
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_data)
        video,fps = loader.load_video_for_classification(tfile.name)
        batch_size = 5
        n_frames_skip = round(fps/5)
        m = ModelInference(batch_size=batch_size,n_frames_skip=n_frames_skip,reset_buffer_every_n_batches=reset_buffer_every_n_seconds)

        start = time.time()
        df_out = m.analyse_vid(video, int(fps), return_df=True, show_plot=False,return_fig=False)
        end = time.time()
        st.header(f'Inference time: {end - start:.2f} seconds')
        fig, ax = plt.subplots()
        sns.lineplot(x='index',y="Probabilities",ax=ax, hue='cols', data=df_out,alpha=0.6,legend=False)
        sns.scatterplot(x='index',y="Probabilities",ax=ax, hue='cols', data=df_out)
        plt.xlabel('Time in seconds')
        #plt.legend()
        plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
        st.pyplot(fig)


def main():
    st.title('Video upload demo')
    load_video()


if __name__ == '__main__':
    main()