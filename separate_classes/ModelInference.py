import re
import sys

from movinets import MoViNet as mn
from movinets.config import _C
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from io import StringIO
import streamlit as st


def extract_number(string):
    pattern = r'\d+'  # Matches one or more digits
    match = re.search(pattern, string)

    if match:
        return int(match.group())
    else:
        return None


def load_model(model_id):
    model_name = f"MoViNet{model_id.upper()}"
    num = extract_number(model_name)
    casual = True if num < 3 else False  # package movinets doesn't support streaming models for versions a3 and up
    return mn(getattr(_C.MODEL, model_name), causal=casual, pretrained=True)


class ModelInference:
    def __init__(self, batch_size=10, top_n=5, n_frames_skip=6, reset_buffer_every_n_batches=3, model_id="a0",
                 label_granularity=1) -> None:
        """
        Initializes model that loads a video file, and returns a list of the top n most likely labels for the video
        
        :param batch_size: The number of frames to be processed at a time, defaults to 10 (optional)
        :param top_n: the number of top predictions to return, defaults to 5 (optional)
        :param n_frames_skip: number of frames to skip between predictions, defaults to 6 (optional)
        :param reset_buffer_every_n_batches: This is the number of batches that the model will process
        before resetting the buffer, defaults to 3 (optional)
        :param return_df: If True, returns a pandas dataframe with the predictions. If False, returns a
        dictionary with the predictions, defaults to False (optional)
        :param show_plot: If True, will show a plot of the predictions, defaults to False (optional)
        :param savefig: If True, saves the plot of the predictions, defaults to False (optional)
        :param label_granularity: 1 or 2. 1 is the default, and it's the more granular of the two, defaults
        to 1 (optional)
        """
        # Create a StringIO object to capture the log output
        log_output = StringIO()
        sys.stdout = log_output
        sys.stderr = log_output
        st.text("Loading model...")
        # Perform the download and log the progress
        self.model = load_model(model_id).eval()

        # Reset the stdout to the default to restore normal printing
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # Display the log output in the Streamlit app
        st.text(log_output.getvalue())

        self.batch_size = batch_size
        self.top_n = top_n
        self.n_frames_skip = n_frames_skip
        self.reset_buffer_every_n_batches = reset_buffer_every_n_batches
        if label_granularity == 1:
            with open(os.path.join(os.getcwd(), 'separate_classes', 'CONV_MATRIX_ALL.pickle'), 'rb') as f:
                self.CONV_MATIX = pickle.load(f)
            with open(os.path.join(os.getcwd(), 'separate_classes', 'k600_all.pkl'), 'rb') as f:
                self.D1 = pickle.load(f)
        if label_granularity == 2:
            with open(os.path.join(os.getcwd(), 'separate_classes', 'CONV_MATIX.pkl'), 'rb') as f:
                self.CONV_MATIX = pickle.load(f)
            with open(os.path.join(os.getcwd(), 'separate_classes', 'k600_reduced_labels.pkl'), 'rb') as f:
                self.D1 = pickle.load(f)
        elif label_granularity == 3:
            with open(os.path.join(os.getcwd(), 'separate_classes', 'CONV_MATIX2.pkl'), 'rb') as f:
                self.CONV_MATIX = pickle.load(f)
            with open(os.path.join(os.getcwd(), 'separate_classes', 'k600_reduced_labels2.pkl'), 'rb') as f:
                self.D1 = pickle.load(f)
        self.class_id_dict = pd.DataFrame(self.D1.keys()).to_dict()[0]

    def get_preds(self, video, fps):
        """
        This function is used when tracking. It takes a video and returns a list of the top n most likely labels for the video
        The function then reduces the labels by multiplting by a conversion matrix.
                
        :param video: the video to be processed
        :param fps: frames per second of the video
        :return: Vector with merged logits of the model.
        """
        # loading data and subsampling

        inputs = video[:, ::self.n_frames_skip, :, :]
        inputs = inputs[None, :]

        # prealloc
        seconds = 0
        df = pd.DataFrame(self.D1.keys())
        df.set_index(0, inplace=True)

        # inference
        self.model.clean_activation_buffers()
        with torch.no_grad():
            self.model.clean_activation_buffers()
            output = F.softmax(self.model(inputs), dim=1)
            output = output @ self.CONV_MATIX
            return output.numpy().flatten()

    def analyse_vid(self, video, fps, return_df=False, show_plot=False, return_fig=True):
        """
        It takes a video path, loads the video, subsamples it, runs it through the model, and plots the top n
        predictions, also can plot the predictions in a lineplot.
        
        :param video_path: path to the video file
        :param batch_size: The number of frames to be processed in per batch, defaults to 10 (optional)
        :param top_n: the number of top predictions to return, defaults to 5 (optional)
        :param n_frames_skip: The number of frames to skip between each frame, defaults to 6 (optional)
        :param reset_buffer_every_n_batches: , defaults to 3 (optional)
        :param return_df: if True, returns a dataframe with the probabilities for each class for each time
        step, defaults to False (optional)
        :return: A dataframe with the probabilities of each class for each time step.
        """
        # loading data and subsampling

        inputs = video[:, ::self.n_frames_skip, :, :]
        inputs = inputs[None, :]

        # prealloc
        seconds = 0
        df = pd.DataFrame(self.D1.keys())
        df.set_index(0, inplace=True)

        # inference
        self.model.clean_activation_buffers()
        with torch.no_grad():
            for batch in range(inputs.shape[2] // self.batch_size):
                if batch % self.reset_buffer_every_n_batches == 0:
                    self.model.clean_activation_buffers()
                output = F.softmax(
                    self.model(inputs[:, :, self.batch_size * batch:self.batch_size * (1 + batch), :, :]), dim=1)
                output = output @ self.CONV_MATIX
                df[str(seconds)] = output.numpy().flatten()
                seconds += int(self.batch_size * self.n_frames_skip / fps)
        # plotting
        df['mean'] = df.mean(axis=1)
        df = df.sort_values('mean', ascending=False).head(self.top_n).drop('mean', axis=1)
        df = df.T.reset_index().melt('index', var_name='cols', value_name='Probabilities')

        df['index'] = df['index'].astype(int)
        if show_plot:
            g = sns.catplot(x="index", y="Probabilities", hue='cols', data=df, kind='point', legend=False)
            plt.xlabel('Time in seconds')
            plt.legend()
            # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(min(10,seconds)))
            # plt.ylim(0,1.1)

        if return_df:
            return df

        if return_fig:
            return g


if __name__ == "__main__":
    model_infer = ModelInference(model_id="a5")
    print(model_infer)
