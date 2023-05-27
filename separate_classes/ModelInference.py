import sys

import pandas as pd
import torch
import torch.nn.functional as F
from io import StringIO
import streamlit as st

from separate_classes.VideoLoader import VideoLoader
from utils import load_movinet_model, KINETICS_PATH, get_label_map, DANCES_PATH_5, DANCES_PATH_18, \
    DANCE_CLASSES_18, KINETICS_CLASSES, DANCE_CLASSES_5


class ModelInference:
    def __init__(self, batch_size=10, top_n=5, n_frames_skip=6, reset_buffer_every_n_batches=3, model_id="a0",
                 label_granularity=1, kinetics_path=KINETICS_PATH) -> None:
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
        self.load_model_print_logs(model_id)
        if label_granularity == KINETICS_CLASSES:
            self.classes_dict = get_label_map(kinetics_path, kinetics_path)
        if label_granularity == DANCE_CLASSES_5:
            self.classes_dict = get_label_map(DANCES_PATH_5, kinetics_path)
        if label_granularity == DANCE_CLASSES_18:
            self.classes_dict = get_label_map(DANCES_PATH_18, kinetics_path)

        self.batch_size = batch_size
        self.top_n = top_n
        self.n_frames_skip = n_frames_skip
        self.reset_buffer_every_n_batches = reset_buffer_every_n_batches

    def load_model_print_logs(self, model_id):
        # Create a StringIO object to capture the log output
        log_output = StringIO()
        sys.stdout = log_output
        sys.stderr = log_output
        st.text(f"Loading model MoviNet_{model_id}...")
        # Perform the download and log the progress
        self.model = load_movinet_model(model_id).eval()

        # Reset sys.stdout and sys.stderr to their defaults
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # Calculate the number of lines in the log output
        # Split the log output into lines
        log_lines = log_output.getvalue().split("\n")

        # Display each line of the log output
        for line in log_lines:
            st.write(line)

    def choose_relevant_classes(self, predictions):
        class_indexes = list(self.classes_dict.keys())
        relevant_predictions = predictions[:, class_indexes]
        return relevant_predictions

    def analyse_vid(self, video, fps, return_df=False, show_plot=False):
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
        df = pd.DataFrame(self.classes_dict.values())
        df.set_index(0, inplace=True)

        # inference
        self.model.clean_activation_buffers()
        with torch.no_grad():
            for batch in range(inputs.shape[2] // self.batch_size):
                if batch % self.reset_buffer_every_n_batches == 0:
                    self.model.clean_activation_buffers()
                output = self.calculate_class_predictions(batch, inputs)
                df[str(seconds)] = output.numpy().flatten()
                seconds += int(self.batch_size * self.n_frames_skip / fps)
        # plotting
        df['mean'] = df.mean(axis=1)
        df = df.sort_values('mean', ascending=False).head(self.top_n).drop('mean', axis=1)
        df = df.T.reset_index().melt('index', var_name='cols', value_name='Probabilities')

        df['index'] = df['index'].astype(int)

        if return_df:
            return df

    def calculate_class_predictions(self, batch, inputs):
        predictions = self.model(inputs[:, :, self.batch_size * batch:self.batch_size * (1 + batch), :, :])
        predictions = self.choose_relevant_classes(predictions)
        predictions /= torch.max(predictions)  # normalize
        output = F.softmax(predictions, dim=1)
        return output


if __name__ == "__main__":
    model_infer = ModelInference(label_granularity="600 kinetics classes", model_id="a5")
    loader = VideoLoader()
    fname = "/media/gentex/6AC5-31BF/kineticks_dances/train/belly_dancing/0EezE5Yljg.mp4"
    video, fps = loader.load_video_for_classification(fname)

    model_infer.analyse_vid(video, fps)
    print(model_infer)
