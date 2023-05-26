import glob
import random

import cv2
import numpy as np
import pandas as pd
import torch
from movinets import MoViNet as mn
from movinets.config import _C
import torch.nn.functional as F

import re

from tqdm import tqdm
import os
import torchvision.transforms as transforms

SEPERATE_CLASSES = "separate_classes"
SHORT_DANCES_CSV = os.path.join(SEPERATE_CLASSES, 'short_dances.csv')
LONG_DANCES_CSV = os.path.join(SEPERATE_CLASSES, 'long_dances.csv')
KINETICS_600_CSV = os.path.join(SEPERATE_CLASSES, "kineticks600_classes")
CLASS_NUM = "class_num"
CLASS_LABEL = "class_label"
NUM_FRAMES = 8
model_id = "a0"
RESOLUTION = 224
OUTPUT_SIZE = (RESOLUTION, RESOLUTION)
BATCH_SIZE = 8


def get_class_name_from_path(path, label_map):
    for class_label in label_map:
        dir_label_name = class_label.replace(" ", "_")
        if dir_label_name in path:
            return class_label
    return None


def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
        frame: Image that needs to be resized and padded.
        output_size: Pixel size of the output frame image.

    Return:
        Formatted frame with padding of specified output size.
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(output_size),
                                    transforms.Pad(
                                        (0, 0, output_size[0] - frame.size[0], output_size[1] - frame.size[1]))
                                    ])

    frame = transform(frame)
    return frame


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


def frames_from_video_file(video_path, n_frames, output_size=OUTPUT_SIZE, frame_step=15):
    """
      Creates frames from each video file present for each category.

      Args:
        video_path: File path to the video.
        n_frames: Number of frames to be created per video file.
        output_size: Pixel size of the output frame image.

      Return:
        An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    # Read each video frame by frame
    result = []
    try:
        src = cv2.VideoCapture(str(video_path))

        video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

        need_length = 1 + (n_frames - 1) * frame_step

        if need_length > video_length:
            start = 0
        else:
            max_start = video_length - need_length
            start = random.randint(0, max_start + 1)

        src.set(cv2.CAP_PROP_POS_FRAMES, start)
        # ret is a boolean indicating whether read was successful, frame is the image itself
        ret, frame = src.read()
        result.append(format_frames(frame, output_size))

        for _ in range(n_frames - 1):
            for _ in range(frame_step):
                ret, frame = src.read()
            if ret:
                frame = format_frames(frame, output_size)
                result.append(frame)
            else:
                result.append(np.zeros_like(result[0]))
        src.release()
        result = np.array(result)[..., [2, 1, 0]]
        result = np.expand_dims(result, axis=0)
    except Exception as e:
        print(e)

    return result


class MoviNet:
    def __init__(self, model_id,n_clip_frames = NUM_FRAMES):
        """
        :param model_id: movinets model_id valid values are a0-a5
        """
        self.model = load_model(model_id)
        self.n_clip_frames = n_clip_frames

    def get_preds(self, video, fps):
        """
        This function is used when tracking. It takes a video and returns a list of the top n most likely labels for the video
        The function then reduces the labels by multiplting by a conversion matrix.

        :param video: the video to be processed
        :param fps: frames per second of the video
        :return: Vector with merged logits of the model.
        """
        # loading data and subsampling

        inputs = video[:, ::self.n_frames_clip, :, :]
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

    def infer_on_dir(self, video_dir, ext="mp4", batch_size=8, output_size=OUTPUT_SIZE, n_frames=NUM_FRAMES,
                     output_csv='output.csv'):
        video_paths = glob.glob(video_dir + f'/**/*.{ext}', recursive=True)

        df = pd.DataFrame(columns=['video_path', 'label', 'prediction', 'probabilities'])

        for batch_num in tqdm(range(int(np.ceil((len(video_paths) / batch_size))))):
            batch_video_paths = video_paths[batch_num * batch_size:(batch_num + 1) * batch_size]
            batch_list = [frames_from_video_file(video_path, n_frames, output_size) for video_path in batch_video_paths]
            filtered_indexes = [index for index, value in enumerate(batch_list) if
                                isinstance(value, np.ndarray) or np.any(value)]
            filtered_batch = [batch_list[i] for i in filtered_indexes]
            filtered_batch_video_paths = [batch_video_paths[i] for i in filtered_indexes]
            batch_arrays = np.concatenate(filtered_batch, axis=0)
            predicted_labels, probabilities = self.predict(batch_arrays)
            class_labels = [get_class_name_from_path(video_path, self.label_map) for video_path in
                            filtered_batch_video_paths]
            # probabilities_str = [np.array2string(arr, separator=',') for arr in probabilities.numpy()]

            batch_df = pd.DataFrame({
                'video_path': filtered_batch_video_paths,
                'label': class_labels,
                'prediction': predicted_labels,
                # 'probabilities': probabilities_str
            })

            df = pd.concat([df, batch_df], ignore_index=True)

        if os.path.exists(output_csv):
            df.to_csv(output_csv, mode='a', index=False, header=False)
        else:
            df.to_csv(output_csv, index=False)  # Save the dataframe to CSV file

        return df


if __name__ == "__main__":
    evaluate = MoviNet("a3")

    print(evaluate.model)
