import numpy as np
import cv2
import torch
from typing import Tuple
import os

class VideoLoader:
    def __init__(self, crop_size:int=172):
        self.crop_size = crop_size


    def load_video_for_detection(self, path:str):
        """
        Loads a video resizes it and converts it to a tensor of frames
        
        :param path: The path to the video file
        :return: a tensor of frames and the fps of the video.
        """
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

        finally:
            cap.release()
        return torch.Tensor(np.array(frames)/1.).permute(3, 0, 1, 2), fps

    def crop_center_square(self, frame):
        """
        Crop the center square of a frame and return it
        :param frame: The frame to crop
        :return: The cropped image.
        """
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]
    
    def load_video_for_classification(self, path:str,resize=(172, 172)):
    
        """
        Loads a video resizes it and converts it to a tensor of frames
        
        :param path: The path to the video file
        :param max_frames: The maximum number of frames to load.
        :param resize: Resize the input frames to this size
        :return: a tensor of frames and the fps of the video.
        """
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.crop_center_square(frame)
                #frame = resize2SquareKeepingAspectRation(frame,172)
                frame = cv2.resize(frame, resize)
                #frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)


        finally:
            cap.release()
        return torch.Tensor(np.array(frames)/255.).permute(3, 0, 1, 2), fps

    
    def resize_with_pad(self, image: np.array, 
                        new_shape: Tuple[int, int], 
                        padding_color: Tuple[int,int,int] = (0, 0, 0)) -> np.array:
        """Maintains aspect ratio and resizes with padding.
        Params:
            image: Image to be resized.
            new_shape: Expected (width, height) of new image.
            padding_color: Tuple in BGR of padding color
        Returns:
            image: Resized image with padding
        """
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(new_shape))/max(original_shape)
        new_size = tuple([int(x*ratio) for x in original_shape])
        image = cv2.resize(image.astype('float32'), new_size)
        delta_w = new_shape[0] - new_size[0]
        delta_h = new_shape[1] - new_size[1]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
        return image.astype(int)


    def pad_video(self, video, bbox, crop_size=172):
        """
        It takes a video, and a bounding box, and returns a video where the bounding box is centered in the
        frame
        
        :param video: the video to be padded
        :param bbox: bounding box of the person in the video
        :param crop_size: the size of the cropped video, defaults to 172 (optional)
        :return: A tensor of shape (3, num_frames, crop_size, crop_size)
        """
        x_1, y_1, x_2, y_2 = bbox.long()

        video = video[:,:,y_1:y_2,x_1:x_2]

        out = torch.zeros(3,video.shape[1],crop_size,crop_size)
        for i,frame in enumerate(video.permute(1,2,3,0)):
            padded_frame = self.resize_with_pad(np.array(frame),new_shape=(crop_size,crop_size))
            out[:,i,:,:] = torch.Tensor(padded_frame.transpose(2,0,1))/255.

        return out


    