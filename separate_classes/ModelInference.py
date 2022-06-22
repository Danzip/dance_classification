from movinets import MoViNet as mn
from movinets.config import _C
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os



class ModelInference:
    def __init__(self,batch_size=10, top_n=5, n_frames_skip=6, reset_buffer_every_n_batches=3,return_df=False, show_plot=False,savefig=False) -> None:
        self.model = mn(_C.MODEL.MoViNetA0, causal = True, pretrained = True).eval()
        self.batch_size = batch_size
        self.top_n = top_n
        self.n_frames_skip = n_frames_skip
        self.reset_buffer_every_n_batches = reset_buffer_every_n_batches
        with open(os.path.join(os.getcwd(),'separate_classes','CONV_MATIX.pkl'), 'rb') as f:
            self.CONV_MATIX = pickle.load(f)
        with open(os.path.join(os.getcwd(),'separate_classes','k600_reduced_labels.pkl'), 'rb') as f:
            self.D1 = pickle.load(f)
        self.class_id_dict = pd.DataFrame(self.D1.keys()).to_dict()[0]
        # with open('k600_all.pkl', 'rb') as f:
        #     self.D1 = pickle.load(f)
    def get_preds(self,video,fps):
          #loading data and subsampling
        
        inputs = video[:,::self.n_frames_skip,:,:]
        inputs = inputs[None, :]

        #prealloc 
        seconds = 0
        df = pd.DataFrame(self.D1.keys())
        df.set_index(0,inplace=True)

        #inference
        self.model.clean_activation_buffers()
        with torch.no_grad():
            self.model.clean_activation_buffers()
            output = F.softmax(self.model(inputs), dim=1)
            output = output @ self.CONV_MATIX
            return output.numpy().flatten()
  
            print(os.getcwd())
    def analyse_vid(self, video, fps, return_df=False, show_plot=False,return_fig=True):
        """
        It takes a video path, loads the video, subsamples it, runs it through the model, and plots the top n
        predictions
        
        :param video_path: path to the video file
        :param batch_size: The number of frames to be processed in per batch, defaults to 10 (optional)
        :param top_n: the number of top predictions to return, defaults to 5 (optional)
        :param n_frames_skip: The number of frames to skip between each frame, defaults to 6 (optional)
        :param reset_buffer_every_n_batches: , defaults to 3 (optional)
        :param return_df: if True, returns a dataframe with the probabilities for each class for each time
        step, defaults to False (optional)
        :return: A dataframe with the probabilities of each class for each time step.
        """
        #loading data and subsampling
        
        inputs = video[:,::self.n_frames_skip,:,:]
        inputs = inputs[None, :]

        #prealloc 
        seconds = 0
        df = pd.DataFrame(self.D1.keys())
        df.set_index(0,inplace=True)

        #inference
        self.model.clean_activation_buffers()
        with torch.no_grad():
            for batch in range(inputs.shape[2]//self.batch_size):
                if batch % self.reset_buffer_every_n_batches == 0:
                    self.model.clean_activation_buffers()
                output = F.softmax(self.model(inputs[:,:,self.batch_size*batch:self.batch_size*(1+batch),:,:]), dim=1)
                output = output @ self.CONV_MATIX
                df[str(seconds)] = output.numpy().flatten()
                seconds += int(self.batch_size*self.n_frames_skip/fps)
        #plotting
        df['mean'] = df.mean(axis=1)
        df = df.sort_values('mean',ascending=False).head(self.top_n).drop('mean',axis=1)
        df = df.T.reset_index().melt('index', var_name='cols',  value_name='Probabilities')

        df['index'] = df['index'].astype(int)
        if show_plot:
            g = sns.catplot(x="index", y="Probabilities", hue='cols', data=df, kind='point',legend=False)
            plt.xlabel('Time in seconds')
            plt.legend()
            # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(min(10,seconds)))
            # plt.ylim(0,1.1)


        if return_df:
            return df
        
        if return_fig:
            return g

