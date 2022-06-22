from visualization import VideoVisualizer
import numpy as np
import torch


class Visualizer:
    def __init__(self, initialized_model, initialized_detector, initialized_loader,  top_n=3):
        self.classes_dict = initialized_model.class_id_dict
        self.top_n = top_n
        self.num_classes = len(self.classes_dict.keys())
        self.visualizer = VideoVisualizer(self.num_classes, self.classes_dict, top_k=3, mode="top-k")
        self.initialized_model = initialized_model
        self.initialized_detector = initialized_detector
        self.initialized_loader = initialized_loader

    
    def get_predictions_every_n_seconds(self,video, fps, n_seconds=1):
        L = []
        for i in range(0,video.shape[1],int(n_seconds*fps)):
            subvid = video[:,i:i+fps,:,:]
            predicted_boxes = self.initialized_detector.get_person_bboxes(subvid[:,int(subvid.shape[1]/2),:,:].permute(1,2,0))
            predictions = []
            if len(predicted_boxes)<1:
                predicted_boxes = torch.Tensor([0,0,video.shape[2],video.shape[3]]).reshape(1,4)
            for bbox in predicted_boxes:
                out = self.initialized_loader.pad_video(subvid,bbox.long())
                predictions.append(self.initialized_model.get_preds(out, 30))
            
            for frame in range(subvid.shape[1]):
                out_img_pred = self.visualizer.draw_one_frame(np.array(subvid[:,frame,:,:].permute(1,2,0)).astype(int),
                torch.Tensor(np.array(predictions)), 
                predicted_boxes.long(),text_alpha = 0.2)
                L.append(out_img_pred)
        return L
    #return torch.Tensor(np.array(predictions))

    
    def foo(self, frame_list, preds_list, bbox_list):
        out_img_pred = self.visualizer.draw_one_frame(frame_list[0], preds_list, bbox_list)


