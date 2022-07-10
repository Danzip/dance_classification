# Lightricks_all_video

### Introduction
A streamlit app that receives an mp4 file and returns the class 


To install:

clone repo
```
git clone https://github.com/bf2harven/Lightricks_all_video.git
```

```
cd Lightricks_all_video
conda env create -f environment.yml
conda activate lightricks_env
```

To run:
```
streamlit run main.py
```

## In the webapp:

You can select the granularity of the classification. (How many classes)\
You can select a file to run inference on.\
You can select how often the buffer is resetted. \
If this is too short - there's not enough information to make a good classification.\
If this is too long - There will be a lag in detection of new actions. Can even miss short actions. 





![alt text](https://github.com/bf2harven/Lightricks_all_video/blob/main/stl_setting.png?raw=true)

![alt text](https://github.com/bf2harven/Lightricks_all_video/blob/main/probs.png?raw=true)


Credits
