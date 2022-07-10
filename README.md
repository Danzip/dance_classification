# Introduction
This work was done in an internship at Lightricks.

The tasks were:
- Basic - Classify action in a video of arbitrary length.
- Strech goal 1 - Classify subsequent actions in a video of arbitrary length.
- Strech goal 2 - Classify multiple subsequent actions in a video of arbitrary length. (Another repo)

** The solutions must be lightweight and run at realtime or better. **

We chose to compare models by their performance on [kinetics 600](https://www.deepmind.com/open-source/kinetics "kinetics 600"). 
We ended up using the [pytorch implementation](https://github.com/Atze00/MoViNet-pytorch "pytorch implementation") of [MoViNets](https://arxiv.org/pdf/2103.11511.pdf "MoViNets").

Benefits of the MoViNets Stream Buffers :
-  Allow the usage of constant memory at inference time.
-  Takes into account longer temporal relationships.

## Demo webapp (Streamlit)
To install:


```
git clone https://github.com/bf2harven/Lightricks_all_video.git
cd Lightricks_all_video
conda env create -f environment.yml
conda activate lightricks_env
```

To run a streamlit webapp of our model:
```
streamlit run main.py
```

## In the webapp:

- You can select the granularity of the classification. (How many classes)\
- You can select a file to run inference on.\
- You can select how often the buffer is resetted. \
If this is too short - there's not enough information to make a good classification.\
If this is too long - There will be a lag in detection of new actions. Can even miss short actions. 





![alt text](https://github.com/bf2harven/Lightricks_all_video/blob/main/stl_setting.png?raw=true)

![alt text](https://github.com/bf2harven/Lightricks_all_video/blob/main/probs.png?raw=true)

