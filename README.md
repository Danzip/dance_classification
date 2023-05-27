# Introduction

The tasks were:
-  Classify dances in a video of arbitrary length.


We chose to compare models by their performance on [kinetics 600](https://www.deepmind.com/open-source/kinetics "kinetics 600"). 
We ended up using the [pytorch implementation](https://github.com/Atze00/MoViNet-pytorch "pytorch implementation") of [MoViNets](https://arxiv.org/pdf/2103.11511.pdf "MoViNets").

Benefits of the MoViNets Stream Buffers :
-  Allow the usage of constant memory at inference time.
-  Takes into account longer temporal relationships. 
-  LightWeight

## Demo webapp (Streamlit)
To install:


```
git clone https://github.com/Danzip/dance_classification
cd dance_classification
conda env create -f environment.yml
conda activate dance_classification_env
```

To run a streamlit webapp of our model:
```
streamlit run main.py
```

## In the webapp:

- You can select the granularity of the classification. (How many classes) 
- You can select a file to run inference on.
- You can select how often the buffer is resetted. \
If this is too short - there's not enough information to make a good classification.\
If this is too long - There will be a lag in detection of new actions. Can even miss short actions. 


![alt text](stl_setting.png?raw=true)


inference demo:
![alt text](probs.png?raw=true)
The repo automatically downloads the models, and saves it for future runs
