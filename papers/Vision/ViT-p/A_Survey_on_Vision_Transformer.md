# A Survey on Vision Transformer.

[url][https://ieeexplore.ieee.org/abstract/document/9716741?casa_token=emECO6fhE74AAAAA:LAtRPoD6lSdS6JSfyMYUA_awDr4ZmEK8Zn_CwWgt9VmcP8lKoHW0J3nukw0JAEQlQeE4DvCvPp4]

CNN introduction : processing shift-invariant data such as images

RNN introduction : Sequential data, time-series data

### DERT (end-to-end object detection with transformer)

Facebook ai : https://arxiv.org/abs/2005.12872

Object detection : instance classes + bounding box problem

###  Max-DeepLab (End-to-End Panoptic Segmentation with Mask Transformers)

Q : embedding patch 를 각각 다른 크기로는 만들 수 없을까? 얘를 들면 이미 segmentation이 되어있는 것들로. 이미지를 이해할때 먼저 결계선이 제일 눈에 띄니까..
Sharpening Spatial Filter를 써서 먼저 구역을 구별해보면 어떨런지..
laplacian filter etc..


<!--computational graphs used : 현재 있느 데이터를 그래프로 표현해야한다. 

static 하게 그래프를 그리다. 실행 시점에서 그래프가 그려지는것. 

pytorch ; 실행시점에서 그래프가 그려짐 (dynamic computational graph, DCG)
실행을 하면서 그래프를 생성하는 방식


텐서프로우 : 그래프를 먼저 정의하는 코드 작성 -> 실행 시점에서 데이터를 feed
데이터를 넣어주고 코드를 작성 
-->

translation equivariance and locality : if insufficient amounts of data일떄, 일반화를 못함.

#### VIT vs CNN

VIT : lack of ability to extract local information




