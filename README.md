# PyTorch_UCMerced_LandUse
## Dataset

[UC Merced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)

## Result
Epoch|4|100
---|---|---
LeNet| 3.33%|55.71%
ResNet18|45.48%|-
ResNet34|44.05%|74.52%

## File structure
```
├── Images
│   ├── agricultural
│   ├── airplane
│   ├── baseballdiamond
│   ├── beach
│   ├── buildings
│   ├── chaparral
│   ├── denseresidential
│   ├── forest
│   ├── freeway
│   ├── golfcourse
│   ├── harbor
│   ├── intersection
│   ├── mediumresidential
│   ├── mobilehomepark
│   ├── overpass
│   ├── parkinglot
│   ├── river
│   ├── runway
│   ├── sparseresidential
│   ├── storagetanks
│   └── tenniscourt
├── README.md
├── checkpoint
│   ├── LeNet
│   │   └── ckpt.pth
│   └── ResNet
│       └── ckpt.pth
├── main.py
├── models
│   ├── __init__.py
│   ├── lenet.py
│   └── resnet.py
└── utils.py

```

## Reference
1. Yi Yang and Shawn Newsam, "[Bag-Of-Visual-Words and Spatial Extensions for Land-Use Classification](http://faculty.ucmerced.edu/snewsam/papers/Yang_ACMGIS10_BagOfVisualWords.pdf)," ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM GIS), 2010.