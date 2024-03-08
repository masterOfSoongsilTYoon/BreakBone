# Dataset
link: https://www.kaggle.com/datasets/akshayramakrishnan28/fracture-classification-dataset?resource=download

# Introduction
FracAtlas is a musculoskeletal bone fracture dataset with annotations for deep learning tasks like classification, localization, and segmentation. The dataset contains a total of 4,083 X-Ray images with annotation in COCO, VGG, YOLO, and Pascal VOC format. The dataset is licensed under a CC-BY 4.0 license. It should be noted that to use the dataset correctly, one needs to have knowledge of medical and radiology fields to understand the results and make conclusions based on the dataset. It's also important to consider the possibility of labeling errors.

# Command saved
central train: CUDA_VISIBLE_DEVICES=1 python train.py --image_dir "./Dataset/FracAtlas/images" --json_path "./Dataset/FracAtlas/Annotations/COCO JSON/COCO_fracture_masks.json" --epoch 200 --normalize True