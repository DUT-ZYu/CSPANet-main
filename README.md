# [Style transfer] CSPANet: Cross-Route Statistical Partition Attention Network for Style Transfer
The paper will be soon to release! 
## Usage

**To run our code, please follow these steps:**
 Training dataset:[MS-COCO/WikiArt](https://www.kaggle.com/datasets/shaorrran/coco-wikiart-nst-dataset-512-100000)

## Environment
You can visit [Kaggle ((https://www.kaggle.com))] 

## Framework
![DEMAL](https://github.com/ZYDeeplearning/DEMAL_Model/blob/main/1.png)
The overall pipeline of our DEMAL framework.
## Results
![DEMAL](https://github.com/ZYDeeplearning/DEMAL_Model/blob/main/2.png)
![DEMAL](https://github.com/ZYDeeplearning/DEMAL_Model/blob/main/3.png)
![DEMAL](https://github.com/ZYDeeplearning/DEMAL_Model/blob/main/4.png)

## Pretrained models
You can visit [Google(https://drive.google.com/drive/folders/1F1v_PDSe0CVtIE2o8ow6uE_yICbas2Y2)],includings the pre-trained model weights and the VGG weight!

## Run CSAPNet

```
python main.py  "" # Training
```
```
python main.py --isTest True  "" # Test
```

## Evaluation

Before executing evalution code, please duplicate the content and style images to match the number of stylized images first. (40 styles, 20 contents -> 800 style images, 800 content images)

run:
```
python util/copy_inputs.py --cnt data/cnt --sty data/sty
```

We largely employ [matthias-wright/art-fid](https://github.com/matthias-wright/art-fid) and [mahmoudnafifi/HistoGAN](https://github.com/mahmoudnafifi/HistoGAN) for our evaluation.

### Art-fid
run:
```
cd evaluation;
python eval_artfid.py --sty your real-style images path --cnt your real-style images path --tar generated images path
```

