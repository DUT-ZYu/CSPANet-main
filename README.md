# [Style transfer] CSPANet: Cross-Route Statistical Partition Attention Network for Style Transfer
The paper will be soon to release! 
## Usage

## Abstract
Attention-based style transfer methods have made remarkable advancements in generating high-quality stylized images. However, they often struggle with an essential duality: maintaining precise content structure while reproducing both fine-grained style details and holistic visual-concept from reference artworks. To address these issues, we propose a Cross-Route Statistical Partition Attention Network (CSPANet) for style transfer. Specifically, our proposed CSPANet is a novel stylization framework that can learn cross-route style information for constructing realistic style details of generated stylized images by an intuitive feature processing. Next, we design a statistical partition attention migrator (SPAM). It not only can enhance the representation of style semantics correlation by learning global dependency and local awareness, but also depict the captured visual-concept information into the content structure in a coarse-to-fine manner. Furthermore, we construct a detail-aware skip connection (DSC) between the encoder and decoder. It can dynamically inject shallow information into the decoder, improving content detail structure retention and overall stylized quality. Extensive experiments have verified the effectiveness of our proposed CSPANet, surpassing previous state-of-the-art style transfer methods.

**To run our code, please follow these steps:**
 Training dataset:[MS-COCO/WikiArt](https://www.kaggle.com/datasets/shaorrran/coco-wikiart-nst-dataset-512-100000)

## Environment
You can visit [Kaggle ((https://www.kaggle.com))] 

## Framework
![CSPANet](https://github.com/DUT-ZYu/CSPANet-main/blob/main/Figure_2_01.jpg)
The overall pipeline of our CSPANet framework.
## Results image style transfer
![CSPANet- ](https://github.com/DUT-ZYu/CSPANet-main/blob/main/Figure_8_01.jpg)
## Results video style transfer
![CSPANet- ](https://github.com/DUT-ZYu/CSPANet-main/blob/main/Figure_12_01.jpg)

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

