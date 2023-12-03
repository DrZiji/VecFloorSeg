# Implementation of VectorFloorSeg: Two-Stream Graph Attention Network for Vectorized Roughcast Floorplan Segmentation

### Environment
    conda install --yes --file requirements.txt
- Install pyg following the instruction from [official site](https://pytorch-geometric.readthedocs.io/en/latest/), 
we recommend pyg==2.0.4
### Data preparation
- Download our processed data: [here](https://drive.google.com/drive/folders/1Rye_6crjcuII2LVaIwh4iDNowFqLp1Q6?usp=sharing)

### Pretrained backbone downloaded
    mkdir models
    cd models
Download ResNet-101 from pytorch official site [here](https://download.pytorch.org/models/resnet101-63fe2227.pth), rename to resnet101-torch.pth and move to models.

### Code preparation

- Replace the *graphgym* and *torch_geometric* in pyg with corresponding dir in our repository

### Train 

    python graphgym/train.py --cfg graphgym/configs/CUBI.yaml --seed 0
### Eval

    python graphgym/eval.py --cfg graphgym/configs/CUBI.yaml --eval train.epoch_resume 1 \
                              train.ckpt_prefix best val.extra_infos True seed 0

### Optional: Processing svg format dataset from CubiCasa-5k source data
**Notice: before running the code, please change the data dir within the code into your souce data dir**
- Source data downloaded: [here](https://zenodo.org/records/2613548)
- Download CubiCasa-5k source code and configure the environment: [here](https://github.com/CubiCasa/CubiCasa5k/tree/master)
- Put **_Replace_with_Cubicasa_** into CubiCasa-5k code repo
- Process source model.svg into roughcast svg format floorplans:


    python Replace_with_CubiCasa/roughcast_data_generation.py
  
- Render svg floorplan into image:


    python Replace_with_CubiCasa/ImgRasterization.py
  
- Render image annotation of floorplans:


    python Replace_with_CubiCasa/svg_loader.py
  
- Draw the wireframe of svg floorplan and turn the wireframe and image annotation into mmseg format:


    python DataPreparation/ImageProcessing_CubiCasa.py
  
- Process svg floorplan as pickle file:
    

    python SvgProcessing_CubiCasa.py
  

