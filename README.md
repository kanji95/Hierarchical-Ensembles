# Test-Time Amendment with a Coarse Classifier for Fine-Grained Classification

Official PyTorch implementation | [Paper](https://arxiv.org/abs/2302.00368)

## Abstract
> We investigate the problem of reducing mistake severity for fine-grained classification. Fine-grained classification can be challenging, mainly due to the requirement of knowledge or domain expertise for accurate annotation. However, humans are particularly adept at performing coarse classification as it requires relatively low levels of expertise. To this end, we present a novel approach for Post-Hoc Correction called Hierarchical Ensembles (HiE) that utilizes label hierarchy to improve the performance of fine-grained classification at test-time using the coarse-grained predictions. By only requiring the parents of leaf nodes, our method significantly reduces avg. mistake severity while improving top-1 accuracy on the iNaturalist-19 and tieredImageNet-H datasets, achieving a new state-of-the-art on both benchmarks. We also investigate the efficacy of our approach in the semi-supervised setting. Our approach brings notable gains in top-1 accuracy while significantly decreasing the severity of mistakes as training data decreases for the fine-grained classes. The simplicity and post-hoc nature of HiE render it practical to be used with any off-the-shelf trained model to improve its predictions further.

![](https://user-images.githubusercontent.com/30688360/166107570-5c941733-ba6c-4864-94f5-b41b3c5b1566.jpg)


## Dataset

Prepate iNaturalist-19 and tieredImageNet-H datasets from this [repo](https://github.com/fiveai/making-better-mistakes)

## Training

* Python: 3.8.5
* Pytorch: 1.7.1
* Torchvision: 0.8.2
* pip install git+https://github.com/andfoy/refer.git
* Wandb: 0.12.9 (for visualization)

## Training

    python3 -W ignore main.py --batch_size 32 --num_workers 4 --optimizer AdamW --dataroot <REFERIT_DATA> --lr 1.2e-4 --weight_decay 9e-5 --image_encoder deeplabv3_plus --loss bce --dropout 0.2 --epochs 30 --gamma 0.7 --num_encoder_layers 2 --image_dim 320 --mask_dim 160 --phrase_len 30 --glove_path <GLOVE_PATH> --threshold 0.40 --task referit --feature_dim 20 --transformer_dim 512 --run_name <WANDB_RUN_NAME> --channel_dim 512 --attn_type normal --save


## Citation

If you found our work useful to your research, please consider citing:

    @article{jain2021comprehensive,
      title={Comprehensive Multi-Modal Interactions for Referring Image Segmentation},
      author={Jain, Kanishk and Gandhi, Vineet},
      journal={arXiv preprint arXiv:2104.10412},
      year={2021}
    }
