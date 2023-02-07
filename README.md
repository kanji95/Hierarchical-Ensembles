# Test-Time Amendment with a Coarse Classifier for Fine-Grained Classification

Official PyTorch implementation | [Paper](https://arxiv.org/abs/2302.00368)

## Abstract
> We investigate the problem of reducing mistake severity for fine-grained classification. Fine-grained classification can be challenging, mainly due to the requirement of knowledge or domain expertise for accurate annotation. However, humans are particularly adept at performing coarse classification as it requires relatively low levels of expertise. To this end, we present a novel approach for Post-Hoc Correction called Hierarchical Ensembles (HiE) that utilizes label hierarchy to improve the performance of fine-grained classification at test-time using the coarse-grained predictions. By only requiring the parents of leaf nodes, our method significantly reduces avg. mistake severity while improving top-1 accuracy on the iNaturalist-19 and tieredImageNet-H datasets, achieving a new state-of-the-art on both benchmarks. We also investigate the efficacy of our approach in the semi-supervised setting. Our approach brings notable gains in top-1 accuracy while significantly decreasing the severity of mistakes as training data decreases for the fine-grained classes. The simplicity and post-hoc nature of HiE render it practical to be used with any off-the-shelf trained model to improve its predictions further.

<img width="940" alt="motivation" src="https://user-images.githubusercontent.com/30688360/217342472-b3a5262f-74e3-4406-9aee-151188ff717e.png">

## Dataset

Prepare iNaturalist-19 and tieredImageNet-H datasets using this [repo](https://github.com/fiveai/making-better-mistakes)

## Training

* For supervised learning setting, refer to this [fork](https://github.com/kanji95/HAF) of this [repo](https://github.com/07Agarg/HAF) 
* For semi-supervised learning setting, refer to this [fork](https://github.com/kanji95/ssl-evaluation) of this [repo](https://github.com/cvl-umass/ssl-evaluation)

## Evaluation

    python evaluate.py --dataset <dataset> --crm 0 --post_hoc 1 --model_path <model_path> --aux_path <aux_path>

# Acknowledgements

Training codes are borrowed from https://github.com/07Agarg/HAF, https://github.com/cvl-umass/ssl-evaluation.
Evaluation code is borrowed from https://github.com/sgk98/CRM-Better-Mistakes.

## Citation

If you found our work useful to your research, please consider citing:

    @misc{https://doi.org/10.48550/arxiv.2302.00368,
      doi = {10.48550/ARXIV.2302.00368},
      url = {https://arxiv.org/abs/2302.00368},
      author = {Jain, Kanishk and Karthik, Shyamgopal and Gandhi, Vineet},
      keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
      title = {Test-Time Amendment with a Coarse Classifier for Fine-Grained Classification},
      publisher = {arXiv},
      year = {2023},
      copyright = {Creative Commons Attribution 4.0 International}
    }

