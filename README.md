# [Model] Prime

<!-- Insert the project banner here -->
<div align="center">
    <a href="https://github.com/ai4protein/Prime/"><img width="200px" height="auto" src="https://github.com/ai4protein/Prime/blob/main/band.png"></a>
</div>

<!-- Select some of the point info, feel free to delete -->
[![GitHub license](https://img.shields.io/github/license/ai4protein/Prime)](https://github.com/ai4protein/Prime/blob/main/LICENSE)

Updated on 2023.10.20



## Key Feature

This repository provides the official implementation of Prime (Protein language model for Intelligent Masked pretraining and Environment (temperature) prediction).

Key feature:
- Zero-shot mutant effect prediction.

## Links

- [Paper](https://arxiv.org/abs/2304.03780)
- [Code](https://github.com/ai4protein/Prime) 

## Details

### What is Prime?
Prime, a novel protein language model, has been developed for predicting the Optimal Growth Temperature (OGT) and enabling zero-shot prediction of protein thermostability and activity. This novel approach leverages temperature-guided language modeling.
<div align="center">
    <a href="https://"><img width="1000px" height="auto" src="https://github.com/ai4protein/Prime/blob/main/model.jpg"></a>
</div>


## Use of PRIME

**Main Requirements**  
biopython==1.81
torch==2.0.1

**Installation**
```bash
pip install -r requirements.txt
```

**Download Model**

[prime-base](https://drive.google.com/file/d/1sjl-0JNBr5EH5PXy6dbkcZaO50zYklGe/view)

[prime-fine-tuning-for-tm-datasets](https://drive.google.com/file/d/1jo3OMJSCNuB_To2gNjOSCqNVjmqo2dZI/view?usp=drive_link)


**Predicting OGT**
```bash
python predict_ogt.py --model_name prime-base \
--fasta ./datasets/OGT/ogt_small.fasta \
--output ogt_prediction.tsv
```


**Predicting Mutant Effect**

Using the prime-base model. (Recommended)
```shell
python predict_mutant.py --model_name prime-base \
--fasta ./datasets/TM/1CF1/1CF1.fasta \
--mutant ./datasets/TM/1CF1/1CF1-7.0.tsv \
--compute_spearman \
--output pred.tsv
```

Or using the model that fine-tuned on the homologous sequence of the proteins in the TM dataset.
```shell
python predict_mutant.py --model_name prime-tm-fine-tuning \
--fasta ./datasets/TM/1CF1/1CF1.fasta \
--mutant ./datasets/TM/1CF1/1CF1-7.0.tsv \
--compute_spearman \
--output pred.tsv
```


## 🙋‍♀️ Feedback and Contact

- [Send Email](mailto:ginnmelich@gmail.com)

## 🛡️ License

This project is under the MIT license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgement

A lot of code is modified from [🤗 transformers](https://github.com/huggingface/transformers) and [esm](https://github.com/facebookresearch/esm).

## 📝 Citation

If you find this repository useful, please consider citing this paper:
```
@misc{tan2023,
      title={Engineering Enhanced Stability and Activity in Proteins through a Novel Temperature-Guided Language Modeling.}, 
      author={Pan Tan and Mingchen Li and Liang Zhang and Zhiqiang Hu and Liang Hong},
      year={2023},
      eprint={2304.03780},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
```