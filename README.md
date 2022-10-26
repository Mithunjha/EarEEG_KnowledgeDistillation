# A KNOWLEDGE DISTILLATION FRAMEWORK FOR ENHANCING EAR-EEG BASED SLEEP STAGING WITH SCALP-EEG DATA

This repository contains implementation of cross-modal knowledge distillation approach combining response based distillation to enhance ear-EEG based sleep staging using the scalp-EEG.

![cross modal distillation.png](cross_modal_distillation.png)

## Getting Started
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mithunjha/EarEEG_KnowledgeDistillation/blob/main/EarEEG_KD_github.ipynb)

### Installation

The algorithms were developed in Pytorch Environment : [https://pytorch.org/](https://pytorch.org/)

```python
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Run below code to install all other dependencies

```python
pip install -r requirements.txt
```

## Training

### Training Phase 01 - Supervised Training

Initially models are trained in supervised manner on scalp-EEG recordings, this pretrained models are used as teacher models in knowledge distillation training.

```python
python main.py --model USleep --data_path <data path> --train_data_list [0,1,2,3,5,6,7] --val_data_list [8] --signals ear-eeg --training_type supervised --is_neptune True --nep_project <neptune project name> --nep_api <neptune API>
```

### Training Phase 02 - Knowledge Distillation

**Offline Knowledge Distillation**

In offline knowledge distillation, the pretrained supervised model is used as teacher model and student model is trained to learn the distribution of teacher model as well as the true targets.

```python
python main.py --data_path <data path> --model_path <model path>
```

**Online Knowledge Distillation**

In online knowledge distillation, both teacher and student models are trained simultaneously. 

```python
python main.py --data_path <data path> --KD_type online
```

**Transfer Learning**

```python
python main.py --data_path <data path> --is_retrain True --training_type supervised --model_path <model path>
```
### Evaluation

Evaluation code script can be used to evaluate the saved models.

```python
python evaluation.py --data_path <data path> --model_path <model path> --n_epochs 20
```
### Reference
[1] Perslev M, Darkner S, Kempfner L, Nikolic M, Jennum PJ, Igel C.U-Sleep: resilient high-frequency sleep staging. npj Digit. Med. 4, 72 (2021).https://github.com/perslev/U-Time/blob/master/utime/models/usleep.py

[2] Pradeepkumar, J., Anandakumar, M., Kugathasan, V., Suntharalingham, D., Kappel, S. L., De Silva, A. C., & Edussooriya, C. U. Towards Interpretable Sleep Stage Classification Using Cross-Modal Transformers. arXiv preprint arXiv:2208.06991 (2022). https://github.com/Jathurshan0330/Cross-Modal-Transformer
