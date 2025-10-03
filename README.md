# emg-codec

Fork of [facebookresearch/emg2qwerty](https://github.com/facebookresearch/emg2qwerty) that incorporates an end-to-end trainable Residual Vector Quantizer (RVQ) for tokenizing/compressing EMG. Key additions are implemented in [vector_quantizer.py](emg2qwerty/vector_quantizer.py).

RVQ was popularized by [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312), Zeghidour et al. (2021) in the context of speech/audio, and has emerged as a prominent method to discretize continuous latent representations into tokens for multimodal LLM training. This enables EMG signals to be represented in a form compatible with token-based models, opening the door to leveraging LLM architectures for neural interface and biosignal applications.

`emg-codec` demonstrates that EMG signals can be tokenized with very high compression rates without hurting task performance. On the [emg2qwerty benchmark](https://arxiv.org/abs/2410.20081), we achieve 300x compression rate (1500 Kbps → 5 Kbps) while maintaining character error rate (CER).

## Training

Follow the [setup instructions](https://github.com/facebookresearch/emg2qwerty#setup) from the original emg2qwerty repository.

Baseline (without RVQ) training of personalized models:
```
python -m emg2qwerty.train \
  user="glob(user*)" \
  trainer.accelerator=gpu trainer.devices=1 \
  --multirun
```

Residual VQ with 4 codebook levels of 1024 entries each (40 bits per timestep):
```
python -m emg2qwerty.train \
  user="glob(user*)" \
  trainer.accelerator=gpu trainer.devices=1 \
  vector_quantizer=residual_vq \
  vector_quantizer.n_vq=4 \
  vector_quantizer.codebook_size=1024 \
  --multirun
```
