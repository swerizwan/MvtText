# MvtText: AutoEncoder and Latent Diffusion for Text-Driven Body Language Synthesis

<img src="https://camo.githubusercontent.com/2722992d519a722218f896d5f5231d49f337aaff4514e78bd59ac935334e916a/68747470733a2f2f692e696d6775722e636f6d2f77617856496d762e706e67" alt="Oryx Video-ChatGPT" data-canonical-src="https://i.imgur.com/waxVImv.png" style="max-width: 100%;">

## Overview

This paper introduces MvtText, a method combining VAEs and latent diffusion models to generate realistic human movements from text. Using a multi-stage process, it progressively refines movements while aligning them with text inputs through dynamic multi-condition fusion. Experiments show MvtText outperforms existing methods, with applications in animation, VR, and human-computer interaction.

# 👁️💬 Architecture

The MvtText framework works as follows: (a) Movement data is modeled dynamically, and text descriptions are encoded using CLIP. (b) Encoder-decoder pairs generate low-dimensional pose representations. (c) A cascaded latent diffusion process refines these representations iteratively, starting coarse and adding details. (d) The result is realistic 3D human movement sequences aligned with the input text. 

<img style="max-width: 100%;" src="https://github.com/swerizwan/MvtText/blob/main/resources/overview.png" alt="VERHM Overview">

# Demo

```
python demo.py --cfg ./configs/config_motiontext_humanml3d.yaml --cfg_assets ./configs/assets.yaml --example demo.txt

/home/abbas/motiontext/blender/blender-2.83.0-linux64/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=/home/abbas/motiontext/outcomes/motiontext/HumanML3D/samples_2024-11-10-18-50-15/ --mode=video --joint_type=HumanML3D

python -m fit --dir /home/abbas/motiontext/outcomes/motiontext/HumanML3D/samples_2024-11-10-18-50-15/ --save_folder /home/abbas/motiontext/outcomes/motiontext/HumanML3D/samples_2024-11-10-18-50-15/tamp --cuda True

/home/abbas/motiontext/blender/blender-2.83.0-linux64/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=/home/abbas/motiontext/results/motiontext/1222_PELearn_Diff_Latent1_MEncDec49_MdiffEnc49_bs64_clip_uncond75_01/samples_2024-10-18-22-15-14/ --mode=video --joint_type=HumanML3D
```

<table>
  <tr>
    <td style="text-align: center;">
      <p>Happy</p>
      <img width="135" src="https://github.com/swerizwan/voiceemo/blob/main/resources/image4.gif" alt="Happy">
    </td>
    <td style="text-align: center;">
      <p>Frustrated</p>
      <img width="135" src="https://github.com/swerizwan/voiceemo/blob/main/resources/image1.gif" alt="Frustrated">
    </td>
    <td style="text-align: center;">
      <p>Sad</p>
      <img width="135" src="https://github.com/swerizwan/voiceemo/blob/main/resources/image2.gif" alt="Sad">
    </td>
    <td style="text-align: center;">
      <p>Angry</p>
      <img width="135" src="https://github.com/swerizwan/voiceemo/blob/main/resources/image3.gif" alt="Angry">
    </td>
    <td style="text-align: center;">
      <p>Surprise</p>
      <img width="135" src="https://github.com/swerizwan/voiceemo/blob/main/resources/image5.gif" alt="Surprise">
    </td>
  </tr>
</table>

## Installation

Follow these steps to set up the project:

1. Download the repository.
2. Create a conda virtual environment using: `conda create -n verhm python=3.7`.
3. Activate the environment: `conda activate verhm`.
4. Install all required dependencies mentioned in the Workflow section.

To begin, ensure you have the following essential libraries installed on your system:

- Pytorch 1.9.0
- CUDA 11.3
- Blender 3.4.1
- ffmpeg 4.4.1
- torch==1.9.0
- torchvision==0.10.0
- torchaudio==0.9.0
- numpy
- scipy==1.7.1
- librosa==0.8.1
- tqdm
- pickle
- transformers==4.6.1
- trimesh==3.9.27
- pyrender==0.1.45
- opencv-python

## Datasets

Our project utilizes several datasets for training and testing purposes:

- **VOCASET**: Offers audio-4D scan pairs for emotion analysis. [Dataset Link](https://voca.is.tue.mpg.de/download.php) `python main.py --dataset vocaset`
- **RAVDESS**: The RAVDESS comprises 7,356 files, totaling 24.8 GB in size. It features recordings from 24 professional actors, evenly split between genders. [Dataset Link](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) `python main.py --dataset ravdess`
- **MEAD**: The MEAD \cite{Mead} features 60 actors and actresses conversing with eight distinct emotions at varying intensity levels. [Dataset Link](https://wywu.github.io/projects/MEAD/MEAD.html/) `python main.py --dataset mead`

## Running the Demo

To run the demo, follow these steps:

1. Download Blender from [Blender Official Website](https://www.blender.org/download/), and place it in the Blender folder within the root directory.
2. Download the pre-trained model from [Pre-Trained Model Link](https://drive.google.com/file/d/1ywEYhMWdxWk9Bqt0UIOdAyYM6v8JUF-K/view?usp=sharing) and put it in the `pre-trained` folder in the root directory.
3. Run the demo by executing `run_demo.py` with the desired input voice. 
