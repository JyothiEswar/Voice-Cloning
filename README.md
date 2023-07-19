# Tortoise TTS

Tortoise TTS is a text-to-speech (TTS) system that utilizes deep learning techniques to generate speech from text input. This project provides a Python API for generating high-quality synthetic speech using custom voice samples.

## Installation

To install the required dependencies, run the following command:

```shell
!pip3 install -U scipy
!git clone https://github.com/jnordberg/tortoise-tts.git
%cd tortoise-tts
!pip3 install -r requirements.txt
!pip3 install transformers==4.19.0 einops==0.5.0 rotary_embedding_torch==0.1.5 unidecode==1.3.5
!python3 setup.py install
```

Make sure you have `scipy` installed and clone the Tortoise TTS repository. Then, navigate to the `tortoise-tts` directory and install the required dependencies using `pip3`. Finally, install the additional packages specified in the `setup.py` file.

## Usage

To generate speech using Tortoise TTS, follow these steps:

1. Import the necessary libraries:

```python
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import IPython
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices
```

2. Create an instance of the `TextToSpeech` class:

```python
tts = TextToSpeech()
```

3. Set the desired text to be spoken:

```python
text = "Hello everyone, This candidate is well suited for this role, he possess a combination of skills that best fit for this role."
```

4. Choose a preset mode to determine the quality of the generated speech:

```python
preset = "high_quality"  # Options: "ultra_fast", "fast", "standard", "high_quality"
```

5. Upload at least two audio clips (WAV files, 6-10 seconds long) to create a custom voice:

```python
CUSTOM_VOICE_NAME = "custom"
custom_voice_folder = f"tortoise/voices/{CUSTOM_VOICE_NAME}"
# Check if the directory already exists
if not os.path.exists(custom_voice_folder):
    os.makedirs(custom_voice_folder)
for i, file_data in enumerate(files.upload().values()):
    with open(os.path.join(custom_voice_folder, f'{i}.wav'), 'wb') as f:
        f.write(file_data)
```

6. Generate speech using the custom voice samples:

```python
voice_samples, conditioning_latents = load_voice(CUSTOM_VOICE_NAME)
gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset=preset)
torchaudio.save(f'generated-{CUSTOM_VOICE_NAME}.wav', gen.squeeze(0).cpu(), 24000)
IPython.display.Audio(f'generated-{CUSTOM_VOICE_NAME}.wav')
```

7. Run the code and enjoy the generated speech!

