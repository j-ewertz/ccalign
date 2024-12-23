# ccalign
ccalign is designed to simplify sentence-level text-audio alignment for analyzing conference calls.
Further details can be found in the original research [paper](https://ssrn.com/abstract=4307178).

Please note that this repository is at an early stage. Further updates and improvements will follow shortly.

# Setup and Installation
This repository has currently only been tested on Linux (*Ubuntu 22.04 jammy*).
- GPU transcription execution requires the NVIDIA libraries cuBLAS 11.x and cuDNN 8.x to be installed on the system.
- You also need to install FFmpeg on your system. You can follow [openAI instructions](https://github.com/openai/whisper#setup).
- Make sure to create and activate a virtual environment for a non-global installation:

    `python3 -m venv venv_ccalign`

    `source venv_ccalign/bin/activate`

### 1. Install torch and CUDA11.8
You can install PyTorch and CUDA using pip:

`pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118`

You can check the PyTorch [homepage](https://pytorch.org/get-started/locally/) for more setups.

### 2. Install whisperx
The repository depends on [whisperx](https://github.com/m-bain/whisperX). You can install the latest version using pip:

`pip install git+https://github.com/m-bain/whisperx.git`

### 3. Install ccalign
You can install ccalign using pip:

`pip install ccalign`


# Preprocessing, Input and Output
## Needed Input
The alignment requires specific preprocessing of the transcript. A pandas DataFrame or a Dataset with the following variables must be provided: 
- A unique ID. Feature name: 'id'
- A path to the conference call audio file. Feature name: 'path_audio'
- A path to the preprocessed conference call transcript. Feature name: 'path_transcript'
### Preprocessed conference call transcript
The conference call transcript must be preprocessed at paragraph level (speaker changes) and contain the original text, the speaker and information about the call section. The call section can be any of the following strings:
- "-OP-": Operator instructions
- "-PR-": Presentation of a company executive
- "-Q-": An analyst question
- "-Q_A-": Question answer of a company executive

The preprocessed information must be provided in the following JSON format:
### Example: Earnings Conference Call JSON Format
```
{
  "paragraphs": [
    {
      "text":
        "Welcome to the ABC Corporation fourth quarter earnings conference call. My name is Jane, and I will be the operator for today's call.
        Thank you for joining us.",
      "speaker": "Operator",
      "call_section": "-OP-"
    },
    {
      "text":
        "Thank you, Jane. Good morning, everyone, and thank you for joining our earnings call.
        Today, we will discuss our results for the fourth quarter and provide an outlook for the upcoming year.",
      "speaker": "John Doe",
      "call_section": "-PR-"
    },
    {
      ...
    }
  ]
}
```
## Output
As a first step, the `execute_whisperx()` function transcribes the audio file with word-level timestamps using [whisperx](https://github.com/m-bain/whisperX). The transcription files (whisper and whisperx) are saved as a JSON file in the same folder as the audio file. If you already have the whisperx timestamps, you can simply define the path in the DataFrame or Dataset as "path_whisperx" and skip this step. 

Second, the `execute_alignment()` function aligns the Whisperx output with the original transcript.  
Sentence-level timestamps, alignment statistics and word-level timestamps are exported to a directory called *ccalign_results* in the current working directory.
Word-level timestamps are experimental at this time.

## Configurations
The following is an explanation of the configuration parameters used in the project:
- model: Specifies the transcription model based on Whisper, which converts audio input into text.
- device: Defines the hardware (e.g., CPU or GPU) used for transcription.
- batch_size: Determines the number of audio data batches processed simultaneously within a single run of the Whisper model.
This impacts processing speed and memory usage.
- compute_type: Specifies the data type (e.g., fp16 or int8) used for computations in the Whisper model, affecting performance and accuracy.
- num_processes_whisperx: Sets the number of parallel transcription processes used for processing. The number should be adjusted based on available GPU memory.
- num_processes_alignment: Defines the number of parallel alignment processes used. These processes run exclusively on the CPU.
- calls_per_core: Specifies how many conference calls are processed by a single process.
A backup of the alignment process is created after processing a number of calls equal to num_processes_alignment * calls_per_core.


## Python usage
Command line execution is not available at this time. For licensing reasons, we are unable to provide conference calls to test the code.

```python
from ccalign import execute_whisperx, execute_alignment 
import pandas as pd
import torch
import multiprocessing as mp

df = pd.read_pickle(r'path/to/dataframe.pkl')

device, dtype = ("cuda", "float16") if torch.cuda.is_available() else ("cpu", "float32")
cpu_cores = mp.cpu_count()

df = execute_whisperx(
    path_data=df,
    model="base.en",
    batch_size_whisper=16,
    num_processes_whisperx=2,
    device=device,
    dtype=dtype
    )
        

df = execute_alignment(
    path_data=df,
    num_processes=cpu_cores - 2,
    calls_per_core=10
    )

```

## License
- This repository depends on [whisperx](https://github.com/m-bain/whisperX) (BSD 2-Clause License).
- This repository is licensed under the MIT license.

## Contact/Support
We look forward to any suggestions for improvement or support.
Feel free to get in [touch](https://www.linkedin.com/in/jonas-ewertz-780b5125a/)!

## Citation
```
@article{ewertz2024,
  title={Listen Closely: Measuring Vocal Tone in Corporate Disclosures},
  author={Ewertz, Jonas and Knickrehm, Charlotte and Nienhaus, Martin and Reichmann, Doron},
  year={2024},
  note={Available at SSRN: \url{https://ssrn.com/abstract=4307178}}
  }