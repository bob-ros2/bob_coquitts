# Bob's Coqui TTS ROS Node

This ROS package provides a robust node that interfaces with the [Coqui TTS](https://github.com/coqui-ai/TTS) library, allowing a ROS 2 system to convert text into speech. It intelligently processes incoming text streams, handles voice cloning with models like XTTS, and offers extensive configuration options.

## Features

-   **Intelligent Text Processing**: Buffers incoming text fragments and splits them into complete sentences to ensure high-quality, natural-sounding speech generation and prevent model "hallucinations".
-   **Advanced Text Filtering**: Automatically filters unwanted characters (e.g., typographical quotes) and trims trailing characters (e.g., colons) from sentences before synthesis.
-   **Real-time Feedback**: Publishes the exact text chunk being synthesized to a separate ROS topic, allowing other nodes to synchronize with the speech output.
-   **Wide Model Support**: Converts text from a ROS topic into audible speech using a wide range of pre-trained Coqui TTS models.
-   **Zero-Shot Voice Cloning**: Enables voice cloning using XTTS models and a reference WAV file.
-   **Flexible Output**: Optionally plays audio directly or saves it to a WAV file with automatic unique filename generation.
-   **Hardware Acceleration**: Supports both GPU (`cuda`) and CPU inference.
-   **Highly Configurable**: All key settings are exposed as ROS parameters for easy tuning.

## Prerequisites

-   ROS 2 (Humble, Iron, or newer).
-   Python 3.8+
-   NVIDIA GPU with CUDA installed for GPU acceleration (optional but recommended for XTTS).
-   An audio output device.
-   System dependencies for `sounddevice` and `libsndfile`.

```bash
# For Debian/Ubuntu-based systems
sudo apt-get update
sudo apt-get install libportaudio2 libasound-dev libsndfile1
```

## Installation

1.  **Clone the Package**:
    Clone this repository into your ROS 2 workspace's `src` directory.

    ```bash
    cd ~/ros2_ws/src
    git clone https://github.com/bob-ros2/bob_coquitts
    ```

2.  **Install Python Dependencies**:
    It is recommended to use a Python virtual environment to avoid package conflicts.

    ```bash
    cd ~/ros2_ws
    # If using a virtual environment, activate it first
    pip install -r src/bob_coquitts/requirements.txt
    ```
    or install them manually
    
    ```bash
    pip install TTS sounddevice numpy soundfile
    ```
    The `TTS` library will install numerous dependencies, including `torch`.

## Building

Source your ROS 2 installation and build the package using `colcon`.

```bash
# Navigate to the root of your workspace
cd ~/ros2_ws

# Source ROS 2 (adjust for your distribution)
source /opt/ros/humble/setup.bash

# Build the package
colcon build --packages-select bob_coquitts
```

## Usage

After building, source the workspace's `setup.bash` file. For detailed troubleshooting, launch the node with `--log-level DEBUG` to see all diagnostic messages.

```bash
# Source the local workspace
source ~/ros2_ws/install/setup.bash

# Launch the node
ros2 run bob_coquitts tts
```

### Example 1: Standard Single-Speaker Model (Default)

This uses the `tts_models/en/ljspeech/vits` model, a single female English voice.

```bash
ros2 run bob_coquitts tts --ros-args -p device:='cpu'
```

### Example 2: XTTS Voice Cloning with Text Cleaning

This example uses the powerful XTTS v2 model for voice cloning, which requires a `reference_wav`. We also override the `sentence_end_trim_chars` parameter to remove trailing colons, which can cause unnatural-sounding audio.

```bash
ros2 run bob_coquitts tts --ros-args \
-p model_name:='tts_models/multilingual/multi-dataset/xtts_v2' \
-p reference_wav:='/path/to/your/voice.wav' \
-p language:='en' \
-p device:='cuda' \
-p sentence_end_trim_chars:="':'"

# In another terminal, publish text with a colon
ros2 topic pub --once /text std_msgs/msg/String "data: 'Here is my statement:'"

# In a third terminal, listen to the cleaned text being spoken
ros2 topic echo /text_speaking
# Output will be: data: Here is my statement
```

### Example 3: German Single-Speaker Model

This example uses a VITS model. Since it's a single-language model, the `language` and `reference_wav` parameters are not needed.

```bash
ros2 run bob_coquitts tts --ros-args \
-p model_name:='tts_models/de/css10/vits-neon' \
-p device:='cpu'

# In another terminal
ros2 topic pub --once /text std_msgs/msg/String "data: 'Die Integration von kreativen Modellen ist sehr sinnvoll.'"
```

### Example 4: Saving Audio to a File

This disables direct playback and saves the generated speech to `tts_output.wav`. If the file exists, it will be saved as `tts_output_001.wav`.

```bash
ros2 run bob_coquitts tts --ros-args \
-p play_audio:=False \
-p output_wav_path:='tts_output.wav'

# In another terminal
ros2 topic pub --once /text std_msgs/msg/String "data: 'This speech will be saved to a file.'"
```

## ROS Interface

### Subscribed Topics

| Topic Name | Message Type           | Description                                    |
|------------|------------------------|------------------------------------------------|
| `/text`    | `std_msgs/msg/String`  | The text to be synthesized into speech. Text is buffered and processed in complete sentences. |

### Published Topics

| Topic Name        | Message Type           | Description                                    |
|-------------------|------------------------|------------------------------------------------|
| `/text_speaking`  | `std_msgs/msg/String`  | Publishes the sentence or chunk of text exactly as it is being sent to the TTS model for synthesis. |

### Parameters

All parameters can be set at runtime via the command line.

| Parameter Name            | Type    | Default Value                 | Description                                                                                                                              |
|---------------------------|---------|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| **General**               |         |                               |                                                                                                                                          |
| `model_name`              | string  | `tts_models/en/ljspeech/vits` | The Coqui TTS model to use. Run `tts --list_models` for options. (env: `COQUITTS_MODEL_NAME`)                                              |
| `language`                | string  | `''` (empty)                  | Language code for multi-lingual models (e.g., `en`, `de`). (env: `COQUITTS_LANGUAGE`)                                                      |
| `device`                  | string  | `cpu`                         | Compute device for TTS inference, e.g., `cuda` or `cpu`. (env: `COQUITTS_DEVICE`)                                                          |
| `reference_wav`           | string  | `''` (empty)                  | Path to a reference WAV file for voice cloning with `xtts` or `your_tts`. (env: `COQUITTS_REFERENCE_WAV`)                                  |
| **Audio Output**          |         |                               |                                                                                                                                          |
| `sample_rate`             | integer | `24000`                       | Audio sample rate for playback. Must match the model's native rate. (env: `COQUITTS_SAMPLE_RATE`)                                        |
| `play_audio`              | boolean | `True`                        | If true, plays the generated audio directly. (env: `COQUITTS_PLAY_AUDIO`)                                                                |
| `output_wav_path`         | string  | `''` (empty)                  | Path to save the output WAV file. If empty and `play_audio` is false, `tts_output.wav` is used. (env: `COQUITTS_OUTPUT_WAV_PATH`)         |
| **Text Processing**       |         |                               |                                                                                                                                          |
| `split_sentences`         | boolean | `False`                       | Enable Coqui's internal sentence splitter. Recommended to be `True` if sending very long, unstructured text blocks. (env: `COQUITTS_SPLIT_SENTENCES`) |
| `sentences_max`           | integer | `2`                           | Max number of sentences from the internal buffer to process at once. (env: `COQUITTS_SENTENCES_MAX`)                                     |
| `sentence_delimiters`     | string  | `.!?\n`                       | Characters used to split the buffered text into sentences. (env: `COQUITTS_SENTENCE_DELIMITERS`)                                         |
| `sentence_end_trim_chars` | string  | `''` (empty)                  | A string of characters to remove from the end of each sentence before synthesis (e.g., `:`). (env: `COQUITTS_SENTENCE_END_TRIM_CHARS`) |
| `text_filter_chars`       | string  | `„”‚‘“”’`                      | Characters to completely remove from the input text before any processing. (env: `COQUITTS_TEXT_FILTER_CHARS`)                            |
| **XTTS Tuning**           |         |                               |                                                                                                                                          |
| `temperature`             | double  | `0.2`                         | Controls randomness in generation. Lower values are more deterministic. (env: `COQUITTS_TEMPERATURE`)                                      |
| `length_penalty`          | double  | `1.0`                         | A factor to penalize longer sequences. (env: `COQUITTS_LENGTH_PENALTY`)                                                                  |
| `repetition_penalty`      | double  | `2.0`                         | Penalty for repeating tokens. Higher values reduce repetition. (env: `COQUITTS_REPETITION_PENALTY`)                                      |
| `top_k`                   | integer | `40`                          | Samples from the k most likely next tokens. (env: `COQUITTS_TOP_K`)                                                                        |
| `top_p`                   | double  | `0.9`                         | Samples from tokens with a cumulative probability of p. (env: `COQUITTS_TOP_P`)                                                            |
