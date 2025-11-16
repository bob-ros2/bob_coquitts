# Bob's Coqui TTS ROS Node

This ROS package provides a robust node that interfaces with the [Coqui TTS](https://github.com/coqui-ai/TTS) library, allowing a ROS 2 system to convert text into speech. It intelligently processes incoming text streams, handles voice cloning with models like XTTS, and offers extensive configuration options.

## Features

-   **Intelligent Text Buffering**: Waits for pauses in the incoming text stream before processing to ensure complete thoughts are synthesized.
-   **Dual Splitting Modes**: Choose between two modes for sentence splitting via the `split_sentences` parameter:
    1.  **Manual Mode (Default)**: Use custom delimiters (`sentence_delimiters`) to precisely control how text is split into sentences.
    2.  **Automatic Mode**: Let Coqui's powerful internal splitter handle long, unstructured text blocks.
-   **Advanced Text Normalization**:
    -   Automatically filters pictorial emojis and symbols by default using a Unicode-aware regex filter.
    -   Removes user-defined characters (e.g., typographical quotes `„“`).
    -   Strips leading and trailing characters (e.g., spaces, punctuation) from sentences before synthesis.
    -   Normalizes numbers by removing thousands separators (e.g., `2.500` -> `2500`) to ensure correct pronunciation.
-   **Real-time Feedback**: Publishes the exact text chunk being synthesized to a separate ROS topic (`/text_speaking`), allowing other nodes to synchronize with the speech output.
-   **Wide Model Support & Voice Cloning**: Supports a vast range of Coqui models, including zero-shot voice cloning with XTTS.
-   **Flexible Output**: Optionally plays audio directly or saves it to a WAV file with automatic unique filename generation.
-   **Hardware Acceleration**: Supports both GPU (`cuda`) and CPU inference.

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

2.  **Install Python Dependencies**:
    This node requires the `regex` library for full Unicode support (e.g., filtering emojis). It is recommended to use a Python virtual environment.

    ```bash
    cd ~/ros2_ws
    # If using a virtual environment, activate it first
    pip install -r src/bob_coquitts/requirements.txt
    ```
    The `requirements.txt` file should contain:
    ```
    TTS
    sounddevice
    numpy
    soundfile
    regex
    ```

## Building

Source your ROS 2 installation and build the package using `colcon`.

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select bob_coquitts
```

## Usage

After building, source the workspace's `setup.bash` file. For detailed troubleshooting, launch the node with `--log-level DEBUG`.

```bash
source ~/ros2_ws/install/setup.bash
ros2 run bob_coquitts tts
```

### Example 1: XTTS Voice Cloning with Text Cleaning

This example uses the powerful XTTS v2 model. We override the `sentence_strip_chars` parameter to only remove colons, which can cause unnatural-sounding audio.

```bash
ros2 run bob_coquitts tts --ros-args \
-p model_name:='tts_models/multilingual/multi-dataset/xtts_v2' \
-p reference_wav:='/path/to/your/voice.wav' \
-p language:='en' \
-p device:='cuda' \
-p sentence_strip_chars:="':'"

# In another terminal, publish text with a colon
ros2 topic pub --once /text std_msgs/msg/String "data: 'Here is my statement:'"

# In a third terminal, listen to the cleaned text being spoken
ros2 topic echo /text_speaking
# Output will be: data: Here is my statement
```

### Example 2: Using Coqui's Internal Splitter for Long Text

If you are feeding a large, unstructured block of text, it's best to let Coqui handle the splitting.

```bash
ros2 run bob_coquitts tts --ros-args -p split_sentences:=True

# Publish a long paragraph
ros2 topic pub --once /text std_msgs/msg/String "data: 'This is the first sentence. This is the second sentence which is much longer and might exceed the character limit if not handled properly. Coquis splitter will take care of it.'"
```

## ROS Interface

### Subscribed Topics

| Topic Name | Message Type           | Description                                    |
|------------|------------------------|------------------------------------------------|
| `/text`    | `std_msgs/msg/String`  | The text to be synthesized. The node buffers incoming text and processes it after a pause. |

### Published Topics

| Topic Name        | Message Type           | Description                                    |
|-------------------|------------------------|------------------------------------------------|
| `/text_speaking`  | `std_msgs/msg/String`  | Publishes the cleaned, normalized sentence or chunk of text exactly as it is being sent to the TTS model. |

### Parameters

| Parameter Name            | Type    | Default Value                                | Description                                                                                                                                          |
|---------------------------|---------|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| **General**               |         |                                              |                                                                                                                                                      |
| `model_name`              | string  | `tts_models/en/ljspeech/vits`                | The Coqui TTS model to use. (env: `COQUITTS_MODEL_NAME`)                                                                                             |
| `language`                | string  | `''`                                         | Language code for multi-lingual models (e.g., `en`, `de`). (env: `COQUITTS_LANGUAGE`)                                                                |
| `device`                  | string  | `cpu`                                        | Compute device for inference (`cuda` or `cpu`). (env: `COQUITTS_DEVICE`)                                                                             |
| `reference_wav`           | string  | `''`                                         | Path to a reference WAV file for voice cloning. (env: `COQUITTS_REFERENCE_WAV`)                                                                      |
| **Audio Output**          |         |                                              |                                                                                                                                                      |
| `sample_rate`             | integer | `24000`                                      | Audio sample rate for playback. Must match the model's native rate. (env: `COQUITTS_SAMPLE_RATE`)                                                    |
| `play_audio`              | boolean | `True`                                       | If true, plays the generated audio directly. (env: `COQUITTS_PLAY_AUDIO`)                                                                            |
| `output_wav_path`         | string  | `''`                                         | Path to save the output WAV file. (env: `COQUITTS_OUTPUT_WAV_PATH`)                                                                                  |
| **Text Processing**       |         |                                              |                                                                                                                                                      |
| `split_sentences`         | boolean | `False`                                      | **Mode switch for splitting.** If `True`, Coqui handles splitting. If `False` (default), the node uses manual splitting below. (env: `COQUITTS_SPLIT_SENTENCES`) |
| `sentence_delimiters`     | string  | `.!?\n`                                      | Characters used for manual splitting (only when `split_sentences` is `False`). (env: `COQUITTS_SENTENCE_DELIMITERS`)                             |
| `sentences_max`           | integer | `1`                                          | Max number of sentences to process at once in manual mode. (env: `COQUITTS_SENTENCES_MAX`)                                                          |
| `number_thousands_separator` | string | `.`                                       | Character to remove from between digits (e.g., `.` in `1.234`). (env: `COQUITTS_NUMBER_THOUSANDS_SEPARATOR`)                                     |
| `sentence_strip_chars`    | string  | `.,:!? `                                     | Characters to remove from the beginning and end of a processed text chunk. (env: `COQUITTS_SENTENCE_STRIP_CHARS`)                                  |
| `text_filter_chars`       | string  | `„”‘“’*—#<>`                                  | Specific characters to remove from the entire text. (env: `COQUITTS_TEXT_FILTER_CHARS`)                                                            |
| `text_filter_regex`       | string  | `[\p{Emoji_Presentation}\p{Extended_Pictographic}]` | Regex to remove patterns from the entire text. **Requires `regex` pip package.** Default filters emojis. (env: `COQUITTS_TEXT_FILTER_REGEX`)     |
| **XTTS Tuning**           |         |                                              |                                                                                                                                                      |
| `temperature`             | double  | `0.2`                                        | Controls randomness. Lower is more deterministic. (env: `COQUITTS_TEMPERATURE`)                                                                    |
| `length_penalty`          | double  | `1.0`                                        | Factor to penalize longer sequences. (env: `COQUITTS_LENGTH_PENALTY`)                                                                              |
| `repetition_penalty`      | double  | `2.0`                                        | Penalty for repeating tokens. (env: `COQUITTS_REPETITION_PENALTY`)                                                                                 |
| `top_k`                   | integer | `40`                                         | Samples from the k most likely next tokens. (env: `COQUITTS_TOP_K`)                                                                                |
| `top_p`                   | double  | `0.9`                                        | Samples from tokens with a cumulative probability of p. (env: `COQUITTS_TOP_P`)                                                                    |