# Bob's Coqui TTS ROS Node

This ROS package provides a node that interfaces with the [Coqui TTS](https://github.com/coqui-ai/TTS) library, allowing a ROS 2 system to convert text into speech. It can use standard single-speaker models or advanced multi-speaker, zero-shot voice cloning models like XTTS.

## Features

-   Converts text from a ROS topic into audible speech.
-   Supports a wide range of pre-trained Coqui TTS models.
-   Enables zero-shot voice cloning using XTTS models and a reference WAV file.
-   GPU (`cuda`) or CPU acceleration.
-   All key settings are configurable via ROS parameters.

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

2.  **Set Up a Python Virtual Environment (Optional but Recommended)**:
    To avoid conflicts with system-wide packages or other projects, it is strongly recommended to create and activate a Python virtual environment before installing dependencies. You can use standard tools like `venv` or `virtualenv`. This ensures that the large number of TTS libraries are isolated to this project.

3.  **Install Python Dependencies**:
    With your virtual environment activated, install the necessary Python libraries using pip.

    ```bash
    cd ~/ros2_ws
    pip install -r src/bob_coquitts/requirements.txt
    ```
    or install them manually
    
    ```bash
    pip install TTS sounddevice numpy
    ```

### A Note on TTS Dependencies

It is important to note that the `TTS` library is a large and complex package. While the `requirements.txt` file is short, running `pip install TTS` will trigger the download and installation of numerous other libraries that it depends on.

These dependencies include major deep learning and scientific computing libraries, such as:

-   `torch` (PyTorch for deep learning)
-   `transformers` & `huggingface-hub` (for loading models like `bark`)
-   `scipy`
-   And many others.

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

After building, source the workspace's `setup.bash` file and run the node using `ros2 run`. For detailed troubleshooting, launch the node with the argument `--ros-args` `--log-level DEBUG` to see all verbose diagnostic messages, including the redirected output from the underlying TTS library.

```bash
# Source the local workspace
source ~/ros2_ws/install/setup.bash

# Launch the node
ros2 run bob_coquitts tts
```

### Example 1: Standard Single-Speaker Model (Default)

This uses the `tts_models/en/ljspeech/vits` model, which is a single female English voice.

```bash
ros2 run bob_coquitts tts --ros-args -p device:='cpu'
```

### Example 2: XTTS Voice Cloning Model

This example uses the powerful XTTS v2 model for voice cloning. This **requires** a `reference_wav` file. For best results, use a clean, 10-20 second audio clip of the target voice sampled at 22050Hz or 24000Hz. An example reference WAV file is provided in the `config` directory of this package.

```bash
ros2 run bob_coquitts tts --ros-args \
-p model_name:='tts_models/multilingual/multi-dataset/xtts_v2' \
-p reference_wav:='/path/to/your/voice.wav' \
-p language:='en' \
-p device:='cuda'

# In another terminal
ros2 topic pub --once --keep-alive 1.0 /text std_msgs/msg/String "data: 'The Eiffel Tower, illuminated at night, offers a spectacular view over the entire city of Paris.'"
```

### Example 3: German Single-Speaker Model

This example uses a VITS model. Since it's a single-language model, the `language` parameter is not needed. Also a reference WAV file is not needed.

```bash
ros2 run bob_coquitts tts --ros-args \
-p model_name:='tts_models/de/css10/vits-neon' \
-p device:='cpu'

# In another terminal
ros2 topic pub --once --keep-alive 1.0 /text std_msgs/msg/String "data: 'Die Integration von kreativen Modellen und Algorithmen zur Förderung der Kreativität wäre ebenfalls sehr sinnvoll.'"
```

### Example 4: Multilingual Voice Cloning (YourTTS)

The `YourTTS` model is another powerful voice cloning model that supports multiple languages but is not based on the XTTS architecture. Like XTTS, it requires a reference WAV file and a language code.

```bash
ros2 run bob_coquitts tts --ros-args \
-p model_name:='tts_models/multilingual/multi-dataset/your_tts' \
-p reference_wav:='/path/to/your/voice.wav' \
-p language:='fr-fr' \
-p device:='cuda'

# In another terminal
ros2 topic pub --once --keep-alive 1.0 /text std_msgs/msg/String "data: Je suis en train d'apprendre à utiliser ce nouveau système, et j'aimerais savoir comment accéder aux paramètres avancés."
```

## ROS Interface

### Subscribed Topics

| Topic Name | Message Type           | Description                                    |
|------------|------------------------|------------------------------------------------|
| `/text`    | `std_msgs/msg/String`  | The text to be synthesized into speech.        |

### Parameters

All parameters can be set at runtime via the command line.

| Parameter Name    | Type    | Default Value                                | Description                                                                                                                              |
|-------------------|---------|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `model_name`      | string  | `tts_models/en/ljspeech/vits`                | The Coqui TTS model to use. Run `tts --list_models` for options. (env: `COQUITTS_MODEL_NAME`)                                              |
| `language`        | string  | `''` (empty)                                 | Language code for multi-lingual models (e.g., `en`, `de`). Leave empty for single-language models. (env: `COQUITTS_LANGUAGE`)              |
| `device`          | string  | `cpu`                                        | The compute device for TTS inference, e.g., `cuda` or `cpu`. (env: `COQUITTS_DEVICE`)                                                      |
| `reference_wav`   | string  | `''` (empty)                                 | Path to a reference WAV file for voice cloning with `xtts` or `your_tts` models. (env: `COQUITTS_REFERENCE_WAV`)                           |
| `sample_rate`     | integer | `24000`                                      | Audio sample rate for playback. **Must match the model's native rate** (e.g., 24000 for XTTS, 22050 for many VITS models). (env: `COQUITTS_SAMPLE_RATE`) |