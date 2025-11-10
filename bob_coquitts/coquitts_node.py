import os
import sys
from io import StringIO
from contextlib import redirect_stdout
from contextlib import redirect_stderr
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType
from std_msgs.msg import String
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import soundfile as sf
from pathlib import Path

DEFAULT_MODEL_NAME = 'tts_models/en/ljspeech/vits'
DEFAULT_SAMPLE_RATE = 24000
#XTTS_MODEL_NAME = 'tts_models/multilingual/multi-dataset/xtts_v2'
#XTTS_REFERENCE_WAV_PATH = "/blue/dev/TTS/eva_24khz.wav"

class RedirectOutput:
    """
    Context manager to redirect stdout and stderr to a logger.

    This is a workaround to capture and redirect the verbose print statements
    from the Coqui TTS library into the ROS logger, keeping the console clean.
    """
    def __init__(self, logger):
        self._logger = logger
        self._buffer = StringIO()

    def __enter__(self):
        self._stdout_redirect = redirect_stdout(self._buffer)
        self._stderr_redirect = redirect_stderr(self._buffer)
        self._stdout_redirect.__enter__()
        self._stderr_redirect.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stdout_redirect.__exit__(exc_type, exc_val, exc_tb)
        self._stderr_redirect.__exit__(exc_type, exc_val, exc_tb)
        output = self._buffer.getvalue().strip()
        if output:
            for line in output.splitlines():
                self._logger.debug(line)

class CoquiTTSnode(Node):
    """
    A ROS 2 node for text-to-speech synthesis using the Coqui TTS library.

    This node subscribes to a topic to receive text messages and uses a
    Coqui TTS model to generate audio, which is then played through the
    system's default audio output device.
    """
    def __init__(self):
        """
        Initialize the CoquiTTSnode.

        This involves declaring ROS parameters for configuration, loading the
        specified TTS model, and setting up a subscriber for incoming text.
        """
        super().__init__('tts')

        # ROS parameters

        self.declare_parameter('model_name', 
            os.environ.get('COQUITTS_MODEL_NAME', DEFAULT_MODEL_NAME),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="The Coqui TTS model to use. Run 'tts --list_models' to see options. (env: COQUITTS_MODEL_NAME)"
            )
        )
        self.declare_parameter('language', 
            os.environ.get('COQUITTS_LANGUAGE', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Language code for multi-lingual models (e.g., 'en', 'de'). Leave empty for single-language models. (env: COQUITTS_LANGUAGE)"
            )
        )
        self.declare_parameter('device', 
            os.environ.get('COQUITTS_DEVICE', 'cpu'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="The compute device for TTS inference, e.g., 'cuda' for NVIDIA GPUs or 'cpu'. (env: COQUITTS_DEVICE)"
            )
        )
        self.declare_parameter('reference_wav', 
            os.environ.get('COQUITTS_REFERENCE_WAV', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Path to a reference WAV file for voice cloning with models like XTTS or YourTTS. (env: COQUITTS_REFERENCE_WAV)"
            )
        )
        self.declare_parameter('sample_rate', 
            int(os.environ.get('COQUITTS_SAMPLE_RATE', DEFAULT_SAMPLE_RATE)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="Audio sample rate for playback. Must match the model's native rate (e.g., 24000 for XTTS). (env: COQUITTS_SAMPLE_RATE)"
            )
        )
        self.declare_parameter('play_audio',
            os.environ.get('COQUITTS_PLAY_AUDIO', 'True').lower() in ('true', '1', 't'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="If true, plays the generated audio directly. (env: COQUITTS_PLAY_AUDIO)"
            )
        )
        self.declare_parameter('output_wav_path',
            os.environ.get('COQUITTS_OUTPUT_WAV_PATH', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Path to save the output WAV file. If empty and play_audio is false, a default name is used. (env: COQUITTS_OUTPUT_WAV_PATH)"
            )
        )

        # Check in case of a xtts or your_tts model type that a reference_wav was provided
        if ('xtts' in self.get_parameter('model_name').value \
            or 'your_tts' in self.get_parameter('model_name').value) \
            and not self.get_parameter('reference_wav').value:
            self.get_logger().error(
                f"Model '{self.get_parameter('model_name').value}' "
                "requires a reference WAV file for voice cloning. "
                "Please set the 'reference_wav' parameter.")
            sys.exit(1)

        self.get_logger().info(
            f"Loading Coqui TTS model {self.get_parameter('model_name').value}")

        try:
            self.tts = TTS(
                model_name=self.get_parameter('model_name').value,
                ).to(self.get_parameter('device').value)
            self.get_logger().info("Coqui TTS model loaded successfully.")

        except Exception as e:
            self.get_logger().error(f"Failed to load TTS model: {e}")
            sys.exit(1)

        self.sub = self.create_subscription(
            String,
            'text',
            self.text_callback,
            10)

    def _get_unique_filepath(self, filepath: Path) -> Path:
        """
        Generates a unique filepath by appending a counter if the file already exists.

        Args:
            filepath (pathlib.Path): The desired initial filepath.

        Returns:
            pathlib.Path: The original filepath if it's unique, otherwise a new
                          unique filepath with a numbered suffix.
        """
        if not filepath.exists():
            return filepath

        parent = filepath.parent
        stem = filepath.stem
        suffix = filepath.suffix
        counter = 1

        while True:
            new_filepath = parent / f"{stem}_{counter:03d}{suffix}"
            if not new_filepath.exists():
                return new_filepath
            counter += 1

    def text_callback(self, msg):
        """
        Callback for the /text topic.

        Receives a string message, generates audio using the TTS model,
        and plays it.

        Args:
            msg (std_msgs.msg.String): The message containing the text to synthesize.
        """
        text = msg.data
        self.get_logger().debug(f"Received text for TTS: '{text}'")

        try:
            play_audio = self.get_parameter('play_audio').value
            output_wav_path = self.get_parameter('output_wav_path').value
            
            save_path_str = output_wav_path
            # If playback is disabled and no path is given, use a default filename
            if not play_audio and not save_path_str:
                save_path_str = "tts_output.wav"

            # If we are neither playing nor saving, do nothing.
            if not play_audio and not save_path_str:
                self.get_logger().warn("Node is configured to neither play nor save audio. No action taken.")
                return

            # 1. Prepare arguments for TTS
            reference_wav = self.get_parameter('reference_wav').value
            model_name = self.get_parameter('model_name').value
            
            tts_args = {
                "text": text,
            }

            if self.get_parameter('language').value:
                tts_args["language"] = self.get_parameter('language').value

            if reference_wav:
                if not os.path.exists(reference_wav):
                    self.get_logger().error(f"Reference WAV file not found at: {reference_wav}")
                    return
                tts_args["speaker_wav"] = reference_wav
            elif 'xtts' in model_name:
                self.get_logger().error(
                    f"Model '{model_name}' requires a reference WAV file for voice cloning. "
                    "Please set the 'reference_wav' parameter.")
                return

            # Generate the audio using the received text
            with RedirectOutput(self.get_logger()):
                wav_data = self.tts.tts(**tts_args)

            # 2. Convert to NumPy array
            wav_np = np.array(wav_data)
            samplerate = self.get_parameter('sample_rate').value

            # 3. Play the audio on the system's default audio output if enabled
            if play_audio:
                self.get_logger().debug(
                    "Playing generated audio...")
                sd.play(wav_np, samplerate=samplerate)
                sd.wait()
                self.get_logger().debug(
                    "Audio playback finished.")

            # 4. Save the audio to a file if a path is specified
            if save_path_str:
                unique_filepath = self._get_unique_filepath(Path(save_path_str))
                self.get_logger().info(f"Saving audio to: {unique_filepath}")
                sf.write(unique_filepath, wav_np, samplerate)

        except Exception as e:
            self.get_logger().error(
                f"Error during audio generation or playback: {e}")

def main(args=None):
    """
    Main entry point for the ROS node.
    """
    rclpy.init(args=args)
    node = CoquiTTSnode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()