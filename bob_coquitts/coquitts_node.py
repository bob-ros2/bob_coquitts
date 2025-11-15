import os
import sys
import re
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout
from contextlib import redirect_stderr
# ROS
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType
from std_msgs.msg import String
# TTS
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import soundfile as sf

DEFAULT_MODEL_NAME = 'tts_models/en/ljspeech/vits'
DEFAULT_SAMPLE_RATE = 24000

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
                self._logger.debug(f"Coqui: {line}")

class CoquiTTSnode(Node):
    """
    A ROS 2 node for text-to-speech synthesis using the Coqui TTS library.

    This node subscribes to a 'text' topic, buffers incoming text fragments,
    and intelligently splits them into complete sentences. These sentences are
    then processed in configurable chunks to generate audio. It offers
    extensive ROS parameters to control the TTS model, voice cloning, audio
    playback, and text processing behavior like filtering and trimming.
    It also publishes the text being synthesized to a 'text_speaking' topic.
    """
    def __init__(self):
        """
        Initialize the CoquiTTSnode.

        Declares all ROS parameters for configuration, loads the specified
        TTS model, and sets up a subscriber for incoming text. It also
        initializes a publisher for the text being spoken and a text buffer
        with a flush timer to handle streaming text input.
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
        self.declare_parameter('temperature',
            float(os.environ.get('COQUITTS_TEMPERATURE', 0.2)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="XTTS only: Controls randomness in generation. Lower values are more deterministic. (env: COQUITTS_TEMPERATURE)"
            )
        )
        self.declare_parameter('length_penalty',
            float(os.environ.get('COQUITTS_LENGTH_PENALTY', 1.0)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="XTTS only: A factor to penalize longer sequences. (env: COQUITTS_LENGTH_PENALTY)"
            )
        )
        self.declare_parameter('repetition_penalty',
            float(os.environ.get('COQUITTS_REPETITION_PENALTY', 2.0)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="XTTS only: Penalty for repeating tokens. Higher values reduce repetition. (env: COQUITTS_REPETITION_PENALTY)"
            )
        )
        self.declare_parameter('top_k',
            int(os.environ.get('COQUITTS_TOP_K', 40)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="XTTS only: Samples from the k most likely next tokens. (env: COQUITTS_TOP_K)"
            )
        )
        self.declare_parameter('top_p',
            float(os.environ.get('COQUITTS_TOP_P', 0.9)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="XTTS only: Samples from a nucleus of tokens with a cumulative probability of p. (env: COQUITTS_TOP_P)"
            )
        )
        self.declare_parameter('split_sentences',
            os.environ.get('COQUITTS_SPLIT_SENTENCES', 'False').lower() in ('true', '1', 't'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="Enable/disable Coqui's internal sentence splitting. Disable if you are sending single, complete sentences. (env: COQUITTS_SPLIT_SENTENCES)"
            )
        )
        self.declare_parameter('sentences_max',
            int(os.environ.get('COQUITTS_SENTENCES_MAX', 2)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="Max number of sentences to process at once. Prevents feeding too large chunks to the model. (env: COQUITTS_SENTENCES_MAX)"
            )
        )
        self.declare_parameter('sentence_delimiters',
            os.environ.get('COQUITTS_SENTENCE_DELIMITERS', '.!?\n'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Characters used to split text into sentences. (env: COQUITTS_SENTENCE_DELIMITERS)"
            )
        )
        self.declare_parameter('sentence_end_trim_chars',
            os.environ.get('COQUITTS_SENTENCE_END_TRIM_CHARS', '.,:!?'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="A string of characters to remove from the end of each sentence before synthesis. (env: COQUITTS_SENTENCE_END_TRIM_CHARS)"
            )
        )
        self.declare_parameter('text_filter_chars',
            os.environ.get('COQUITTS_TEXT_FILTER_CHARS', '„”‘“’*-—#<>'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="A string of characters to completely remove from the input text before any processing. (env: COQUITTS_TEXT_FILTER_CHARS)"
            )
        )
        self.declare_parameter('text_filter_regex',
            os.environ.get('COQUITTS_TEXT_FILTER_REGEX', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="A regex pattern to remove matches from the input text. Applied before 'text_filter_chars'. (env: COQUITTS_TEXT_FILTER_REGEX)"
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
            100)

        # Publisher for the text currently being spoken
        self.speaking_pub = self.create_publisher(String, 'text_speaking', 100)

        # Buffer for incoming text and a timer to process it after a pause
        self.text_buffer = ""
        # Timer to process the buffer after a pause in incoming text
        self.flush_timer = None

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

        Filters unwanted characters from the incoming text, then appends it to a
        buffer. It splits the buffer into complete sentences based on configurable
        delimiters. These sentences are processed in chunks, and any incomplete
        sentence fragment is kept in the buffer for the next message. A timer is
        used to flush any remaining text after a pause in the input stream.

        Args:
            msg (std_msgs.msg.String): The message containing the text chunk.
        """
        text_chunk = msg.data

        # 1. Apply Regex Filter
        regex_pattern = self.get_parameter('text_filter_regex').value
        if regex_pattern:
            try:
                text_chunk = re.sub(regex_pattern, '', text_chunk)
            except re.error as e:
                self.get_logger().error(
                    f"Invalid regex for 'text_filter_regex': '{regex_pattern}'. Error: {e}. Skipping regex filter for this chunk.")

        # 2. Apply Character Filter
        filter_chars = self.get_parameter('text_filter_chars').value
        if filter_chars:
            # Create a translation table to remove specified characters
            translation_table = str.maketrans('', '', filter_chars)
            text_chunk = text_chunk.translate(translation_table)

        # Always cancel the flush timer when new text arrives
        if self.flush_timer is not None:
            self.flush_timer.cancel()

        self.text_buffer += text_chunk
        
        # Get delimiters from ROS param and build a safe regex pattern
        delimiters = self.get_parameter('sentence_delimiters').value
        pattern = '|'.join(map(re.escape, delimiters))
        pattern = f'({pattern})'

        # Split the text by delimiters, but keep the delimiters in the list
        parts = re.split(pattern, self.text_buffer)
        
        # If parts has fewer than 2 elements, we don't have a complete sentence yet.
        if len(parts) < 2:
            # (Re)start a timer to flush the buffer if no new text comes in
            self.flush_timer = self.create_timer(0.5, self.flush_buffer_callback)
            return

        # Reconstruct sentences by joining text parts with their delimiters
        sentences = []
        # Iterate in pairs (text, delimiter)
        for i in range(0, len(parts) - 1, 2):
            sentence = (parts[i] + parts[i+1]).strip()
            if sentence: # Avoid adding empty/whitespace-only sentences
                sentences.append(sentence)
        
        # The very last part is the new, incomplete buffer content
        self.text_buffer = parts[-1]

        sentences_max = self.get_parameter('sentences_max').value
        
        # Process sentence chunks if we have any
        if sentences:
            for i in range(0, len(sentences), sentences_max):
                chunk = sentences[i:i + sentences_max]
                text_to_process = " ".join(chunk)
                self._process_text(text_to_process)

        # (Re)start a timer to flush the buffer if no new text comes in
        self.flush_timer = self.create_timer(0.5, self.flush_buffer_callback)

    def flush_buffer_callback(self):
        """
        Processes any remaining text in the buffer after a pause.

        This method is triggered by a timer when no new text has arrived for a
        short period, ensuring that the last sentence or fragment is not left
        unspoken.
        """
        # This timer is a one-shot, so cancel it to prevent it from running again.
        if self.flush_timer is not None:
            self.flush_timer.cancel()
            self.flush_timer = None

        text_to_process = self.text_buffer.strip()
        self.text_buffer = ""  # Clear buffer

        if text_to_process:
            self._process_text(text_to_process)
        else:
            self.get_logger().debug("Flush timer ran, but buffer was empty.")

    def _process_text(self, text_to_process: str):
        """
        The core TTS processing and audio generation logic.

        This function takes a prepared text chunk, performs final trimming of
        unwanted end characters, publishes the text to a topic, and then uses
        the Coqui TTS model to generate audio. The audio is then played or
        saved according to node parameters.

        Args:
            text_to_process (str): The text string to synthesize.
        """
        # Trim unwanted trailing characters from the text before processing
        trim_chars = self.get_parameter('sentence_end_trim_chars').value
        if trim_chars:
            text_to_process = text_to_process.rstrip(trim_chars)

        if not text_to_process:
            return

        # Publish the text that is about to be spoken
        self.speaking_pub.publish(String(data=text_to_process))
        self.get_logger().debug(f"Processing for TTS: '{text_to_process}'")

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
                "text": text_to_process,
                "split_sentences": self.get_parameter('split_sentences').value
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

            if 'xtts' in model_name:
                tts_args["temperature"] = self.get_parameter('temperature').value
                tts_args["length_penalty"] = self.get_parameter('length_penalty').value
                tts_args["repetition_penalty"] = self.get_parameter('repetition_penalty').value
                tts_args["top_k"] = self.get_parameter('top_k').value
                tts_args["top_p"] = self.get_parameter('top_p').value

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