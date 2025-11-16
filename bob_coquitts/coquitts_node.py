import os
import sys
import logging
import regex as re # use extended regex
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout
from contextlib import redirect_stderr
# ROS
import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity
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
    def __init__(self, logger, level='info'):
        self._logger = logger
        self._log_level = level
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
                if self._log_level == 'debug':
                    self._logger.debug(f"Coqui: {line}")
                elif self._log_level == 'warn':
                    self._logger.warn(f"Coqui: {line}")
                elif self._log_level == 'info':
                    self._logger.info(f"Coqui: {line}")

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

        # Synchronize logging level with ROS logger verbosity for library output.
        logging.basicConfig(
            level = (logging.DEBUG
                if self.get_logger().get_effective_level() \
                    == LoggingSeverity.DEBUG \
                else logging.INFO),
            format="[%(levelname)s] [%(asctime)s.] [%(name)s]: %(message)s",
            datefmt="%s")

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
            int(os.environ.get('COQUITTS_SENTENCES_MAX', 1)),
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
        self.declare_parameter('sentence_strip_chars',
            os.environ.get('COQUITTS_SENTENCE_STRIP_CHARS', '.,:!? '),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="A string of characters to remove from the beginning and end of each sentence before synthesis. (env: COQUITTS_SENTENCE_STRIP_CHARS)"
            )
        )
        self.declare_parameter('text_filter_chars',
            os.environ.get('COQUITTS_TEXT_FILTER_CHARS', '„”‘“’*—#<>'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="A string of characters to completely remove from the input text before any processing. (env: COQUITTS_TEXT_FILTER_CHARS)"
            )
        )
        self.declare_parameter('text_filter_regex',
            os.environ.get('COQUITTS_TEXT_FILTER_REGEX', r'[\p{Emoji_Presentation}\p{Extended_Pictographic}]'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="A regex pattern to remove matches from the input text. Applied before 'text_filter_chars'. (env: COQUITTS_TEXT_FILTER_REGEX)"
            )
        )
        self.declare_parameter('number_thousands_separator',
            os.environ.get('COQUITTS_NUMBER_THOUSANDS_SEPARATOR', '.'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="The character used as a thousands separator in numbers (e.g., '.' in '1.234'). This character will be removed from numbers before synthesis. (env: COQUITTS_NUMBER_THOUSANDS_SEPARATOR)"
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
            with RedirectOutput(self.get_logger(), 'info'):
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
            1000)

        # Publisher for the text currently being spoken
        self.speaking_pub = self.create_publisher(String, 'text_speaking', 1000)

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
        Callback for the /text topic; appends text and manages a flush timer.

        This function's sole responsibility is to append incoming raw text chunks
        to the internal buffer and reset a timer. All complex processing is
        deferred to the `flush_buffer_callback` to ensure that text is only
        processed after a natural pause in the input stream.

        Args:
            msg (std_msgs.msg.String): The message containing the text chunk.
        """
        # Always cancel the flush timer when new text arrives
        if self.flush_timer is not None:
            self.flush_timer.cancel()

        # STEP 1: Append the RAW text chunk to the buffer. That's it.
        self.text_buffer += msg.data

        # STEP 2: (Re)start a timer to process the buffer after a pause.
        self.flush_timer = self.create_timer(0.5, self.flush_buffer_callback)

    def flush_buffer_callback(self):
        """
        Normalizes, splits, and processes the entire text buffer for TTS.

        This method is the core of the text processing pipeline, triggered by a
        timer after a pause in incoming text. It performs the following steps:
        1. Takes the entire accumulated raw text from the buffer.
        2. Applies the full normalization suite (regex filter, character filter,
           and number separator removal) to the complete text block.
        3. Based on the `split_sentences` parameter, it either:
           a) (Default) Splits the cleaned text into sentences using custom
              delimiters and processes them in chunks.
           b) Hands the entire cleaned text block to the TTS engine, relying on
              Coqui's internal splitter.
        """
        # This timer is a one-shot, so cancel it.
        if self.flush_timer is not None:
            self.flush_timer.cancel()
            self.flush_timer = None

        if not self.text_buffer.strip():
            self.get_logger().debug("Flush timer ran, but buffer was empty.")
            return

        # Take the entire buffer content and clear the instance buffer immediately.
        text_to_process_full = self.text_buffer
        self.text_buffer = ""

        # --- Start: All Filtering and Normalization on the complete text ---
        # (This part is common to both splitting methods)
        regex_pattern = self.get_parameter('text_filter_regex').value
        if regex_pattern:
            try:
                text_to_process_full = re.sub(regex_pattern, '', text_to_process_full)
            except re.error as e:
                self.get_logger().error(f"Invalid regex: {e}")

        filter_chars = self.get_parameter('text_filter_chars').value
        if filter_chars:
            translation_table = str.maketrans('', '', filter_chars)
            text_to_process_full = text_to_process_full.translate(translation_table)

        separator = self.get_parameter('number_thousands_separator').value
        if separator:
            escaped_separator = re.escape(separator)
            text_to_process_full = re.sub(fr'(?<=\d){escaped_separator}(?=\d)', '', text_to_process_full)
        # --- End: Filtering and Normalization ---

        use_coqui_splitter = self.get_parameter('split_sentences').value

        if use_coqui_splitter:
            # --- PATH 1: COQUI HANDLES SPLITTING ---
            self.get_logger().debug("Using Coqui's internal sentence splitter for the entire text block.")
            if text_to_process_full.strip():
                self._process_text(text_to_process_full)
        else:
            # --- PATH 2: MANUAL SPLITTING ---
            self.get_logger().debug("Using manual splitting via 'sentence_delimiters'.")
            delimiters = self.get_parameter('sentence_delimiters').value
            pattern = '|'.join(map(re.escape, delimiters))
            pattern = f'({pattern})'
            parts = re.split(pattern, text_to_process_full)

            sentences = []
            for i in range(0, len(parts) - 1, 2):
                sentence = (parts[i] + parts[i+1]).strip()
                if sentence:
                    sentences.append(sentence)

            last_fragment = parts[-1].strip()
            if last_fragment:
                sentences.append(last_fragment)

            if sentences:
                sentences_max = self.get_parameter('sentences_max').value
                for i in range(0, len(sentences), sentences_max):
                    chunk = sentences[i:i + sentences_max]
                    text_to_process_chunk = " ".join(chunk)
                    self._process_text(text_to_process_chunk)

    def _process_text(self, text_to_process: str):
        """
        Handles the final synthesis of a given text chunk.

        This function receives a fully prepared string (either a manually
        created sentence chunk or a complete text block for Coqui to split).
        It performs a final strip of leading/trailing characters, publishes the
        text, and calls the Coqui TTS engine to generate and play/save the audio.

        Args:
            text_to_process (str): The text string to synthesize.
        """
        strip_chars = self.get_parameter('sentence_strip_chars').value
        if strip_chars:
            text_to_process = text_to_process.strip(strip_chars)

        if not text_to_process:
            self.get_logger().debug("Text is empty after cleanup, skipping.")
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
            with RedirectOutput(self.get_logger(), 'debug'):
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