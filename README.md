# Faster Whisper CLI

Faster Whisper CLI is a Python package that provides an easy-to-use interface for generating transcriptions and translations from audio files using pre-trained Transformer-based models.

This CLI version of Faster Whisper allows you to quickly transcribe or translate an audio file using a command-line interface.

## Installation

You can install Faster Whisper CLI using `pip`:

```
pip install faster-whisper-cli
```

## Usage

To use Faster Whisper CLI, simply run the `faster-whisper` command followed by the path to the input audio file:

```
faster-whisper path/to/audio.wav
```

This will transcribe the audio file using the default settings and print the output to the console.

You can also specify various options to customize the transcription process:

```
usage: faster-whisper [-h] [-o OUTPUT] [--language LANGUAGE] [--task TASK]
                      [--beam_size BEAM_SIZE] [--best_of BEST_OF]
                      [--patience PATIENCE]
                      [--length_penalty LENGTH_PENALTY]
                      [--temperature TEMPERATURE [TEMPERATURE ...]]
                      [--compression_ratio_threshold COMPRESSION_RATIO_THRESHOLD]
                      [--log_prob_threshold LOG_PROB_THRESHOLD]
                      [--no_speech_threshold NO_SPEECH_THRESHOLD]
                      [--condition_on_previous_text CONDITION_ON_PREVIOUS_TEXT]
                      [--initial_prompt INITIAL_PROMPT] [--prefix PREFIX]
                      [--suppress_blank SUPPRESS_BLANK]
                      [--suppress_tokens SUPPRESS_TOKENS [SUPPRESS_TOKENS ...]]
                      [--without_timestamps WITHOUT_TIMESTAMPS]
                      [--max_initial_timestamp MAX_INITIAL_TIMESTAMP]
                      [--word_timestamps WORD_TIMESTAMPS]
                      [--prepend_punctuations PREPEND_PUNCTUATIONS]
                      [--append_punctuations APPEND_PUNCTUATIONS]
                      [--vad_filter VAD_FILTER] [--model_size_or_path MODEL_SIZE_OR_PATH]
                      [--device DEVICE] [--device_index DEVICE_INDEX [DEVICE_INDEX ...]]
                      [--compute_type COMPUTE_TYPE] [--cpu_threads CPU_THREADS]
                      [--num_workers NUM_WORKERS]
                      audio
```