import argparse
from datetime import timedelta
import os
from faster_whisper import WhisperModel


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Generate .srt file from an input audio file.")

    # transcribe params
    parser.add_argument(
        "audio", help="Path to the input file (or a file-like object), or the audio waveform.")
    parser.add_argument("-o", "--output", help="Output .srt file")
    parser.add_argument('--language', type=str, default=None,
                        help='the language spoken in the audio')
    parser.add_argument('--task', type=str, default='transcribe',
                        help='task to execute (transcribe or translate)')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='beam size to use for decoding')
    parser.add_argument('--best_of', type=int, default=5,
                        help='number of candidates when sampling with non-zero temperature')
    parser.add_argument('--patience', type=float, default=1,
                        help='beam search patience factor')
    parser.add_argument('--length_penalty', type=float,
                        default=1, help='exponential length penalty constant')
    parser.add_argument('--temperature', type=float, nargs='+', default=[
                        0.0, 0.2, 0.4, 0.6, 0.8, 1.0], help='temperature for sampling. It can be a tuple of temperatures')
    parser.add_argument('--compression_ratio_threshold', type=float, default=2.4,
                        help='if the gzip compression ratio is above this value, treat as failed')
    parser.add_argument('--log_prob_threshold', type=float, default=-1.0,
                        help='if the average log probability over sampled tokens is below this value, treat as failed')
    parser.add_argument('--no_speech_threshold', type=float, default=0.6,
                        help='if the no_speech probability is higher than this value AND the average log probability over sampled tokens is below `log_prob_threshold`, consider the segment as silent')
    parser.add_argument('--condition_on_previous_text', type=bool, default=True,
                        help='if True, the previous output of the model is provided as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop')
    parser.add_argument('--initial_prompt', type=str, default=None,
                        help='optional text to provide as a prompt for the first window')
    parser.add_argument('--prefix', type=str, default=None,
                        help='optional text to provide as a prefix for the first window')
    parser.add_argument('--suppress_blank', type=bool, default=True,
                        help='suppress blank outputs at the beginning of the sampling')
    parser.add_argument('--suppress_tokens', type=int, nargs='+',
                        default=[-1], help='list of token IDs to suppress. -1 will suppress a default set of symbols as defined in the model config.json file')
    parser.add_argument('--without_timestamps', type=bool,
                        default=False, help='only sample text tokens')
    parser.add_argument('--max_initial_timestamp', type=float, default=1.0,
                        help='the initial timestamp cannot be later than this')
    parser.add_argument('--word_timestamps', type=bool, default=False,
                        help='extract word-level timestamps using the cross-attention pattern and dynamic time warping, and include the timestamps for each word in each segment')
    parser.add_argument('--prepend_punctuations', type=str,
                        default='\"\'“¿([{-', help='if word_timestamps is True, merge these punctuation symbols with the next word')
    parser.add_argument('--append_punctuations', type=str, default='\"\'.。,，!！?？:：”)]}、',
                        help='if word_timestamps is True, merge these punctuation symbols with the previous word')
    parser.add_argument('--vad_filter', type=bool, default=False,
                        help='Enable the voice activity detection (VAD) to filter out parts of the audio without speech. This step is using the Silero VAD model https://github.com/snakers4/silero-vad.')

    # WhisperModel params
    parser.add_argument('--model_size_or_path', type=str, default='medium',
                        help='Size of the model to use (tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, or large-v2) or a path to a converted model directory. When a size is configured, the converted model is downloaded from the Hugging Face Hub.')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use for computation ("cpu", "cuda", "auto").')
    parser.add_argument('--device_index', type=int, nargs='+', default=[
                        0], help='Device ID to use. The model can also be loaded on multiple GPUs by passing a list of IDs (e.g. [0, 1, 2, 3]). In that case, multiple transcriptions can run in parallel when transcribe() is called from multiple Python threads (see also num_workers).')
    parser.add_argument('--compute_type', type=str, default='default',
                        help='Type to use for computation. See https://opennmt.net/CTranslate2/quantization.html.')
    parser.add_argument('--cpu_threads', type=int, default=4,
                        help='Number of threads to use when running on CPU (4 by default). A non zero value overrides the OMP_NUM_THREADS environment variable.')
    parser.add_argument('--num_workers', type=int, default=1, help='When transcribe() is called from multiple Python threads, having multiple workers enables true parallelism when running the model (concurrent calls to self.model.generate() will run in parallel). This can improve the global throughput at the cost of increased memory usage.')
    parser.add_argument('--download_root', type=str, default=None,
                        help='Directory where the models should be saved. If not set, the models are saved in the standard Hugging Face cache directory.')
    parser.add_argument('--local_files_only', type=bool, default=False,
                        help='If True, avoid downloading the file and return the path to the local cached file if it exists.')

    # Parse arguments
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.isfile(args.audio):
        print("Input file %s does not exist." % args.audio)
        exit()

    # Create WhisperModel instance
    model = WhisperModel(
        model_size_or_path=args.model_size_or_path,
        device=args.device,
        device_index=args.device_index,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        num_workers=args.num_workers,
        download_root=args.download_root,
        local_files_only=args.local_files_only
    )

    # Generate subtitle
    segments, info = model.transcribe(
        audio=args.audio,
        language=args.language,
        task=args.task,
        beam_size=args.beam_size,
        best_of=args.best_of,
        patience=args.patience,
        length_penalty=args.length_penalty,
        temperature=args.temperature,
        compression_ratio_threshold=args.compression_ratio_threshold,
        log_prob_threshold=args.log_prob_threshold,
        no_speech_threshold=args.no_speech_threshold,
        condition_on_previous_text=args.condition_on_previous_text,
        initial_prompt=args.initial_prompt,
        prefix=args.prefix,
        suppress_blank=args.suppress_blank,
        suppress_tokens=args.suppress_tokens,
        without_timestamps=args.without_timestamps,
        max_initial_timestamp=args.max_initial_timestamp,
        word_timestamps=args.word_timestamps,
        prepend_punctuations=args.prepend_punctuations,
        append_punctuations=args.append_punctuations,
        vad_filter=args.vad_filter
    )

    # Set default output filename
    if not args.output:
        # Use the input filename with a .srt extension
        args.output = os.path.splitext(args.audio)[0] + ".srt"

    # Write subtitle to output file
    with open(args.output, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments):
            segmentId = i+1
            startTime = str(0)+str(timedelta(seconds=int(segment.start)))+',000'
            endTime = str(0)+str(timedelta(seconds=int(segment.end)))+',000'
            text = segment.text
            newSegment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] == ' ' else text}\n\n"
            f.write(newSegment)

    print("Subtitle file saved to %s" % args.output)


if __name__ == "__main__":
    main()
