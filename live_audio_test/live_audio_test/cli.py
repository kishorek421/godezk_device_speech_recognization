import argparse
import queue
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from walker_audio.models import CanaryTranscriber

from live_audio_test.analyzer import LiveAudioAnalyzer
from live_audio_test.remote import RemoteAnalyzer, default_server_url


class LiveTester:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.work_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self.command_queue: queue.Queue[str] = queue.Queue()
        self.rolling: deque[np.ndarray] = deque()
        self.rolling_samples = 0
        self.running = True
        self.manual = args.manual
        if args.server_url:
            self.analyzer = RemoteAnalyzer(
                args.server_url, args.sample_rate, args.request_timeout
            )
        else:
            transcriber = None
            if not args.no_asr:
                transcriber = CanaryTranscriber(args.model, args.model_device, args.dtype)
            self.analyzer = LiveAudioAnalyzer(
                args.sample_rate, transcriber, not args.no_denoise, args.vad_threshold_db
            )

    def audio_callback(
        self, input_data: np.ndarray, frames: int, timing: object, status: object
    ) -> None:
        if status:
            print(f"\naudio status: {status}")
        self.audio_queue.put(input_data[:, 0].copy())

    def command_reader(self) -> None:
        while self.running:
            try:
                command = input().strip().lower()
            except EOFError:
                command = "q"
            self.command_queue.put(command)
            if command == "q":
                return

    def worker(self) -> None:
        while self.running:
            audio = self.work_queue.get()
            if audio is None:
                return
            started = time.perf_counter()
            try:
                path = self.save(audio)
                result = self.analyzer.analyze(audio)
                latency = time.perf_counter() - started
                print(f"\n\n{datetime.now(UTC).astimezone():%H:%M:%S} analysis ({latency:.2f}s)")
                print(f"saved: {path}")
                print(
                    f"peak: {result.peak_db:.1f} dBFS | rms: {result.rms_db:.1f} dBFS | "
                    f"speech: {result.speech_seconds:.2f}s"
                )
                if result.transcript:
                    print(f"transcript: {result.transcript}")
                elif result.reason:
                    print(f"result: {result.reason}")
            except Exception as exc:
                print(f"\nanalysis error: {exc}")
            finally:
                self.work_queue.task_done()

    def save(self, audio: np.ndarray) -> Path:
        self.args.output.mkdir(parents=True, exist_ok=True)
        path = self.args.output / f"mic-{datetime.now(UTC).astimezone():%Y%m%d-%H%M%S-%f}.wav"
        sf.write(path, audio, self.args.sample_rate)
        return path

    def submit(self) -> None:
        if self.work_queue.full() or not self.rolling:
            return
        audio = np.concatenate(tuple(self.rolling))
        self.work_queue.put_nowait(audio)
        self.rolling.clear()
        self.rolling_samples = 0

    def drain_audio(self) -> tuple[float, bool]:
        peak = 0.0
        endpoint_triggered = False
        frame_threshold = getattr(self.args, "speech_threshold", 0.015)
        min_silence_samples = int(self.args.sample_rate * getattr(self.args, "endpoint_silence_ms", 350) / 1000)
        min_speech_samples = int(self.args.sample_rate * getattr(self.args, "min_speech_seconds", 1.2))
        max_chunk_samples = int(self.args.sample_rate * getattr(self.args, "max_chunk_seconds", 12.0))

        while True:
            try:
                chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            self.rolling.append(chunk)
            self.rolling_samples += len(chunk)
            chunk_peak = float(np.max(np.abs(chunk))) if len(chunk) > 0 else 0.0
            peak = max(peak, chunk_peak)

            # Speech energy monitoring for speech-aware endpointing
            if chunk_peak >= frame_threshold:
                self.speech_samples_accumulated += len(chunk)
                self.silence_samples_accumulated = 0
            else:
                if self.speech_samples_accumulated > 0:
                    self.silence_samples_accumulated += len(chunk)

            # Check endpoint condition: silence after sufficient speech, or safety limit
            if (
                self.speech_samples_accumulated >= min_speech_samples
                and self.silence_samples_accumulated >= min_silence_samples
            ) or (self.rolling_samples >= max_chunk_samples):
                endpoint_triggered = True
                self.speech_samples_accumulated = 0
                self.silence_samples_accumulated = 0

        return peak, endpoint_triggered

    def handle_commands(self) -> None:
        while True:
            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                return
            if command == "q":
                self.running = False
            elif command == "m":
                self.manual = not self.manual
                print(f"\nmode: {'manual' if self.manual else 'auto'}")
            elif command in {"", "r"}:
                self.submit()

    def run(self) -> None:
        worker = threading.Thread(target=self.worker, daemon=True)
        commands = threading.Thread(target=self.command_reader, daemon=True)
        worker.start()
        commands.start()
        self.speech_samples_accumulated = 0
        self.silence_samples_accumulated = 0
        print("Commands followed by Enter: r/blank analyze | m toggle mode | q quit")
        print(
            f"Mode: {'manual' if self.manual else 'speech-aware'} | "
            f"silence endpoint: {self.args.endpoint_silence_ms}ms | max: {self.args.max_chunk_seconds}s"
        )
        if self.args.server_url:
            print(f"ASR: remote {self.args.server_url}")
        with sd.InputStream(
            samplerate=self.args.sample_rate,
            channels=1,
            dtype="float32",
            device=self.args.input_device,
            blocksize=self.args.block_size,
            callback=self.audio_callback,
        ):
            while self.running:
                peak, endpoint_ready = self.drain_audio()
                meter = min(40, int(peak * 80))
                print(f"\rMic [{'#' * meter:<40}] peak {peak:0.3f}", end="", flush=True)
                self.handle_commands()
                if not self.manual and endpoint_ready:
                    self.submit()
                time.sleep(0.05)
        self.work_queue.put(None)
        print("\nstopped")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="walker-live-audio")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--input-device")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--block-size", type=int, default=1600)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--endpoint-silence-ms", type=int, default=350, help="Min silence to trigger chunk endpoint")
    parser.add_argument("--max-chunk-seconds", type=float, default=12.0, help="Max duration of speech chunk")
    parser.add_argument("--min-speech-seconds", type=float, default=1.0, help="Min speech before endpointing")
    parser.add_argument("--speech-threshold", type=float, default=0.018, help="Peak amplitude threshold for active speech")
    parser.add_argument("--manual", action="store_true")
    asr_group = parser.add_mutually_exclusive_group()
    asr_group.add_argument("--no-asr", action="store_true")
    asr_group.add_argument(
        "--server-url",
        nargs="?",
        const=default_server_url(),
        default=None,
        help="send audio to a remote transcription API instead of running Canary locally "
        "(defaults to AUDIO_API_URL or the deployed server)",
    )
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--vad-threshold-db", type=float, default=3.5)
    parser.add_argument("--output", type=Path, default=Path("recordings"))
    parser.add_argument("--model", default="nvidia/canary-qwen-2.5b")
    parser.add_argument("--model-device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    return parser



def main() -> None:
    args = build_parser().parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    try:
        LiveTester(args).run()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
