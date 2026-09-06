import queue
import sounddevice as sd


class Microphone:
    def __init__(self, audio_queue, samplerate = 16000, channels = 1, dtype = 'int16', block_duration_ms=20, device=None):
        self.audio_queue = audio_queue
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.block_size = int(samplerate * block_duration_ms / 1000)
        self.device = device
        self.stream = None

    def _callback(self, indata, _frames, _time, status):

        if status:
            print(f"Status: {status}")

        try:
            self.audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def start(self):
        
        self.stream = sd.RawInputStream(samplerate=self.samplerate, channels=self.channels, dtype=self.dtype, blocksize=self.block_size, device=self.device, callback=self._callback)
        self.stream.start()


    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None