"""Simple request scheduler for SSD-macOS inference engine."""
from ssd_macos.config import Config
from ssd_macos.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(self, config: Config):
        self.config = config
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Returns (sequences_to_run, is_prefill).

        Moves one waiting sequence to running (prefill phase),
        or returns all running sequences (decode phase).
        """
        # Remove finished sequences from running
        self.running = [
            s for s in self.running if s.status != SequenceStatus.FINISHED
        ]

        # Move waiting to running (prefill)
        if self.waiting:
            seq = self.waiting.pop(0)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            return [seq], True

        # Continue running (decode)
        return self.running[:], False

    def is_finished(self) -> bool:
        return len(self.waiting) == 0 and len(self.running) == 0

    def num_waiting(self) -> int:
        return len(self.waiting)

    def num_running(self) -> int:
        return len(self.running)
