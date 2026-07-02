import time
from datetime import timedelta

class PipelineProfiler:
    """Phase 12: Profiling. Track timings per phase to ensure <240s budget."""
    def __init__(self):
        self.phases = {}
        self._current_phase = None
        self._current_start = None
        self._overall_start = time.time()

    def start(self, phase_name: str):
        if self._current_phase:
            self.stop()
        self._current_phase = phase_name
        self._current_start = time.time()

    def stop(self):
        if self._current_phase and self._current_start:
            elapsed = time.time() - self._current_start
            self.phases[self._current_phase] = elapsed
            self._current_phase = None
            self._current_start = None

    def report(self) -> str:
        if self._current_phase:
            self.stop()
            
        total_time = time.time() - self._overall_start
        lines = []
        lines.append(f"\n{'='*50}")
        lines.append("PIPELINE PROFILING REPORT")
        lines.append(f"{'='*50}")
        
        for name, duration in self.phases.items():
            pct = (duration / total_time) * 100 if total_time > 0 else 0
            lines.append(f"  {name:<35} {duration:>6.1f}s  ({pct:3.0f}%)")
            
        lines.append(f"{'-'*50}")
        lines.append(f"  {'TOTAL':<35} {total_time:>6.1f}s")
        lines.append(f"{'='*50}")
        
        return "\n".join(lines)
