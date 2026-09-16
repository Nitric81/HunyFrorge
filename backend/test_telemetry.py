import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from .telemetry import StageTelemetry, rss_mb


class TelemetryTests(unittest.TestCase):
    def test_nested_durations_and_memory_scopes_are_not_cumulative(self):
        cuda = Mock()
        cuda.is_initialized.return_value = True
        cuda.memory_allocated.return_value = 100 * 1048576
        cuda.memory_reserved.return_value = 120 * 1048576
        cuda.max_memory_allocated.return_value = 190 * 1048576
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'shape.json'
            meter = StageTelemetry(output, gpu=True)
            with patch.dict('sys.modules', {'torch': SimpleNamespace(cuda=cuda)}), patch('backend.telemetry.rss_mb', return_value=200), patch('backend.telemetry.time.perf_counter', return_value=10) as clock:
                with meter.measure('shape'):
                    clock.return_value = 12
                    with meter.measure('inference'):
                        clock.return_value = 14
                    clock.return_value = 15
            data = json.loads(output.read_text())
            self.assertEqual(data['stages']['shape']['elapsed_seconds'], 5)
            self.assertEqual(data['stages']['inference']['elapsed_seconds'], 2)
            self.assertEqual(data['stages']['shape']['allocator_peak_vram_mb'], 190)
            self.assertIsNone(data['stages']['inference']['allocator_peak_vram_mb'])
            self.assertEqual(data['stages']['inference']['sampled_peak_allocated_vram_mb'], 100)
            self.assertEqual(data['stages']['shape']['peak_rss_mb'], 200)
            cuda.reset_peak_memory_stats.assert_called_once()

    def test_unavailable_gpu_is_null_and_failures_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'api.json'
            meter = StageTelemetry(output)
            with self.assertRaisesRegex(ValueError, 'failure'):
                with meter.measure('export'):
                    raise ValueError('failure')
            stage = json.loads(output.read_text())['stages']['export']
            self.assertEqual(stage['state'], 'failed')
            self.assertIsNone(stage['allocator_peak_vram_mb'])
            self.assertIsNone(stage['sampled_peak_allocated_vram_mb'])
            self.assertGreaterEqual(stage['elapsed_seconds'], 0)

    def test_persistence_errors_are_visible_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            meter = StageTelemetry(Path(directory) / 'api.json')
            with patch.object(meter, '_flush', side_effect=OSError('injected')), self.assertLogs('backend.telemetry', level='WARNING'):
                meter.update('operation')
            meter.update('recovered')
            data = json.loads(meter.path.read_text())
            self.assertEqual(data['telemetry_errors'], ['OSError'])
            self.assertEqual(data['operation'], 'recovered')

    def test_current_process_rss_is_positive_when_available(self):
        value = rss_mb()
        self.assertIsNotNone(value)
        self.assertGreater(value, 0)


if __name__ == '__main__':
    unittest.main()
