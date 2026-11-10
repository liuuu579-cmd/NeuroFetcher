"""Compact exact-selection backend with an experimental predictor dtype."""
from pathlib import Path

_source = Path(__file__).with_name('reviewer_offload_backend_v12.py').read_text()
_source = _source.replace('V12 compact exact-selection staging',
                          'V15 compact staging with predictor-native lookup dtype')
_source = _source.replace('F.embedding(prefix_ids, self.predictor_embedding).float()',
                          'F.embedding(prefix_ids, self.predictor_embedding).to(next(self.predictor.parameters()).dtype)')
_source = _source.replace('Request-boundary empty_cache precedes timing;',
                          'Predictor lookup uses predictor parameter dtype; Request-boundary empty_cache precedes timing;')
exec(compile(_source, str(Path(__file__).with_name('reviewer_offload_backend_v12.py')), 'exec'), globals())
