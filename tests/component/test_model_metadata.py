"""Deployment descriptions are external to, and bound to, a weight file."""
import json

import pytest

from hoshicore.component.model_metadata import load_model_metadata, validate_model_metadata


@pytest.mark.parametrize('task,adapter', [('segmentation', 'sky_logits'),
                                        ('detection', 'boxes'), ('image_transform', 'denoise')])
def test_common_envelope_handles_multiple_task_interfaces(tmp_path, task, adapter):
    description = dict(schema_version=1, model={'file': 'model.onnx'}, task=task, adapter=adapter,
                       inputs=[dict(name='image', dtype='float32', layout='NCHW')],
                       outputs=[dict(name='result', dtype='float32', layout='NCHW')])
    (tmp_path / 'metadata.json').write_text(json.dumps(description), encoding='utf-8')
    assert load_model_metadata(tmp_path / 'model.onnx') == description
    with pytest.raises(ValueError, match='not different.onnx'):
        load_model_metadata(tmp_path / 'different.onnx')


def test_missing_description_is_distinct_from_invalid_description(tmp_path):
    assert load_model_metadata(tmp_path / 'model.onnx') is None
    (tmp_path / 'metadata.json').write_text('{broken', encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid model metadata'):
        load_model_metadata(tmp_path / 'model.onnx')
    with pytest.raises(ValueError, match='schema_version'):
        validate_model_metadata({'schema_version': 2})
