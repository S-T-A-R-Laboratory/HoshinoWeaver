"""Read a model's deployment description; never imports a model runtime."""
import json
from pathlib import Path


def validate_model_metadata(metadata, *, model_name=None):
    """Validate the common envelope; task adapters validate their own semantics.

    schema_version versions this project's JSON structure, not the model or ONNX.
    Descriptions declare compatibility; they cannot prove the graph's semantics.
    """
    if not isinstance(metadata, dict) or type(metadata.get('schema_version')) is not int or metadata['schema_version'] != 1:
        raise ValueError('Unsupported model metadata schema_version; expected 1')
    model = metadata.get('model')
    if (not isinstance(model, dict) or not isinstance(model.get('file'), str)
            or not model['file'] or '/' in model['file'] or '\\' in model['file']
            or model['file'] in ('.', '..')):
        raise ValueError('Model metadata must name a weight file in its own directory')
    if model_name is not None and model['file'] != model_name:
        raise ValueError(f'Model metadata describes {model["file"]}, not {model_name}')
    if metadata.get('task') not in ('segmentation', 'detection', 'image_transform'):
        raise ValueError('Model task must be segmentation, detection or image_transform')
    if not isinstance(metadata.get('adapter'), str) or not metadata['adapter']:
        raise ValueError('Model metadata requires an adapter identifier')
    for direction in ('inputs', 'outputs'):
        tensors = metadata.get(direction)
        if not isinstance(tensors, list) or not tensors:
            raise ValueError(f'Model metadata requires nonempty {direction}')
        names = []
        for tensor in tensors:
            if not isinstance(tensor, dict) or any(
                    not isinstance(tensor.get(key), str) or not tensor[key]
                    for key in ('name', 'dtype', 'layout')):
                raise ValueError(f'Each {direction} tensor requires name, dtype and layout')
            names.append(tensor['name'])
        if len(names) != len(set(names)):
            raise ValueError(f'Duplicate {direction} tensor names')
    return metadata


def load_model_metadata(model_path):
    """Load adjacent metadata.json, or return None when none was supplied.

    A present but malformed or mismatched description is always an error.
    """
    model_path = Path(model_path)
    path = model_path.parent / 'metadata.json'
    if not path.is_file():
        return None
    try:
        metadata = json.loads(path.read_text(encoding='utf-8'))
        return validate_model_metadata(metadata, model_name=model_path.name)
    except (ValueError, TypeError) as error:
        raise ValueError(f'Invalid model metadata {path}: {error}') from error
