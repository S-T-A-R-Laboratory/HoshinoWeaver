"""GUI completion must accept video-only results and allow another run."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip('PySide6')
from ui import UIUtils


def test_video_only_completion_can_run_twice(monkeypatch):
    video = {'path': 'result.mp4'}
    run = AsyncMock(return_value={'video': video})
    monkeypatch.setattr(UIUtils, 'run_from_yaml', run)
    monkeypatch.setattr(UIUtils, 'QtSignalTracker', lambda: SimpleNamespace(
        progress_updated=Mock(), finished=Mock()))
    monkeypatch.setattr(UIUtils, 'QMessageBox', Mock(
        side_effect=AssertionError('successful video export must not show an error')))
    window = SimpleNamespace(
        _preview_useable=True, _current_meta_yaml_path='workflow.yaml',
        _input_files={'亮场': ['input.tif']}, _status_n={},
        img_view_label=Mock(), view_next_img=Mock(), view_pre_img=Mock(),
        config_panel=SimpleNamespace(collect_configs=lambda: {},
                                     collect_route_choices=lambda: {}),
        output_panel=SimpleNamespace(
            collect_outputs=lambda: {'enable_sequence_export': False,
                                     'enable_video_export': True,
                                     'video_output_filename': video['path']},
            set_video_result=Mock()))
    handler = SimpleNamespace(
        window=window, set_widget_handleable=Mock(),
        _set_start_btn_stop_mode=Mock(), update_status_display=Mock(),
        update_progress_bar=Mock())
    handler.view_file = lambda *args, **kwargs: UIUtils.SlotHandler.view_file(
        handler, *args, **kwargs)

    async def twice():
        for _ in range(2):
            await UIUtils.SlotHandler.start_task(handler)
            assert window._status == 'successed'
            assert window._status_n['status'] == '任务完成'
            assert window._last_output_path == video['path']
            assert window.output_panel.set_video_result.call_args.args == (video,)
            assert handler.set_widget_handleable.call_args.kwargs == {'handleable': True}

    asyncio.run(twice())
    assert run.await_count == 2
    window.img_view_label.initImg.assert_not_called()
    assert window._preview_img == ['', None]
