"""Photoshop is always mocked; these tests need no Adobe installation or desktop."""
import asyncio
import json
from pathlib import Path
import threading
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile

from hoshicore.component import dataloader, photoshop
from hoshicore.services import photoshop as photoshop_service
from hoshicore.component.exif import ExifData
from hoshicore.services import create_service_registry


def loader(*, services=None, **options):
    return dataloader.ImgFileListLoader(None, None, {
        'raw_backend': 'photoshop', **options}, services=services)


@pytest.mark.asyncio
async def test_non_raw_keeps_four_concurrent_reads_without_constructing_ps(monkeypatch):
    barrier = threading.Barrier(4, timeout=3)
    constructor = Mock(side_effect=AssertionError('Photoshop must stay lazy'))
    monkeypatch.setattr(photoshop_service, 'PhotoshopRawRenderer', constructor)

    def read(path):
        barrier.wait()
        return path

    monkeypatch.setattr(dataloader, 'load_img', read)
    metadata = Mock(side_effect=AssertionError('Unconnected EXIF must not be read'))
    monkeypatch.setattr(dataloader, 'read_exif_data', metadata)
    source = loader(photoshop_path=None)
    paths = ['a.jpg', 'b.tiff', 'c.PNG', 'd.fits']
    assert await asyncio.wait_for(asyncio.gather(*(source.aload(p) for p in paths)), 5) == [
        (p, None) for p in paths]
    await source.aclose()
    constructor.assert_not_called()
    metadata.assert_not_called()


@pytest.mark.asyncio
async def test_mixed_inputs_serialize_only_raw_and_reuse_one_renderer(monkeypatch):
    entered, release, jpeg_read = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def render(path):
        calls.append(path)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
        return path, ExifData({'rendered': path}, b'rendered-icc')

    renderer = Mock(render=Mock(side_effect=render))
    constructor = Mock(return_value=renderer)
    monkeypatch.setattr(photoshop_service, 'PhotoshopRawRenderer', constructor)

    def read(path):
        jpeg_read.set()
        return path

    monkeypatch.setattr(dataloader, 'load_img', read)
    monkeypatch.setattr(dataloader, 'read_exif_data', lambda path: ExifData({}, b'native'))
    scope = create_service_registry().scope({'photoshop_path': 'mock-photoshop'})
    source = loader(services=scope)
    tasks = [asyncio.create_task(source.aload(p, with_exif=True))
             for p in ['a.CR2', 'b.nef', 'c.jpg']]
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert await asyncio.to_thread(jpeg_read.wait, 3)
        assert calls == ['a.CR2']
    finally:
        release.set()
    results = await asyncio.gather(*tasks)
    await source.aclose()
    renderer.close.assert_not_called()  # The graph, not its Loader, owns the lease.
    await scope.aclose()
    assert [p for p, _ in results] == ['a.CR2', 'b.nef', 'c.jpg']
    assert results[0][1].colorprofile == b'rendered-icc'
    assert results[2][1].colorprofile == b'native'
    constructor.assert_called_once()
    renderer.close.assert_called_once()


@pytest.mark.asyncio
async def test_cancellation_drains_active_render_before_cleanup_and_skips_waiters(monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def render(path):
        entered.set()
        assert release.wait(3)
        finished.set()
        return path, None

    def close():
        assert finished.is_set()

    renderer = Mock(render=Mock(side_effect=render), close=Mock(side_effect=close))
    monkeypatch.setattr(photoshop_service, 'PhotoshopRawRenderer', Mock(return_value=renderer))
    scope = create_service_registry().scope({'photoshop_path': 'mock-photoshop'})
    source = loader(services=scope)
    active = asyncio.create_task(source.aload('a.CR2'))
    waiting = asyncio.create_task(source.aload('b.CR2'))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        active.cancel()
        waiting.cancel()
        await asyncio.sleep(0)
        assert not active.done()
        active.cancel()  # A second graph failure must not abandon the worker.
        await asyncio.sleep(0)
        assert not active.done()
        renderer.close.assert_not_called()
    finally:
        release.set()
    results = await asyncio.gather(active, waiting, return_exceptions=True)
    assert all(isinstance(r, asyncio.CancelledError) for r in results)
    await source.aclose()
    await scope.aclose()
    renderer.render.assert_called_once_with('a.CR2')
    renderer.close.assert_called_once()


@pytest.mark.asyncio
async def test_native_raw_does_not_construct_photoshop(monkeypatch):
    constructor = Mock(side_effect=AssertionError('native RAW must not start PS'))
    monkeypatch.setattr(photoshop_service, 'PhotoshopRawRenderer', constructor)
    monkeypatch.setattr(dataloader, 'load_img', lambda path: path)
    source = loader(raw_backend='native')
    assert await source.aload('a.CR2') == ('a.CR2', None)
    await source.aclose()
    constructor.assert_not_called()


@pytest.fixture
def mock_host(monkeypatch):
    state = []
    identity = {'pid': 123, 'created': 100.5}
    monkeypatch.setattr(photoshop, 'resolve_photoshop', lambda path: Path('mock-photoshop'))
    monkeypatch.setattr(photoshop, 'photoshop_processes', lambda target: list(state))
    quit_mock = Mock()
    monkeypatch.setattr(photoshop, 'quit_photoshop', quit_mock)

    def render(path, **kwargs):
        host = kwargs['_host']
        host._prepare_host(list(state))
        state[:] = [identity]
        host._observe_host(list(state), Mock(pid=identity['pid']))
        return np.array([[[10, 20, 30]]], np.uint16), {
            'tiff_photometric': 'RGB', 'exif': {'Exif.Photo.ExposureTime': '1/8'},
            'icc_profile': b'rendered-icc'}

    render_mock = Mock(side_effect=render)
    monkeypatch.setattr(photoshop, 'render_raw', render_mock)
    return state, identity, render_mock, quit_mock


def test_owned_session_preserves_bgr_dtype_exif_icc_and_quits_once(mock_host):
    state, identity, render, quit_mock = mock_host
    source = photoshop.PhotoshopRawRenderer('mock')
    frame, exif = source.render('a.CR2')
    source.render('b.CR2')
    source.close()
    source.close()
    assert frame.dtype == np.uint16 and frame.flags.c_contiguous
    assert frame.tolist() == [[[30, 20, 10]]]
    assert exif == ExifData({'Exif.Photo.ExposureTime': '1/8'}, b'rendered-icc')
    assert render.call_count == 2
    quit_mock.assert_called_once_with(Path('mock-photoshop'), identity,
                                     terminate_closed_host=(photoshop.sys.platform == 'win32'))
    assert all(not call.kwargs['output_dir'].parent.exists() for call in render.call_args_list)


def test_successful_render_defers_locked_artifact_cleanup_until_host_close(mock_host, monkeypatch):
    import shutil

    state, identity, render, quit_mock = mock_host
    original_remove = shutil.rmtree
    def remove(path, *args, **kwargs):
        if Path(path).name.startswith('hnw-photoshop-') and not quit_mock.called:
            raise PermissionError('launcher.stderr.log is held by the Photoshop host')
        return original_remove(path, *args, **kwargs)
    monkeypatch.setattr(photoshop.shutil, 'rmtree', remove)
    renderer = photoshop.PhotoshopRawRenderer('mock')
    frame, _ = renderer.render('a.CR2')
    assert frame.shape == (1, 1, 3)
    folder = render.call_args.kwargs['output_dir'].parent
    assert folder.exists()
    renderer.close()
    assert not folder.exists()


def test_existing_host_is_automatically_borrowed_and_never_quit(mock_host):
    state, identity, render, quit_mock = mock_host
    state[:] = [identity]
    borrowed = photoshop.PhotoshopRawRenderer('mock')
    borrowed.render('a.CR2')
    borrowed.close()
    render.assert_called_once()
    assert borrowed._identity == identity
    assert not borrowed._owned
    quit_mock.assert_not_called()


def test_timeout_does_not_dispatch_a_quit_or_a_second_render(mock_host):
    state, identity, render, quit_mock = mock_host

    def timeout(*args, **kwargs):
        host = kwargs['_host']
        host._prepare_host(list(state))
        state[:] = [identity]
        host._observe_host(list(state), Mock(pid=identity['pid']))
        raise TimeoutError('opening timed out')

    render.side_effect = timeout
    source = photoshop.PhotoshopRawRenderer('mock')
    with pytest.raises(TimeoutError, match='opening'):
        source.render('a.CR2')
    with pytest.raises(RuntimeError, match='Previous Photoshop request'):
        source.render('b.CR2')
    source.close()
    render.assert_called_once()
    quit_mock.assert_not_called()
    # The real transport reports this retained directory in its timeout error.
    import shutil
    shutil.rmtree(render.call_args.kwargs['output_dir'].parent)


def test_replaced_host_is_not_sent_an_exit_request(monkeypatch):
    identity = {'pid': 123, 'created': 100.5}
    monkeypatch.setattr(photoshop, 'photoshop_processes',
                        lambda target: [{'pid': 123, 'created': 101.0}])
    sender = Mock(side_effect=AssertionError('must not launch a replacement'))
    monkeypatch.setattr(photoshop, 'launch_script', sender)
    photoshop.quit_photoshop(Path('mock'), identity)
    sender.assert_not_called()


def test_exit_timeout_retains_quit_stage_and_sender_diagnostics(monkeypatch, tmp_path):
    identity = {'pid': 123, 'created': 100.5}
    monkeypatch.setattr(photoshop, 'photoshop_processes', lambda target: [identity])
    output = tmp_path / 'exit'
    output.mkdir()
    monkeypatch.setattr(photoshop.tempfile, 'mkdtemp', lambda **kwargs: str(output))
    def send(app, script, folder, timeout, hide):
        (folder / 'status.txt').write_text('quitting', encoding='utf-8')
        return Mock(poll=Mock(return_value=0))
    monkeypatch.setattr(photoshop, 'launch_script', send)
    monkeypatch.setattr(photoshop.time, 'monotonic', Mock(side_effect=[0, 11]))
    with pytest.raises(TimeoutError, match='last status: quitting.*artifacts:'):
        photoshop.quit_photoshop(Path('mock'), identity)
    assert (output / 'quit.jsx').exists()
    assert (output / 'status.txt').read_text(encoding='utf-8') == 'quitting'


def test_exit_refuses_open_documents(monkeypatch):
    identity = {'pid': 123, 'created': 100.5}
    monkeypatch.setattr(photoshop, 'photoshop_processes', lambda target: [identity])

    def send(app, script, output, timeout, hide):
        assert 'app.documents.length' in script.read_text(encoding='utf-8')
        (output / 'status.txt').write_text('documents-open', encoding='utf-8')
        return Mock(poll=Mock(return_value=0))

    monkeypatch.setattr(photoshop, 'launch_script', send)
    with pytest.raises(RuntimeError, match='documents-open'):
        photoshop.quit_photoshop(Path('mock'), identity)


def test_windows_exit_posts_close_only_after_empty_document_handshake(monkeypatch, tmp_path):
    identity = {'pid': 123, 'created': 100.5}
    state = [identity]
    monkeypatch.setattr(photoshop.sys, 'platform', 'win32')
    monkeypatch.setattr(photoshop, 'photoshop_processes', lambda target: list(state))
    def send(app, script, output, timeout, hide):
        text = script.read_text(encoding='utf-8')
        assert 'app.documents.length' in text and 'executeAction' not in text
        (output / 'status.txt').write_text('ready-to-quit', encoding='utf-8')
        return Mock(poll=Mock(return_value=0))
    monkeypatch.setattr(photoshop, 'launch_script', send)
    close = Mock(side_effect=lambda target, current: state.clear())
    monkeypatch.setattr(photoshop, '_request_windows_close', close)
    photoshop.quit_photoshop(Path('mock'), identity)
    close.assert_called_once_with(Path('mock'), identity)


def test_owned_exit_reaps_only_after_verified_handshake_and_normal_close_timeout(monkeypatch):
    identity = {'pid': 123, 'created': 100.5}
    state = [identity]
    monkeypatch.setattr(photoshop.sys, 'platform', 'win32')
    monkeypatch.setattr(photoshop, 'photoshop_processes', lambda target: list(state))
    def send(app, script, output, timeout, hide):
        (output / 'status.txt').write_text('ready-to-quit', encoding='utf-8')
        return Mock(poll=Mock(return_value=0))
    monkeypatch.setattr(photoshop, 'launch_script', send)
    close = Mock(return_value=456)
    monkeypatch.setattr(photoshop, '_request_windows_close', close)
    reap = Mock(side_effect=lambda *args: state.clear())
    monkeypatch.setattr(photoshop, '_terminate_closed_host', reap)
    monkeypatch.setattr(photoshop.time, 'monotonic', Mock(side_effect=[0, 11]))
    photoshop.quit_photoshop(Path('mock'), identity, terminate_closed_host=True)
    close.assert_called_once_with(Path('mock'), identity)
    reap.assert_called_once_with(Path('mock'), identity, 456)


@pytest.mark.parametrize('foreign', [False, True])
def test_windows_exit_distinguishes_own_forwarder_from_foreign_host(monkeypatch, foreign):
    identity = {'pid': 123, 'created': 100.5}
    state = [identity]
    monkeypatch.setattr(photoshop.sys, 'platform', 'win32')
    monkeypatch.setattr(photoshop, 'photoshop_processes', lambda target: list(state))
    def send(app, script, output, timeout, hide):
        state.append({'pid': 999 if foreign else 456, 'created': 101.})
        (output / 'status.txt').write_text('ready-to-quit', encoding='utf-8')
        return Mock(pid=456, poll=Mock(return_value=0))
    monkeypatch.setattr(photoshop, 'launch_script', send)
    close = Mock(side_effect=lambda *args: state.clear())
    monkeypatch.setattr(photoshop, '_request_windows_close', close)
    if foreign:
        with pytest.raises(RuntimeError, match='host changed before exit'):
            photoshop.quit_photoshop(Path('mock'), identity)
        close.assert_not_called()
    else:
        photoshop.quit_photoshop(Path('mock'), identity)
        close.assert_called_once_with(Path('mock'), identity)


@pytest.mark.parametrize('blocked', [None, 'window_open', 'pid_reused', 'foreign_process',
                                    'new_window', 'wrong_exe', 'platform'])
def test_owned_residual_termination_identity_and_window_guards(monkeypatch, blocked):
    identity = {'pid': 123, 'created': 100.5}
    target = Path('Photoshop.exe').resolve()
    process = Mock(create_time=Mock(return_value=100.5), exe=Mock(return_value=str(target)))
    monkeypatch.setattr(photoshop.psutil, 'Process', Mock(return_value=process))
    monkeypatch.setattr(photoshop.sys, 'platform', 'darwin' if blocked == 'platform' else 'win32')
    monkeypatch.setattr(photoshop, '_window_exists', lambda hwnd: blocked == 'window_open')
    current = [identity, {'pid': 456, 'created': 5.}] if blocked == 'foreign_process' else [identity]
    monkeypatch.setattr(photoshop, 'photoshop_processes', lambda path: current)
    monkeypatch.setattr(photoshop, 'photoshop_windows', lambda path: [{'main': True}] if blocked == 'new_window' else [])
    if blocked == 'pid_reused':
        process.create_time.return_value = 101.
    if blocked == 'wrong_exe':
        process.exe.return_value = str(Path('unrelated.exe').resolve())
    if blocked:
        with pytest.raises(RuntimeError, match='refusing termination'):
            photoshop._terminate_closed_host(target, identity, 789)
        process.terminate.assert_not_called()
    else:
        photoshop._terminate_closed_host(target, identity, 789)
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)


def test_transport_reads_pixels_and_metadata_from_rendered_tiff(monkeypatch, tmp_path):
    source = tmp_path / 'input.CR2'
    source.write_bytes(b'mocked RAW input')
    target = tmp_path / 'Photoshop.exe'
    target.touch()
    output = tmp_path / 'render'
    rgb = np.array([[[100, 200, 300], [400, 500, 600]]], np.uint16)
    icc = b'rendered-icc'
    expected_exif = {'Exif.Photo.ExposureTime': '1/8'}
    monkeypatch.setattr(photoshop, 'resolve_photoshop', lambda path: target)
    monkeypatch.setattr(photoshop, 'photoshop_processes', lambda path: [])
    monkeypatch.setattr(photoshop, 'photoshop_windows', lambda *args, **kwargs: [])

    def launch(app, script, folder, timeout, hide):
        text = script.read_text(encoding='utf-8')
        assert 'app.open(new File(cfg.input))' in text
        assert 'new CameraRAWOpenOptions' not in text
        tifffile.imwrite(folder / 'rendered.tif', rgb, photometric='rgb',
                         extratags=[(34675, 'B', len(icc), icc, False)])
        (folder / 'rendered.xmp').write_text('<rendered-xmp/>', encoding='utf-8')
        (folder / 'result.json').write_text(json.dumps({'ok': True}), encoding='utf-8')
        return Mock(pid=999)

    monkeypatch.setattr(photoshop, 'launch_script', launch)
    reader = Mock(read_exif=Mock(return_value=expected_exif), read_icc=Mock(return_value=icc))
    context = Mock(__enter__=Mock(return_value=reader), __exit__=Mock(return_value=False))
    metadata_reader = Mock(return_value=context)
    monkeypatch.setattr(photoshop.pyexiv2, 'ImageData', metadata_reader)
    pixels, metadata = photoshop.render_raw(source, photoshop=target, output_dir=output)
    np.testing.assert_array_equal(pixels, rgb)
    assert metadata['icc_profile'] == icc
    assert metadata['exif'] == expected_exif
    assert metadata['xmp'] == '<rendered-xmp/>'
    metadata_reader.assert_called_once_with((output / 'rendered.tif').read_bytes())


@pytest.fixture
def managed_transport(monkeypatch, tmp_path):
    """Exercise real session/transport control flow, replacing only OS/Adobe calls."""
    source = tmp_path / 'input.CR2'
    source.touch()
    target = tmp_path / 'Photoshop.exe'
    target.touch()
    identity = {'pid': 123, 'created': 100.5}
    state = {'processes': [], 'pending': False, 'on_poll': None,
             'sender_pid': 123, 'sender_exit_code': 0}
    rgb = np.array([[[10, 20, 30]]], np.uint16)
    icc = b'profile'
    original_mkdtemp = photoshop.tempfile.mkdtemp
    monkeypatch.setattr(photoshop.tempfile, 'mkdtemp', lambda suffix=None, prefix=None, dir=None:
                        original_mkdtemp(suffix=suffix, prefix=prefix,
                                         dir=tmp_path if dir is None else dir))
    monkeypatch.setattr(photoshop, 'resolve_photoshop', lambda path: target)
    monkeypatch.setattr(photoshop, 'photoshop_processes', lambda path: list(state['processes']))
    windows = Mock(return_value=[{'main': True, 'visible': False}])
    monkeypatch.setattr(photoshop, 'photoshop_windows', windows)
    quit_mock = Mock()
    monkeypatch.setattr(photoshop, 'quit_photoshop', quit_mock)
    monkeypatch.setattr(photoshop.time, 'sleep', lambda seconds: None)
    reader = Mock(read_exif=Mock(return_value={}), read_icc=Mock(return_value=icc))
    context = Mock(__enter__=Mock(return_value=reader), __exit__=Mock(return_value=False))
    monkeypatch.setattr(photoshop.pyexiv2, 'ImageData', Mock(return_value=context))

    def launch(app, script, output, timeout, hide):
        if not state['processes']:
            state['processes'] = [identity]
        (output / 'stage.txt').write_text('opening', encoding='utf-8')
        def complete():
            tifffile.imwrite(output / 'rendered.tif', rgb, photometric='rgb',
                             extratags=[(34675, 'B', len(icc), icc, False)])
            (output / 'rendered.xmp').write_text('<xmp/>', encoding='utf-8')
            (output / 'result.json').write_text(json.dumps({'ok': True}), encoding='utf-8')

        state['complete'] = complete
        if not state['pending']:
            complete()

        def poll():
            if state['on_poll']:
                state['on_poll']()
            return state['sender_exit_code']

        return Mock(pid=state['sender_pid'], poll=Mock(side_effect=poll))

    sender = Mock(side_effect=launch)
    monkeypatch.setattr(photoshop, 'launch_script', sender)
    renderer = photoshop.PhotoshopRawRenderer(target, hide=True, timeout=0.1)
    return renderer, source, identity, state, sender, windows, quit_mock


def test_windows_sender_exit_one_still_waits_for_result(managed_transport, monkeypatch):
    renderer, source, identity, state, sender, windows, quit_mock = managed_transport
    monkeypatch.setattr(photoshop.sys, 'platform', 'win32')
    state.update(processes=[identity], pending=True, sender_exit_code=1)
    polls = []

    def complete_after_forwarding():
        polls.append(None)
        if len(polls) == 2:
            state['complete']()

    state['on_poll'] = complete_after_forwarding
    frame, _ = renderer.render(source)
    renderer.close()
    assert len(polls) == 2
    assert frame.tolist() == [[[30, 20, 10]]]
    sender.assert_called_once()
    quit_mock.assert_not_called()


def test_macos_sender_nonzero_exit_fails_early(managed_transport, monkeypatch):
    renderer, source, identity, state, sender, windows, quit_mock = managed_transport
    monkeypatch.setattr(photoshop.sys, 'platform', 'darwin')
    state.update(processes=[identity], pending=True, sender_exit_code=1)
    poll = Mock()
    state['on_poll'] = poll
    with pytest.raises(RuntimeError, match='Photoshop script sender failed: exit 1'):
        renderer.render(source)
    renderer.close()
    poll.assert_called_once()
    sender.assert_called_once()
    quit_mock.assert_not_called()


def test_managed_transport_borrows_without_hiding_or_exiting(managed_transport):
    renderer, source, identity, state, sender, windows, quit_mock = managed_transport
    state['processes'] = [identity]
    renderer.render(source)
    renderer.close()
    assert sender.call_args.args[-1] is False
    assert all(call.kwargs.get('show') is None for call in windows.call_args_list)
    assert not renderer._owned
    quit_mock.assert_not_called()


def test_managed_transport_owns_cold_launch_and_exits_it(managed_transport):
    renderer, source, identity, state, sender, windows, quit_mock = managed_transport
    renderer.render(source)
    renderer.close()
    assert renderer._owned
    assert sender.call_args.args[-1] is True
    assert any(call.kwargs.get('show') is False for call in windows.call_args_list)
    quit_mock.assert_called_once_with(renderer._target, identity,
                                     terminate_closed_host=(photoshop.sys.platform == 'win32'))


@pytest.mark.parametrize('borrowed', [False, True])
@pytest.mark.parametrize('replacement', [False, True])
@pytest.mark.parametrize('platform', ['win32', 'darwin'])
def test_host_loss_during_render_fails_without_adopting_or_touching_new_host(
        managed_transport, monkeypatch, borrowed, replacement, platform):
    renderer, source, identity, state, sender, windows, quit_mock = managed_transport
    monkeypatch.setattr(photoshop.sys, 'platform', platform)
    if borrowed:
        state['processes'] = [identity]
    state['pending'] = True
    next_processes = [{'pid': identity['pid'], 'created': identity['created'] + 1}] if replacement else []
    state['on_poll'] = lambda: state.update(processes=next_processes)
    with pytest.raises(RuntimeError, match='host exited or was replaced.*last stage: opening'):
        renderer.render(source)
    renderer.close()
    assert renderer._identity == identity
    sender.assert_called_once()
    quit_mock.assert_not_called()
    assert not any(call.kwargs.get('show') is True for call in windows.call_args_list)
    if borrowed:
        windows.assert_not_called()


@pytest.mark.parametrize('replacement', [False, True])
def test_host_loss_between_frames_does_not_launch_again(managed_transport, replacement):
    renderer, source, identity, state, sender, windows, quit_mock = managed_transport
    renderer.render(source)
    state['processes'] = [{'pid': 456, 'created': 101}] if replacement else []
    with pytest.raises(RuntimeError, match='host exited or was replaced'):
        renderer.render(source)
    renderer.close()
    sender.assert_called_once()
    quit_mock.assert_not_called()


def test_startup_race_borrows_foreign_windows_pid_without_claiming_it(managed_transport, monkeypatch):
    renderer, source, identity, state, sender, windows, quit_mock = managed_transport
    monkeypatch.setattr(photoshop.sys, 'platform', 'win32')
    state['sender_pid'] = 999
    renderer.render(source)
    renderer.close()
    assert renderer._identity == identity
    assert not renderer._owned
    assert all(call.kwargs.get('show') is None for call in windows.call_args_list)
    quit_mock.assert_not_called()


@pytest.mark.parametrize('platform', ['win32', 'darwin'])
def test_borrowed_failure_does_not_reveal_hidden_user_host(managed_transport, monkeypatch, platform):
    renderer, source, identity, state, sender, windows, quit_mock = managed_transport
    monkeypatch.setattr(photoshop.sys, 'platform', platform)
    state.update(processes=[identity], pending=True)
    with pytest.raises(TimeoutError):
        renderer.render(source)
    renderer.close()
    windows.assert_called_once_with(renderer._target)  # Status only, no visibility mutation.
    quit_mock.assert_not_called()


def test_cleanup_rechecks_identity_after_preparing_exit_script(monkeypatch):
    identity = {'pid': 123, 'created': 100.5}
    replacement = {'pid': 123, 'created': 101}
    monkeypatch.setattr(photoshop, 'photoshop_processes',
                        Mock(side_effect=[[identity], [replacement]]))
    sender = Mock(side_effect=AssertionError('must not launch quit into a replacement'))
    monkeypatch.setattr(photoshop, 'launch_script', sender)
    photoshop.quit_photoshop(Path('mock'), identity)
    sender.assert_not_called()


@pytest.mark.parametrize('replacement', [False, True])
@pytest.mark.parametrize('platform', ['win32', 'darwin'])
def test_complete_result_survives_host_exit_without_touching_replacement(
        managed_transport, monkeypatch, replacement, platform):
    renderer, source, identity, state, sender, windows, quit_mock = managed_transport
    monkeypatch.setattr(photoshop.sys, 'platform', platform)
    state['pending'] = True
    next_processes = [{'pid': identity['pid'], 'created': identity['created'] + 1}] if replacement else []

    def finish_and_exit():
        state['complete']()
        state['processes'] = next_processes

    def inspect_visibility(*args, **kwargs):
        if kwargs.get('show') is not None:
            assert state['processes'] == [identity]
        return [{'main': True, 'visible': False}]

    state['on_poll'] = finish_and_exit
    windows.side_effect = inspect_visibility
    frame, _ = renderer.render(source)
    renderer.close()
    assert frame.tolist() == [[[30, 20, 10]]]
    assert state['processes'] == next_processes
    sender.assert_called_once()
    quit_mock.assert_not_called()


@pytest.mark.parametrize('current', [
    [{'pid': 123, 'created': 100.5}, {'pid': 456, 'created': 101}],
    [{'pid': 123, 'created': None}],
])
def test_unidentifiable_existing_host_is_never_launched_into_or_managed(managed_transport, current):
    renderer, source, identity, state, sender, windows, quit_mock = managed_transport
    state['processes'] = current
    with pytest.raises(RuntimeError, match='cannot identify|Cannot establish'):
        renderer.render(source)
    renderer.close()
    sender.assert_not_called()
    windows.assert_not_called()
    quit_mock.assert_not_called()
