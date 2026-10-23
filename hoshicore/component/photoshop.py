"""Experimental Photoshop/ACR transport; macOS is unverified.

RAW opening uses Photoshop defaults, including adjacent same-name XMP.
The rendered TIFF supplies both pixels and EXIF/ICC.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import psutil
import pyexiv2
import tifffile
from loguru import logger


def resolve_photoshop(path=None) -> Path:
    if sys.platform not in ('win32', 'darwin'):
        raise OSError('This experiment supports Windows and macOS only')
    if path is None:
        raise ValueError('Configure photoshop_path before decoding RAW with Photoshop')
    app = Path(path).expanduser().resolve(strict=True)
    if sys.platform == 'darwin' and app.suffix.lower() != '.app':
        raise ValueError('--photoshop must be a .app bundle on macOS')
    return app


def photoshop_processes(photoshop=None) -> list[dict]:
    app = resolve_photoshop(photoshop)
    executable = app
    if sys.platform == 'darwin':
        with (app / 'Contents/Info.plist').open('rb') as stream:
            executable = app / 'Contents/MacOS' / plistlib.load(stream)['CFBundleExecutable']
    expected = os.path.normcase(str(executable.resolve(strict=True)))
    found = []
    for process in psutil.process_iter(['pid', 'exe', 'create_time']):
        try:
            actual = process.info['exe']
            if actual and os.path.normcase(str(Path(actual).resolve())) == expected:
                found.append({'pid': process.pid, 'created': process.info['create_time']})
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    return found


# Experimental macOS branch: not yet run on a Mac. App visibility, not per-window.
MAC_VISIBILITY = r"""
ObjC.import('AppKit');
function run(argv) {
    var result = [];
    JSON.parse(argv[0]).forEach(function (pid) {
        var app = $.NSRunningApplication.runningApplicationWithProcessIdentifier(pid);
        if (!app || Number(app.processIdentifier) !== pid || app.isTerminated) return;
        if (argv[1] === 'hide') app.hide;
        if (argv[1] === 'show') app.unhide;
        $.NSRunLoop.currentRunLoop.runUntilDate($.NSDate.dateWithTimeIntervalSinceNow(0.05));
        result.push({pid: pid, main: true, visible: !Boolean(app.isHidden), scope: 'application'});
    });
    return JSON.stringify(result);
}
"""


def photoshop_windows(photoshop=None, *, show=None) -> list[dict]:
    pids = {item['pid'] for item in photoshop_processes(photoshop)}
    if not pids:
        return []
    if sys.platform == 'darwin':
        action = 'status' if show is None else ('show' if show else 'hide')
        result = subprocess.run(
            ['/usr/bin/osascript', '-l', 'JavaScript', '-e', MAC_VISIBILITY,
             json.dumps(sorted(pids)), action], capture_output=True,
            text=True, encoding='utf-8', timeout=10)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return json.loads(result.stdout)

    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    windows = []

    @callback_type
    def collect(hwnd, _):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in pids:
            title, kind = ctypes.create_unicode_buffer(1024), ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, title, len(title))
            user32.GetClassNameW(hwnd, kind, len(kind))
            main = kind.value == 'Photoshop'
            if main and show is not None:
                user32.ShowWindow(hwnd, 8 if show else 0)
            visible = bool(user32.IsWindowVisible(hwnd))
            if main or visible:
                windows.append({'pid': pid.value, 'title': title.value, 'main': main,
                                'visible': visible, 'scope': 'window'})
        return True

    if not user32.EnumWindows(collect, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    return windows


def launch_script(app: Path, script: Path, output: Path, timeout: float, hide: bool):
    kwargs = {}
    if sys.platform == 'win32':
        startup = subprocess.STARTUPINFO()
        if hide:
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 0
        kwargs['startupinfo'] = startup
        command = [str(app), '-r', str(script)]
    else:
        # AppleScript's Photoshop terminology is loaded from the chosen .app.
        sender = output / 'run.applescript'
        sender.write_text(
            'on run argv\n'
            '  set jsxFile to (POSIX file (item 2 of argv)) as alias\n'
            f'  using terms from application {json.dumps(str(app), ensure_ascii=False)}\n'
            '    tell application (item 1 of argv)\n'
            f'      with timeout of {max(60, math.ceil(timeout) + 30)} seconds\n'
            '        do javascript jsxFile\n'
            '      end timeout\n'
            '    end tell\n'
            '  end using terms from\n'
            'end run\n', encoding='utf-8')
        command = ['/usr/bin/osascript', str(sender), str(app), str(script)]
    with (output / 'launcher.stderr.log').open('wb') as stderr:
        return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr, **kwargs)


def sender_error(output: Path) -> str:
    path = output / 'launcher.stderr.log'
    return path.read_text(encoding='utf-8', errors='replace')[-4096:].strip() if path.exists() else ''


def to_exif_data(metadata: dict):
    """One-way adapter; use when imported from the repository root."""
    from .exif import ExifData
    return ExifData(exif=dict(metadata['exif']), colorprofile=metadata['icc_profile'])


# ES3-compatible: CC2018 ExtendScript does not reliably provide JSON.stringify.
JSX = r"""
#target photoshop
(function () {
    function quote(s) {
        return '"' + String(s).replace(/[\\"\u0000-\u001f\u2028\u2029]/g, function (c) {
            if (c === '\\') return '\\\\';
            if (c === '"') return '\\"';
            return '\\u' + ('0000' + c.charCodeAt(0).toString(16)).slice(-4);
        }) + '"';
    }
    function json(v) {
        if (v === null || v === undefined) return 'null';
        if (typeof v === 'string') return quote(v);
        if (typeof v === 'number' || typeof v === 'boolean') return String(v);
        var parts = [], k;
        if (v instanceof Array) {
            for (k = 0; k < v.length; k++) parts.push(json(v[k]));
            return '[' + parts.join(',') + ']';
        }
        for (k in v) if (v.hasOwnProperty(k)) parts.push(quote(k) + ':' + json(v[k]));
        return '{' + parts.join(',') + '}';
    }
    function write(path, text) {
        var f = new File(path);
        f.encoding = 'UTF8';
        if (!f.open('w')) throw new Error('Cannot write ' + path + ': ' + f.error);
        try {
            if (!f.write(text)) throw new Error('Write failed: ' + path);
        } finally { f.close(); }
    }
    var oldDialogs = app.displayDialogs;
    var doc = null, result, start = new Date().getTime(), stage = 'opening';
    function phase(name) { stage = name; write(cfg.progress, name); }
    try {
        app.displayDialogs = DialogModes.NO;
        phase('opening');
        // No CameraRAWOpenOptions override: let ACR use its normal image settings.
        doc = app.open(new File(cfg.input));
        var opened = new Date().getTime();
        phase('metadata');
        result = {
            ok: true,
            photoshop_version: app.version,
            document_name: doc.name,
            width: doc.width.as('px'),
            height: doc.height.as('px'),
            mode: String(doc.mode),
            bits_per_channel: String(doc.bitsPerChannel),
            color_profile: doc.colorProfileName,
            photoshop_exif: doc.info.exif,
            open_seconds: (opened - start) / 1000
        };
        write(cfg.xmp, doc.xmpMetadata.rawData);
        var options = new TiffSaveOptions();
        options.imageCompression = TIFFEncoding.NONE;
        options.byteOrder = ByteOrder.IBM;
        options.layers = false;
        options.alphaChannels = false;
        options.embedColorProfile = true;
        phase('exporting');
        doc.saveAs(new File(cfg.tiff), options, true, Extension.LOWERCASE);
        result.export_seconds = (new Date().getTime() - opened) / 1000;
    } catch (e) {
        result = {ok: false, error: String(e), line: e.line, number: e.number, stage: stage};
    } finally {
        if (doc !== null) {
            try { doc.close(SaveOptions.DONOTSAVECHANGES); }
            catch (e) { result.ok = false; result.error = String(e); result.stage = 'closing'; }
        }
        app.displayDialogs = oldDialogs;
    }
    write(cfg.result + '.tmp', json(result));
    if (!(new File(cfg.result + '.tmp')).rename('result.json')) {
        throw new Error('Cannot publish result.json');
    }
})();
"""


def _render_raw(
    raw_path: str | Path,
    *,
    photoshop: str | Path | None = None,
    output_dir: str | Path | None = None,
    timeout: float = 180,
    hide: bool = False,
    _host=None,
) -> tuple[np.ndarray, dict]:
    """Return Photoshop TIFF pixels (RGB order) and metadata, retaining artifacts.

    Metadata contains full XMP text and ICC bytes, plus the artifact paths.
    A timeout stops waiting; it does not kill Photoshop or stop an active script.
    """
    source = Path(raw_path).resolve(strict=True)
    target = resolve_photoshop(photoshop)
    if not source.is_file():
        raise ValueError('Input must be a file')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    if output_dir is None:
        output = Path(tempfile.mkdtemp(prefix='hnw-photoshop-')) / 'render'
    else:
        output = Path(output_dir).resolve()
    # A fresh directory prevents overwriting an earlier export or input sidecar.
    output.mkdir(parents=True, exist_ok=False)
    paths = {name: output / filename for name, filename in {
        'tiff': 'rendered.tif', 'xmp': 'rendered.xmp', 'result': 'result.json',
        'progress': 'stage.txt'
    }.items()}
    config = {'input': source.as_posix(), **{
        name: path.as_posix() for name, path in paths.items()
    }}
    script = output / 'render.jsx'
    # #target must be the first directive, before the injected configuration.
    script.write_text('#target photoshop\nvar cfg = ' + json.dumps(config) + ';\n'
                      + JSX.replace('#target photoshop\n', ''), encoding='utf-8')
    logger.debug('Photoshop artifacts: {}', output)
    before = photoshop_processes(target)
    if _host is not None:
        hide = _host._prepare_host(before)
    started = time.perf_counter()
    # Photoshop's script launcher also works when that Photoshop is already open.
    if hide and _host is None:
        photoshop_windows(target, show=False)
    process = launch_script(target, script, output, timeout, hide)
    while True:
        if _host is not None:
            current = photoshop_processes(target)
            # A complete result remains usable if PS exits immediately afterwards.
            if paths['result'].exists() and _host._identity is not None:
                hide = _host.hide and _host._owned and current == [_host._identity]
                break
            try:
                _host._observe_host(current, process)
            except RuntimeError as error:
                stage = (paths['progress'].read_text(encoding='utf-8-sig')
                         if paths['progress'].exists() else 'waiting for JSX to start')
                raise RuntimeError(f'{error}; last stage: {stage}; artifacts: {output}') from error
            hide = _host.hide and _host._owned and current == [_host._identity]
        if paths['result'].exists():
            break
        if hide:
            photoshop_windows(target, show=False)
        exit_code = process.poll()
        # CC2018 can return 1 after forwarding JSX to an existing instance.
        # Only fail early on a Windows crash status, not a normal launcher exit.
        failed = exit_code is not None and (
            (sys.platform == 'darwin' and exit_code != 0) or
            (sys.platform == 'win32' and exit_code & 0x80000000))
        if failed:
            raise RuntimeError(f'Photoshop script sender failed: exit {exit_code}; '
                               f'{sender_error(output)}; artifacts: {output}')
        if time.perf_counter() - started >= timeout:
            windows = photoshop_windows(target)
            stage = (paths['progress'].read_text(encoding='utf-8-sig')
                     if paths['progress'].exists() else 'waiting for JSX to start')
            raise TimeoutError(
                f'No Photoshop result after {timeout:g}s (launcher exit={process.poll()}). '
                f'Last stage: {stage}. Visibility: {windows}. '
                f'Sender error: {sender_error(output)}. Artifacts: {output}. '
                'Photoshop was left running; the script may still finish.')
        time.sleep(0.5 if sys.platform == 'darwin' else 0.25)
    metadata = json.loads(paths['result'].read_text(encoding='utf-8-sig'))
    if not metadata['ok']:
        raise RuntimeError(f"Photoshop: {metadata['error']} "
                           f"(stage={metadata.get('stage')}, code={metadata.get('number')}, "
                           f"JSX line={metadata.get('line')}); artifacts: {output}")
    if _host is not None:
        hide = hide and _host._can_control_host()
    windows = photoshop_windows(target, show=False if hide else None)
    after = photoshop_processes(target)
    metadata['host_platform'] = sys.platform
    metadata['photoshop_target'] = str(target)
    metadata['processes_before'] = before
    metadata['processes_after'] = after
    metadata['reused_process'] = bool({(p['pid'], p['created']) for p in before} &
                                      {(p['pid'], p['created']) for p in after})
    metadata['launcher_pid'] = process.pid
    metadata['windows_after'] = windows
    metadata['hide_requested'] = hide
    if hide and (not any(w['main'] for w in windows)
                 or any(w['main'] and w['visible'] for w in windows)):
        logger.warning('Could not verify Photoshop visibility is hidden')
    extras = [w for w in windows if w['visible'] and not w['main']]
    if hide and extras:
        logger.warning('Other Photoshop windows remain visible: {}', extras)
    with tifffile.TiffFile(paths['tiff']) as image:
        pixels = image.asarray()
        page = image.pages[0]
        icc_tag = page.tags.get(34675)
        icc = bytes(icc_tag.value) if icc_tag is not None else b''
        metadata['tiff_photometric'] = page.photometric.name
        metadata['tiff_planarconfig'] = page.planarconfig.name
    # Localized doc.info.exif pairs are diagnostic only. Read canonical Exiv2
    # keys from the rendered TIFF, matching ExifData/read_exif_data conventions.
    # ImageData also handles Unicode paths on older Windows Exiv2 builds.
    with pyexiv2.ImageData(paths['tiff'].read_bytes()) as image:
        metadata['exif'] = image.read_exif()
        if image.read_icc() != icc:
            raise RuntimeError('TIFF readers disagree about the rendered ICC profile')
    if metadata['tiff_planarconfig'] == 'SEPARATE' and pixels.ndim == 3:
        pixels = np.moveaxis(pixels, 0, -1)
    pixels = np.ascontiguousarray(pixels)
    metadata['shape'] = list(pixels.shape)
    metadata['dtype'] = str(pixels.dtype)
    metadata['channel_order'] = metadata['tiff_photometric']
    metadata['total_seconds'] = time.perf_counter() - started
    metadata['source'] = str(source)
    sidecar = source.with_suffix('.xmp')
    if not sidecar.exists():
        sidecar = source.with_suffix('.XMP')
    metadata['adjacent_xmp'] = str(sidecar) if sidecar.exists() else None
    metadata['artifacts'] = {name: str(path) for name, path in paths.items()}
    metadata['icc_bytes'] = len(icc)
    if icc:
        profile = output / 'rendered.icc'
        profile.write_bytes(icc)
        metadata['artifacts']['icc'] = str(profile)
    (output / 'metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata['xmp'] = paths['xmp'].read_text(encoding='utf-8-sig')
    metadata['icc_profile'] = icc
    return pixels, metadata


def render_raw(raw_path: str | Path, *, photoshop: str | Path | None = None,
               output_dir: str | Path | None = None, timeout: float = 180,
               hide: bool = False, _host=None) -> tuple[np.ndarray, dict]:
    """Return RGB/EXIF/ICC; failed managed runs preserve borrowed host visibility."""
    try:
        return _render_raw(raw_path, photoshop=photoshop, output_dir=output_dir,
                           timeout=timeout, hide=hide, _host=_host)
    except Exception:
        if (hide and sys.platform in ('win32', 'darwin') and
                (_host is None or _host._can_control_host())):
            try:
                photoshop_windows(photoshop, show=True)
            except Exception as error:
                logger.warning('Could not reveal Photoshop after failure: {}', error)
        raise


def quit_photoshop(target: Path, identity: dict, timeout: float = 10) -> None:
    """Request a normal exit only for the recorded process, with no open documents.

    Never kill a process or launch a replacement if the recorded one has exited.
    """
    current = photoshop_processes(target)
    if identity not in current:
        return
    if current != [identity]:
        raise RuntimeError('Photoshop process identity changed; refusing to request exit')
    with tempfile.TemporaryDirectory(prefix='hnw-photoshop-exit-') as folder:
        output = Path(folder)
        status = output / 'status.txt'
        script = output / 'quit.jsx'
        script.write_text(
            '#target photoshop\n'
            '(function () {\n'
            f'  var f = new File({json.dumps(status.as_posix())});\n'
            '  function report(s) { f.encoding = "UTF8"; f.open("w"); f.write(s); f.close(); }\n'
            '  if (app.documents.length) { report("documents-open"); return; }\n'
            '  report("quitting");\n'
            '  try { executeAction(charIDToTypeID("quit"), new ActionDescriptor(), DialogModes.NO); }\n'
            '  catch (e) { report("error: " + String(e)); }\n'
            '})();\n', encoding='utf-8')
        # Recheck after writing the script, too: never deliberately launch a
        # replacement if the owned host has exited while preparing cleanup.
        if photoshop_processes(target) != [identity]:
            return
        sender = launch_script(target, script, output, timeout, True)
        deadline = time.monotonic() + timeout
        try:
            while identity in photoshop_processes(target):
                if status.exists():
                    message = status.read_text(encoding='utf-8-sig')
                    if message == 'documents-open' or message.startswith('error:'):
                        raise RuntimeError(f'Photoshop left running: {message}')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Photoshop did not exit; it was left running')
                time.sleep(0.1)
        finally:
            # The sender can be a short-lived forwarding process, not the PS host.
            if sender.poll() is not None:
                sender.wait()


class PhotoshopRawRenderer:
    """One service's lazy Photoshop session. Calls must be serialized.

    Existing hosts are borrowed without changing visibility or requesting exit.
    Hosts started by this session are hidden if requested and exited on close.
    Failed renders retain artifacts. A timed-out JSX may still be executing;
    an owned host is revealed and left running, and borrowed visibility is kept.
    """

    def __init__(self, path=None, *, timeout=180, hide=True):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Photoshop timeout must be finite and positive')
        self.path = path
        self.timeout = timeout
        self.hide = hide
        self._target = None
        self._identity = None
        self._owned = False
        self._started = False
        self._timed_out = False
        self._closed = False

    def _prepare_host(self, current):
        if self._identity is not None:
            self._observe_host(current, None)
            return self.hide and self._owned
        if self._started:
            raise RuntimeError('Photoshop host was not identified; refusing to restart it')
        if len(current) > 1:
            raise RuntimeError('Multiple Photoshop processes found; cannot identify the host')
        if current and current[0].get('created') is None:
            raise RuntimeError('Cannot establish Photoshop host identity; leaving it running')
        self._started = True
        if current:
            self._identity = current[0]
            return False
        return self.hide

    def _observe_host(self, current, sender):
        if self._identity is not None:
            if self._identity not in current:
                raise RuntimeError(f'Photoshop host exited or was replaced: {self._identity}')
            return
        if not current:
            return  # The first cold launch may not be visible to psutil yet.
        if len(current) != 1 or current[0].get('created') is None:
            raise RuntimeError('Cannot establish Photoshop host identity; leaving it running')
        self._identity = current[0]
        # On Windows, the cold launcher is the PS host. A different PID means
        # another instance appeared during startup; borrow it conservatively.
        self._owned = sys.platform != 'win32' or self._identity['pid'] == sender.pid

    def _can_control_host(self):
        return self._owned and photoshop_processes(self._target) == [self._identity]

    def render(self, path):
        if self._closed:
            raise RuntimeError('Photoshop renderer is closed')
        if self._timed_out:
            raise RuntimeError('Previous Photoshop request timed out; refusing another request')
        if self._target is None:
            self._target = resolve_photoshop(self.path)
        folder = Path(tempfile.mkdtemp(prefix='hnw-photoshop-'))
        try:
            pixels, metadata = render_raw(path, photoshop=self._target,
                                          output_dir=folder / 'render',
                                          timeout=self.timeout, hide=self.hide, _host=self)
            if self._identity is None:
                raise RuntimeError('Photoshop host was not identified; leaving it running')
            if metadata['tiff_photometric'] != 'RGB' or pixels.ndim != 3 or pixels.shape[2] != 3:
                raise ValueError('Photoshop RAW output must be a three-channel RGB image')
            result = np.ascontiguousarray(pixels[:, :, ::-1]), to_exif_data(metadata)
        except TimeoutError:
            self._timed_out = True
            raise
        except Exception:
            logger.error('Photoshop render failed; artifacts retained at {}', folder)
            raise
        else:
            shutil.rmtree(folder)
            return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        if not self._owned:
            return
        if self._timed_out:
            logger.warning('Photoshop request timed out; host left running for diagnosis')
            return
        if not self._can_control_host():
            logger.debug('Owned Photoshop host exited or changed; skipping exit request')
            return
        try:
            quit_photoshop(self._target, self._identity)
        except Exception:
            if self.hide and self._can_control_host():
                try:
                    photoshop_windows(self._target, show=True)
                except Exception as error:
                    logger.warning('Could not reveal Photoshop after exit failure: {}', error)
            raise
