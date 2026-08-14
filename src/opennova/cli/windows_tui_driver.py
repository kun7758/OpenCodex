"""终端交互层中的Windows终端界面驱动模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

LEFT_ALT_PRESSED = 0x0002
RIGHT_ALT_PRESSED = 0x0001
SHIFT_PRESSED = 0x0010
LEFT_CTRL_PRESSED = 0x0008
RIGHT_CTRL_PRESSED = 0x0004
ENABLE_WINDOW_INPUT = 0x0008
ENABLE_MOUSE_INPUT = 0x0010
ENABLE_EXTENDED_FLAGS = 0x0080
ENABLE_QUICK_EDIT_MODE = 0x0040
ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200

SPECIAL_KEYS = {
    8: "backspace",
    9: "tab",
    13: "enter",
    27: "escape",
    33: "pageup",
    34: "pagedown",
    35: "end",
    36: "home",
    37: "left",
    38: "up",
    39: "right",
    40: "down",
    45: "insert",
    46: "delete",
    **{code: f"f{code - 111}" for code in range(112, 124)},
}


def should_queue_console_key(
    *,
    key: str,
    key_down: bool,
    control_key_state: int,
    virtual_key_code: int,
) -> bool:
    """判断`queue_console_key`条件是否成立。

    参数：
        key: 本次操作使用的`key`。
        key_down: 本次操作使用的`key_down`。
        control_key_state: 本次操作使用的`control_key_state`。
        virtual_key_code: 本次操作使用的`virtual_key_code`。

    返回：
        表示条件是否成立。
    """
    if not key_down or not key or key == "\x00":
        return False

    if control_key_state and virtual_key_code == 0:
        return key.isprintable()

    return True


def format_windows_virtual_key(virtual_key_code: int, control_key_state: int) -> str | None:
    """把`format_windows_virtual_key`整理为稳定、便于展示的文本格式。

    参数：
        virtual_key_code: 本次操作使用的`virtual_key_code`。
        control_key_state: 本次操作使用的`control_key_state`。

    返回：
        `str | None` 类型的处理结果。
    """
    key = SPECIAL_KEYS.get(virtual_key_code)
    if key is None:
        return None

    if key == "tab" and control_key_state & SHIFT_PRESSED:
        return "shift+tab"

    modifiers: list[str] = []
    if control_key_state & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED):
        modifiers.append("ctrl")
    if control_key_state & SHIFT_PRESSED:
        modifiers.append("shift")
    if control_key_state & (LEFT_ALT_PRESSED | RIGHT_ALT_PRESSED):
        modifiers.append("alt")

    return "+".join([*modifiers, key]) if modifiers else key


def build_console_key_debug_record(
    *,
    key: str,
    key_down: bool,
    control_key_state: int,
    virtual_key_code: int,
    virtual_scan_code: int,
) -> dict[str, Any]:
    """根据当前输入和状态构造`build_console_key_debug_record`。

    参数：
        key: 本次操作使用的`key`。
        key_down: 本次操作使用的`key_down`。
        control_key_state: 本次操作使用的`control_key_state`。
        virtual_key_code: 本次操作使用的`virtual_key_code`。
        virtual_scan_code: 本次操作使用的虚拟扫描代码。

    返回：
        供后续逻辑或序列化使用的结构化字典。
    """
    queued = should_queue_console_key(
        key=key,
        key_down=key_down,
        control_key_state=control_key_state,
        virtual_key_code=virtual_key_code,
    )
    return {
        "key": key,
        "repr": repr(key),
        "codepoint": f"U+{ord(key):04X}" if len(key) == 1 and key != "\x00" else None,
        "key_down": key_down,
        "control_key_state": control_key_state,
        "virtual_key_code": virtual_key_code,
        "virtual_scan_code": virtual_scan_code,
        "queued": queued,
        "textual_key": None
        if queued
        else format_windows_virtual_key(virtual_key_code, control_key_state),
    }


def write_console_key_debug_record(path: str | Path, record: dict[str, Any]) -> None:
    """写入 `console_key_debug_record` 对应的数据，并按照当前组件的约定返回结果。

    参数：
        path: 需要读取、检查或写入的路径。
        record: 本次操作使用的记录。

    说明：
        该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
    """
    debug_path = Path(path)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with debug_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_ime_console_input_mode(current_console_mode_in: int, mouse: bool) -> int:
    """根据当前输入和状态构造`build_ime_console_input_mode`。

    参数：
        current_console_mode_in: 本次操作使用的`current_console_mode_in`。
        mouse: 本次操作使用的`mouse`。

    返回：
        `int` 类型的处理结果。
    """
    input_mode = current_console_mode_in
    input_mode &= ~(0x0004 | 0x0002 | 0x0001)
    input_mode |= ENABLE_WINDOW_INPUT | ENABLE_VIRTUAL_TERMINAL_INPUT
    if mouse:
        input_mode |= ENABLE_MOUSE_INPUT | ENABLE_EXTENDED_FLAGS
        input_mode &= ~ENABLE_QUICK_EDIT_MODE
    return input_mode


def get_ime_friendly_windows_driver_class() -> type[Any]:
    """读取 `ime_friendly_windows_driver_class` 对应的数据，不改变当前对象的业务状态。

    返回：
        `type[Any]` 类型的处理结果。
    """
    if sys.platform != "win32":
        raise RuntimeError("The IME-friendly Textual driver is only available on Windows")

    return _build_ime_friendly_windows_driver_class()


def _build_ime_friendly_windows_driver_class() -> type[Any]:
    import asyncio
    import threading
    from asyncio import AbstractEventLoop, run_coroutine_threadsafe

    from textual import constants, events
    from textual._xterm_parser import XTermParser
    from textual.drivers import win32
    from textual.drivers._writer_thread import WriterThread
    from textual.drivers.windows_driver import WindowsDriver
    from textual.geometry import Size

    key_event_type = 0x0001
    window_buffer_size_event_type = 0x0004

    def enable_ime_application_mode(mouse: bool) -> Callable[[], None]:
        terminal_in = sys.__stdin__
        terminal_out = sys.__stdout__

        current_console_mode_in = win32.get_console_mode(terminal_in)
        current_console_mode_out = win32.get_console_mode(terminal_out)

        def restore() -> None:
            win32.set_console_mode(terminal_in, current_console_mode_in)
            win32.set_console_mode(terminal_out, current_console_mode_out)

        win32.set_console_mode(
            terminal_out,
            current_console_mode_out | win32.ENABLE_VIRTUAL_TERMINAL_PROCESSING,
        )

        input_mode = build_ime_console_input_mode(current_console_mode_in, mouse)
        win32.set_console_mode(terminal_in, input_mode)
        return restore

    class IMEFriendlyWindowsEventMonitor(threading.Thread):
        """封装`IMEFriendlyWindowsEventMonitor`相关的状态和操作，使调用方通过稳定接口使用该能力。"""

        def __init__(
            self,
            loop: AbstractEventLoop,
            app: Any,
            exit_event: threading.Event,
            process_event: Callable[[events.Event], None],
        ) -> None:
            self.loop = loop
            self.app = app
            self.exit_event = exit_event
            self.process_event = process_event
            super().__init__(name="opennova-textual-input")

        def run(self) -> None:
            exit_requested = self.exit_event.is_set
            parser = XTermParser(debug=constants.DEBUG)
            debug_path = os.environ.get("OPENNOVA_TUI_INPUT_DEBUG")

            try:
                read_count = win32.wintypes.DWORD(0)
                input_handle = win32.GetStdHandle(win32.STD_INPUT_HANDLE)

                max_events = 1024
                arrtype = win32.INPUT_RECORD * max_events
                input_records = arrtype()
                read_console_input = win32.KERNEL32.ReadConsoleInputW
                queued_keys: list[str] = []

                def flush_queued_keys() -> None:
                    if not queued_keys:
                        return
                    text = "".join(queued_keys).encode("utf-16", "surrogatepass").decode("utf-16")
                    queued_keys.clear()
                    for parsed_event in parser.feed(text):
                        self.process_event(parsed_event)

                while not exit_requested():
                    for parsed_event in parser.tick():
                        self.process_event(parsed_event)

                    if win32.wait_for_handles([input_handle], 100) is None:
                        continue

                    read_console_input(
                        input_handle,
                        win32.byref(input_records),
                        max_events,
                        win32.byref(read_count),
                    )

                    new_size: tuple[int, int] | None = None
                    for input_record in input_records[: read_count.value]:
                        event_type = input_record.EventType

                        if event_type == key_event_type:
                            key_event = input_record.Event.KeyEvent
                            key = key_event.uChar.UnicodeChar
                            control_state = key_event.dwControlKeyState
                            virtual_key_code = key_event.wVirtualKeyCode
                            if debug_path:
                                write_console_key_debug_record(
                                    debug_path,
                                    build_console_key_debug_record(
                                        key=key,
                                        key_down=bool(key_event.bKeyDown),
                                        control_key_state=control_state,
                                        virtual_key_code=virtual_key_code,
                                        virtual_scan_code=key_event.wVirtualScanCode,
                                    ),
                                )

                            if should_queue_console_key(
                                key=key,
                                key_down=bool(key_event.bKeyDown),
                                control_key_state=control_state,
                                virtual_key_code=virtual_key_code,
                            ):
                                queued_keys.append(key)
                                continue

                            if key_event.bKeyDown:
                                special_key = format_windows_virtual_key(
                                    virtual_key_code,
                                    control_state,
                                )
                                if special_key is not None:
                                    flush_queued_keys()
                                    self.process_event(events.Key(special_key, None))

                        elif event_type == window_buffer_size_event_type:
                            size = input_record.Event.WindowBufferSizeEvent.dwSize
                            new_size = (size.X, size.Y)

                    flush_queued_keys()
                    if new_size is not None:
                        self.on_size_change(*new_size)

            except Exception as error:
                self.app.log.error("EVENT MONITOR ERROR", error)

        def on_size_change(self, width: int, height: int) -> None:
            size = Size(width, height)
            event = events.Resize(size, size)
            run_coroutine_threadsafe(self.app._post_message(event), loop=self.loop)

    class IMEFriendlyWindowsDriver(WindowsDriver):
        """封装输入法兼容兼容Windows驱动相关的状态和操作，使调用方通过稳定接口使用该能力。"""

        def start_application_mode(self) -> None:
            loop = asyncio.get_running_loop()

            self._restore_console = enable_ime_application_mode(mouse=self._mouse)

            self._writer_thread = WriterThread(self._file)
            self._writer_thread.start()

            self.write("\x1b[?1049h")  # 启用终端备用屏幕缓冲区。
            self._enable_mouse_support()
            self.write("\x1b[?25l")  # 进入全屏界面后隐藏终端光标。
            self.write("\033[?1004h")  # 启用终端焦点进入和离开事件。
            self.write("\x1b[>1u")  # 启用 Kitty 键盘协议以获得更完整的按键信息。
            self.flush()
            self._enable_bracketed_paste()

            self._event_thread = IMEFriendlyWindowsEventMonitor(
                loop,
                self._app,
                self.exit_event,
                self.process_message,
            )
            self._event_thread.start()

    return IMEFriendlyWindowsDriver
