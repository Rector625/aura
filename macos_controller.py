import json
import math
import sys
import time
import threading
from typing import Optional, Tuple

import asyncio
import websockets
from pynput import keyboard

# macOS Native Quartz & Accessibility APIs via PyObjC
try:
    import Quartz
    import ApplicationServices
    MACOS_NATIVE_AVAILABLE = True
except ImportError:
    MACOS_NATIVE_AVAILABLE = False


class MacOSController:
    """
    Native macOS OS Controller for Aura:
    - Hijacks system cursor with CoreGraphics/Quartz (zero latency).
    - Executes click events based on acoustic triggers.
    - Inspects macOS Accessibility tree (AXUIElement) for Dynamic Magnetic Friction over buttons.
    - Global Hardware/Software Kill Switch (Spacebar toggle).
    """

    def __init__(self, ws_url: str = "ws://127.0.0.1:8765"):
        self.ws_url = ws_url
        self.is_enabled: bool = True           # Master hardware killswitch state
        self.is_paused_by_failsafe: bool = False
        self.last_click_state: bool = False

        # Sticky friction parameters
        self.enable_magnetic_friction: bool = True
        self.current_friction_multiplier: float = 1.0  # 1.0 = normal, 0.35 = sticky over button

        # Setup keyboard killswitch listener
        self._setup_killswitch()

    def _setup_killswitch(self):
        """Global key listener for Spacebar kill switch."""
        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    self.is_enabled = not self.is_enabled
                    state_str = "ENABLED" if self.is_enabled else "KILLED / DISABLED"
                    print(f"\n[Aura Killswitch] OS Mouse Control {state_str} via Spacebar!")
            except Exception:
                pass

        self.key_listener = keyboard.Listener(on_press=on_press)
        self.key_listener.daemon = True
        self.key_listener.start()

    def _check_accessibility_element(self, x: float, y: float) -> bool:
        """
        Queries macOS Accessibility API to inspect whether cursor is over a clickable element.
        Returns True if over a button, link, tab, or menu item.
        """
        if not MACOS_NATIVE_AVAILABLE:
            return False

        try:
            system_wide = ApplicationServices.AXUIElementCreateSystemWide()
            err, element = ApplicationServices.AXUIElementCopyElementAtPosition(system_wide, float(x), float(y), None)
            if err == 0 and element is not None:
                err_role, role = ApplicationServices.AXUIElementCopyAttributeValue(
                    element, ApplicationServices.kAXRoleAttribute, None
                )
                if err_role == 0 and role in [
                    ApplicationServices.kAXButtonRole,
                    ApplicationServices.kAXPopUpButtonRole,
                    ApplicationServices.kAXMenuButtonRole,
                    ApplicationServices.kAXRadioButtonRole,
                    ApplicationServices.kAXCheckBoxRole,
                    ApplicationServices.kAXTabRole,
                    ApplicationServices.kAXLinkRole,
                ]:
                    return True
        except Exception:
            pass
        return False

    def move_mouse(self, x: float, y: float):
        """Zero-latency cursor positioning via CoreGraphics."""
        if not MACOS_NATIVE_AVAILABLE or not self.is_enabled or self.is_paused_by_failsafe:
            return

        # Check for sticky button friction
        if self.enable_magnetic_friction and self._check_accessibility_element(x, y):
            self.current_friction_multiplier = 0.35  # Tactile "heavy" button friction
        else:
            self.current_friction_multiplier = 1.0

        # Move cursor
        point = Quartz.CGPoint(x=float(x), y=float(y))
        Quartz.CGWarpMouseCursorPosition(point)

    def click_mouse(self, x: float, y: float):
        """Simulate hardware mouse click at (x, y)."""
        if not MACOS_NATIVE_AVAILABLE or not self.is_enabled or self.is_paused_by_failsafe:
            return

        point = Quartz.CGPoint(x=float(x), y=float(y))
        # Mouse Down
        down_event = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventLeftMouseDown, point, Quartz.kCGMouseButtonLeft
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, down_event)

        # Mouse Up
        time.sleep(0.015)  # 15ms click dwell
        up_event = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventLeftMouseUp, point, Quartz.kCGMouseButtonLeft
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, up_event)
        print(f"[Aura OS Controller] Click dispatched at ({x:.1f}, {y:.1f})")

    async def run(self):
        """Main loop listening to WebSocket telemetry stream."""
        print(f"[Aura OS Controller] Connecting to telemetry stream at {self.ws_url}...")
        async with websockets.connect(self.ws_url) as websocket:
            print("[Aura OS Controller] Connected! Press [SPACEBAR] at any time for emergency Kill Switch.")

            prev_x, prev_y = None, None

            async for message in websocket:
                data = json.loads(message)
                if data.get("type") != "telemetry":
                    continue

                raw_target_x = data.get("x", 0.0)
                raw_target_y = data.get("y", 0.0)
                click_intent = data.get("click", False)
                is_blink_locked = data.get("is_blink_locked", False)
                abort_triggered = data.get("abort_triggered", False)

                # Update failsafe pause state
                self.is_paused_by_failsafe = is_blink_locked or abort_triggered

                # If no face is detected, do not seize the OS mouse pointer
                if not data.get("face_detected", False):
                    prev_x, prev_y = None, None
                    continue

                if prev_x is None:
                    prev_x, prev_y = raw_target_x, raw_target_y

                # Apply magnetic friction multiplier to movement delta
                delta_x = (raw_target_x - prev_x) * self.current_friction_multiplier
                delta_y = (raw_target_y - prev_y) * self.current_friction_multiplier
                current_x = prev_x + delta_x
                current_y = prev_y + delta_y
                prev_x, prev_y = current_x, current_y

                # Execute movement
                self.move_mouse(current_x, current_y)

                # Execute click on rising edge of acoustic intent
                if click_intent and not self.last_click_state:
                    self.click_mouse(current_x, current_y)

                self.last_click_state = click_intent


if __name__ == "__main__":
    controller = MacOSController()
    try:
        asyncio.run(controller.run())
    except KeyboardInterrupt:
        print("\n[Aura OS Controller] Exited cleanly.")
