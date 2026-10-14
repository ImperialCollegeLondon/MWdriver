"""Application lifecycle wiring for the Napari desktop viewer."""


def run_application(camera_viewer_factory, event_loop):
    camera_viewer = camera_viewer_factory()
    closed = False

    def close_viewer():
        nonlocal closed
        if closed:
            return
        closed = True
        camera_viewer.close()

    def close_event(event):
        try:
            close_viewer()
        finally:
            event.accept()

    camera_viewer.viewer.window._qt_window.closeEvent = close_event
    try:
        event_loop()
    finally:
        close_viewer()
    return camera_viewer
