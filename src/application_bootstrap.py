"""Application lifecycle wiring for the Napari desktop viewer."""


def run_application(camera_viewer_factory, event_loop):
    camera_viewer = camera_viewer_factory()

    def close_event(event):
        camera_viewer.close()
        event.accept()

    camera_viewer.viewer.window._qt_window.closeEvent = close_event
    event_loop()
    return camera_viewer
