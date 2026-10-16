"""UI wiring helpers for Napari ROI menu controls."""


def create_roi_menu(menu_bar, parent, callbacks, action_factory):
    menu = menu_bar.addMenu("ROI Tools")
    definitions = (
        (
            "ellipse_roi_action",
            "Ellipse ROI",
            "Draw the single retained elliptical ROI; area outside is masked",
            "start_ellipse_roi",
        ),
        ("save_roi_action", "Save ROI", "Save the current ellipse ROI", "save_roi"),
        ("reload_roi_action", "Reload ROI", "Reload and display the saved ROI", "reload_roi"),
        ("clear_roi_action", "Clear ROI", "Clear ROI and mask", "clear_roi"),
    )
    actions = {}
    for key, label, status_tip, callback_name in definitions:
        action = action_factory(label, parent)
        action.setStatusTip(status_tip)
        action.triggered.connect(callbacks[callback_name])
        actions[key] = action

    menu.addAction(actions["ellipse_roi_action"])
    menu.addSeparator()
    menu.addAction(actions["save_roi_action"])
    menu.addAction(actions["reload_roi_action"])
    menu.addAction(actions["clear_roi_action"])
    return menu, actions
