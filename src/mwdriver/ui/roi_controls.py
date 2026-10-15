"""UI wiring helpers for Napari ROI menu controls."""


def create_roi_menu(menu_bar, parent, callbacks, action_factory):
    menu = menu_bar.addMenu("ROI Tools")
    definitions = (
        ("rect_roi_action", "Rect ROI", "Draw a rectangular ROI", "start_rectangle_roi"),
        (
            "circle_roi_action",
            "Circle ROI",
            "Draw a circular / elliptical ROI",
            "start_circle_roi",
        ),
        (
            "mask_outside_action",
            "Mask Outside Area",
            "Keep the inside of ROI and mask the outside",
            "mask_outside_area",
        ),
        (
            "mask_inside_action",
            "Mask Inside Area",
            "Keep the outside of ROI and mask the inside",
            "mask_inside_area",
        ),
        ("save_roi_action", "Save ROI", "Save ROI parameters and mask", "save_roi"),
        ("reload_roi_action", "Reload ROI", "Reload and display the saved ROI", "reload_roi"),
        ("clear_roi_action", "Clear ROI", "Clear ROI and mask", "clear_roi"),
    )
    actions = {}
    for key, label, status_tip, callback_name in definitions:
        action = action_factory(label, parent)
        action.setStatusTip(status_tip)
        action.triggered.connect(callbacks[callback_name])
        actions[key] = action

    menu.addAction(actions["rect_roi_action"])
    menu.addAction(actions["circle_roi_action"])
    menu.addSeparator()
    menu.addAction(actions["mask_outside_action"])
    menu.addAction(actions["mask_inside_action"])
    menu.addSeparator()
    menu.addAction(actions["save_roi_action"])
    menu.addAction(actions["reload_roi_action"])
    menu.addAction(actions["clear_roi_action"])
    return menu, actions
