"""Napari layer, preview and phase-analysis view orchestration."""

from qtpy.QtCore import QTimer

from mwdriver.domain.layer_visibility import build_layer_visibility_plan
from mwdriver.ui.phase_analysis_controls import (
    build_analysis_layer_presentation,
    build_phase_analysis_result_options,
    choose_analysis_layer_operation,
    create_phase_analysis_controls,
)
from mwdriver.domain.stream_policy import get_camera_stream_mode


class CameraViewController:
    def on_active_layer_changed(self, event=None):
        try:
            active_layer = self.viewer.layers.selection.active
            analysis_record = next(
                (
                    record
                    for record in self.phase_analysis_layer_records
                    if record["layer"] is active_layer
                ),
                None,
            )

            if analysis_record is not None:
                self.phase_analysis_layer = active_layer
                self.phase_analysis_results = analysis_record["results"]
                self.phase_analysis_source_stack_name = analysis_record["source_name"]
                self.phase_analysis_view_active = True
                self.create_phase_analysis_controls(
                    selected_key=analysis_record["selected_key"]
                )
                self.apply_layer_visibility_plan(
                    "analysis",
                    selected_layer=active_layer,
                )
                if self.view_phase_analysis_button is not None:
                    self.view_phase_analysis_button.setText("Return to Camera Preview")
            elif active_layer is self.layer:
                self.show_camera_preview()
            elif active_layer is self.roi_layer:
                self.apply_layer_visibility_plan("roi")
            elif active_layer in self.phase_capture_time_series_layers:
                self.apply_layer_visibility_plan(
                    "time_series",
                    selected_layer=active_layer,
                )

            self.sync_stream_with_active_layer()

        except Exception as e:
            print("Active layer change handling warning:", e)

        QTimer.singleShot(100, self.simplify_layer_controls)
        QTimer.singleShot(150, self.simplify_napari_interface)
        QTimer.singleShot(250, self.place_camera_adjust_controls_in_layer_controls)
        QTimer.singleShot(300, self.place_camera_status_controls_in_layer_controls)

    def on_phase_analysis_layer_removed(self, event):
        self.phase_analysis_layer_records = [
            record
            for record in self.phase_analysis_layer_records
            if record["layer"] is not event.value
        ]
        if event.value is self.phase_analysis_layer:
            self.phase_analysis_layer = None

    def apply_layer_visibility_plan(self, mode, selected_layer=None):
        plan = build_layer_visibility_plan(
            mode,
            [record["layer"] for record in self.phase_analysis_layer_records],
            self.phase_capture_time_series_layers,
            selected_layer=selected_layer,
        )
        self.layer.visible = plan.camera_visible
        if self.roi_layer is not None:
            self.roi_layer.visible = True
            roi_index = self.viewer.layers.index(self.roi_layer)
            top_index = len(self.viewer.layers) - 1
            if roi_index != top_index:
                self.viewer.layers.move(roi_index, top_index)
        for record, visible in zip(
            self.phase_analysis_layer_records,
            plan.analysis_layers_visible,
        ):
            record["layer"].visible = visible
        for layer, visible in zip(
            self.phase_capture_time_series_layers,
            plan.time_series_layers_visible,
        ):
            layer.visible = visible

    def sync_stream_with_active_layer(self):
        stream_mode = get_camera_stream_mode(
            streaming_allowed=self.streaming_allowed,
            preview_layer_visible=(
                self.layer is not None and self.layer.visible
            ),
            camera_mode=self.current_camera_mode,
            phase_capture_active=self.phase_capture_active,
            recalibration_active=self.recalibration_active,
        )
        if stream_mode != "stopped":
            self.start_stream()
        else:
            self.stop_stream()

    def on_camera_preview_visibility_changed(self, event=None):
        self.sync_stream_with_active_layer()


    def allow_streaming(self):
        self.streaming_allowed = True
        self.update_stream_menu_actions()
        self.sync_stream_with_active_layer()

    def disallow_streaming(self):
        self.streaming_allowed = False
        self.update_stream_menu_actions()
        self.sync_stream_with_active_layer()

    def create_phase_analysis_controls(self, selected_key=None):
        result_options = build_phase_analysis_result_options(
            self.phase_analysis_results
        )

        if self.phase_analysis_controls_widget is None:
            (
                self.phase_analysis_controls_widget,
                self.phase_analysis_title_label,
                self.phase_analysis_selector,
                self.view_phase_analysis_button,
            ) = create_phase_analysis_controls(
                self.on_phase_analysis_selection_changed,
                self.toggle_phase_analysis_view,
            )

        self.phase_analysis_title_label.setText(
            f"Review: {self.phase_analysis_source_stack_name or 'analysis'}"
        )
        current_key = selected_key or self.phase_analysis_selector.currentData()
        self.phase_analysis_selector.blockSignals(True)
        self.phase_analysis_selector.clear()
        for label, key in result_options:
            self.phase_analysis_selector.addItem(label, key)

        current_index = self.phase_analysis_selector.findData(current_key)
        if current_index < 0 and self.phase_analysis_selector.count() > 0:
            current_index = 0
        if current_index >= 0:
            self.phase_analysis_selector.setCurrentIndex(current_index)
        self.phase_analysis_selector.blockSignals(False)
        self.view_phase_analysis_button.setEnabled(True)

        self.place_phase_analysis_controls_in_layer_controls()

    def on_phase_analysis_selection_changed(self, index):
        if index >= 0 and self.phase_analysis_view_active:
            self.show_selected_phase_analysis_result()

    def show_selected_phase_analysis_result(self, create_new_layer=False):
        result_key = self.phase_analysis_selector.currentData()
        if result_key not in self.phase_analysis_results:
            return

        result_data = self.phase_analysis_results[result_key]
        result_name = self.phase_analysis_selector.currentText()
        source_name = self.phase_analysis_source_stack_name or "unknown stack"
        presentation = build_analysis_layer_presentation(
            result_key,
            result_data,
            source_name,
            result_name,
        )
        layer_name = presentation["name"]
        is_rgb = presentation["is_rgb"]
        colormap = presentation["colormap"]
        contrast_limits = presentation["contrast_limits"]

        current_record = next(
            (
                record
                for record in self.phase_analysis_layer_records
                if record["layer"] is self.phase_analysis_layer
            ),
            None,
        )
        layer_is_rgb = bool(getattr(self.phase_analysis_layer, "rgb", False))
        layer_operation = choose_analysis_layer_operation(
            create_new_layer=create_new_layer,
            has_current_layer=self.phase_analysis_layer is not None,
            has_current_record=current_record is not None,
            current_layer_is_rgb=layer_is_rgb,
            result_is_rgb=is_rgb,
        )

        if layer_operation != "update":
            if layer_operation == "replace":
                try:
                    self.viewer.layers.remove(self.phase_analysis_layer)
                except ValueError:
                    pass
            if is_rgb:
                analysis_layer = self.viewer.add_image(
                    result_data,
                    name=layer_name,
                    rgb=True,
                )
            else:
                analysis_layer = self.viewer.add_image(
                    result_data,
                    name=layer_name,
                    colormap=colormap,
                    contrast_limits=contrast_limits,
                )
            self.phase_analysis_layer = analysis_layer
            current_record = {
                "layer": analysis_layer,
                "results": self.phase_analysis_results,
                "source_name": source_name,
                "selected_key": result_key,
            }
            self.phase_analysis_layer_records.append(current_record)
        else:
            self.phase_analysis_layer.data = result_data
            self.phase_analysis_layer.name = layer_name
            if not is_rgb:
                self.phase_analysis_layer.colormap = colormap
                if contrast_limits is not None:
                    self.phase_analysis_layer.contrast_limits = contrast_limits
            current_record["selected_key"] = result_key

        self.apply_layer_visibility_plan(
            "analysis",
            selected_layer=self.phase_analysis_layer,
        )
        self.viewer.layers.selection.active = self.phase_analysis_layer
        self.viewer.reset_view()

    def show_camera_preview(self):
        self.apply_layer_visibility_plan("preview")
        self.phase_analysis_view_active = False
        if self.view_phase_analysis_button is not None:
            self.view_phase_analysis_button.setText("View Analysis Results")
        if self.viewer.layers.selection.active is not self.layer:
            self.viewer.layers.selection.active = self.layer
        self.viewer.reset_view()

    def toggle_phase_analysis_view(self):
        if not self.phase_analysis_results:
            self.status_label.setText("Status: No phase analysis results loaded")
            return

        if self.phase_analysis_view_active:
            self.show_camera_preview()
            self.status_label.setText("Status: Returned to camera preview")
            return

        self.phase_analysis_view_active = True
        self.show_selected_phase_analysis_result()
        self.view_phase_analysis_button.setText("Return to Camera Preview")
        self.status_label.setText(
            f"Status: Viewing {self.phase_analysis_selector.currentText()}"
        )
