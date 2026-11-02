"""Entry point for the desktop application and its packaged worker modes."""

import sys
import multiprocessing


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)

    if arguments and arguments[0] == "--mwdriver-phase-check":
        from mwdriver.analysis.check_measured_vs_expected_phase import main as phase_check_main

        result = phase_check_main(arguments[1:])
        return int(result or 0)

    if arguments and arguments[0] == "--mwdriver-phase-analysis":
        from mwdriver.analysis.phase_analysis_new import main as phase_analysis_main

        phase_analysis_main(arguments[1:])
        return 0

    from camera import main as camera_main

    camera_main()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())