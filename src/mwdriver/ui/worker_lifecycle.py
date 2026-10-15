"""Small ordered lifecycle operations for Qt worker threads."""


def stop_worker_thread(worker, thread):
    if worker is not None:
        worker.stop()
    if thread is not None:
        thread.quit()
        thread.wait()