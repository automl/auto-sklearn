import time

from autosklearn.util.stopwatch import StopWatch


def test_stopwatch_overhead() -> None:
    wall_start = time.time()
    cpu_start = time.process_time()

    watch = StopWatch()
    for i in range(1, 1000):
        watch.start("task_%d" % i)
        watch.stop("task_%d" % i)

    cpu_end = time.process_time()
    wall_end = time.time()

    wall_duration = wall_end - wall_start
    cpu_duration = cpu_end - cpu_start

    cpu_overhead = cpu_duration - watch.total_cpu()
    wall_overhead = wall_duration - watch.total_wall()

    assert cpu_overhead < 1
    assert wall_overhead < 1
    assert watch.total_cpu() < 2 * watch.total_wall()


def test_contextmanager() -> None:
    watch = StopWatch()

    with watch.time("task"):
        assert watch["task"].started()

    assert "task" in watch
    assert watch["task"].finished()

    assert watch["task"].cpu_duration is not None
    assert watch["task"].wall_duration is not None
