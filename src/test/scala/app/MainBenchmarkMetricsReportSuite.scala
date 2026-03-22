package app

import munit.FunSuite
import observability._

import java.nio.file.Path

class MainBenchmarkMetricsReportSuite extends FunSuite:
  private def sampleRun(topStages: Vector[TopStage] = Vector(TopStage("matMul", 120.0, 60.0))): RunMetrics =
    RunMetrics(
      runId = "benchmark-abc",
      timestampEpochMs = 1L,
      mode = "benchmark",
      runLabel = Some("demo"),
      configFingerprint = "fp",
      dataset = "data/corpus/example-corpus.txt",
      requestedBackend = "all",
      precision = "all",
      throughputExPerSec = 1234.56,
      trainLoss = None,
      valLoss = None,
      valPerplexity = None,
      epochs = 0,
      domainValLosses = Vector.empty,
      retentionByDomainPct = Vector.empty,
      profile = RunProfile(
        totalSeconds = 2.5,
        epochSecondsTotal = 1.4,
        epochSecondsAvg = 0.35,
        batchMillisAvg = 0.0,
        batchMillisP95 = 0.0,
        backendProfileSummary = "matMul: calls=10 timeMs=120.00",
        topStages = topStages
      ),
      gpu = GpuHotPathMetrics(
        requestedBackend = "all",
        effectiveBackend = "gpu",
        diagnostics = "Metal backend (device=Apple M1, precision=fp32)",
        enabledOps = Vector("batchMatMul"),
        profileSummary = "matMul: calls=10 timeMs=120.00",
        fallbackDetected = false
      ),
      platform = PlatformInfo("macOS", "14.4", "aarch64", "21.0.6", "Apple M1"),
      fallbackOps = Vector.empty,
      memoryStart = MemoryMetrics(1L, 10, 20, 5, 100),
      memoryEnd = MemoryMetrics(2L, 30, 40, 8, 200),
      memoryPeak = MemoryMetrics(3L, 50, 60, 9, 300),
      gcCountDelta = 2,
      gcTimeMsDelta = 15,
      benchmarkMatrix = Vector(
        BenchmarkCell("cpu", "fp32", "cpu", 200.0, 1.0, Vector.empty, "CPU backend (precision=fp32)"),
        BenchmarkCell("cpu", "fp64", "cpu", 100.0, 2.0, Vector.empty, "CPU backend (precision=fp64)"),
        BenchmarkCell("gpu", "fp64", "gpu", 600.0, 0.3, Vector.empty, "Metal backend (...)"),
        BenchmarkCell("gpu", "fp32", "gpu", 900.0, 0.2, Vector("batchSoftmax"), "Metal backend (...)")
      ),
      notes = Map.empty
    )

  test("renderBenchmarkMetricsReport includes required benchmark report sections") {
    val run = sampleRun()
    val persisted = Some(
      RunObservability.PersistOutcome(
        current = RunObservability.StoredRunIndex("benchmark-abc", 1L, "benchmark", Some("demo"), 1234.56, "runs.jsonl"),
        baseline = Some(RunObservability.StoredRunIndex("benchmark-prev", 0L, "benchmark", None, 1100.0, "runs.jsonl")),
        regression = Some(
          RegressionCheckResult(
            metric = "throughput_ex_per_sec",
            baselineValue = 1100.0,
            currentValue = 1234.56,
            deltaPct = 12.23,
            thresholdPct = 5.0,
            status = "ok",
            message = "Throughput delta ..."
          )
        ),
        latestSummaryPath = Path.of("data/metrics/latest-summary.txt"),
        latestDiffSummaryPath = Some(Path.of("data/metrics/latest-diff-summary.txt"))
      )
    )

    val out = Main.renderBenchmarkMetricsReport(run, persisted, p => p.toString)
    assert(out.contains("=== Metrics Report (benchmark --metrics) ==="))
    assert(out.contains("Run ID: benchmark-abc"))
    assert(out.contains("Platform: macOS 14.4 | arch=aarch64 | java=21.0.6 | device=Apple M1"))
    assert(out.contains("Total runtime: 2.50 s"))
    assert(out.contains("Memory peak RSS: 300 bytes (0.00 MB)"))
    assert(out.contains("GC delta: count=2 timeMs=15"))
    assert(out.contains("Top stages (preferred run: gpu/fp32):"))
    assert(out.contains("matMul"))
    assert(out.contains("Relative speedup:"))
    assert(out.contains("gpu fp32 vs cpu fp32: 4.50x"))
    assert(out.contains("gpu fp64 vs cpu fp64: 6.00x"))
    assert(out.contains("fp32 vs fp64 on gpu: 1.50x"))
    assert(out.contains("fp32 vs fp64 on cpu: 2.00x"))
    assert(out.contains("Benchmark matrix:"))
    assert(out.contains("backend  precision"))
    assert(out.contains("cpu     fp32"))
    assert(out.contains("cpu     fp64"))
    assert(out.contains("gpu     fp32"))
    assert(out.contains("gpu     fp64"))
    assert(out.contains("none"))
    assert(out.contains("batchSoftmax"))
    assert(out.contains("Baseline: benchmark-prev"))
    assert(out.contains("Regression verdict: ok"))
    assert(out.contains("Summary file: data/metrics/latest-summary.txt"))
    assert(out.contains("Diff file: data/metrics/latest-diff-summary.txt"))
  }

  test("renderBenchmarkMetricsReport omits optional sections when data unavailable") {
    val run = sampleRun(topStages = Vector.empty).copy(benchmarkMatrix = Vector.empty)
    val out = Main.renderBenchmarkMetricsReport(run, persisted = None)
    assert(!out.contains("Top stages:"))
    assert(out.contains("Benchmark matrix:"))
    assert(!out.contains("Baseline:"))
    assert(!out.contains("Regression verdict:"))
  }
