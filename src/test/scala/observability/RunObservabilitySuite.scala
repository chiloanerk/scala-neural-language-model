package observability

import munit.FunSuite

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

class RunObservabilitySuite extends FunSuite:
  private def sampleMemory(ts: Long): MemoryMetrics =
    MemoryMetrics(
      timestampEpochMs = ts,
      heapUsedBytes = 100,
      heapCommittedBytes = 200,
      nonHeapUsedBytes = 50,
      rssBytes = 300
    )

  private def sampleRun(runId: String, mode: String = "train", throughput: Double = 100.0, label: Option[String] = Some("exp-a")): RunMetrics =
    RunMetrics(
      runId = runId,
      timestampEpochMs = 1700000000000L,
      mode = mode,
      runLabel = label,
      configFingerprint = "abc123",
      dataset = "data/corpus/example-corpus.txt",
      requestedBackend = "gpu",
      precision = "fp32",
      throughputExPerSec = throughput,
      trainLoss = Some(1.2),
      valLoss = Some(1.1),
      valPerplexity = Some(3.0),
      epochs = 3,
      domainValLosses = Vector("d1" -> 1.1),
      retentionByDomainPct = Vector("d1" -> 98.0),
      profile = RunProfile(10.0, 9.0, 3.0, 2.0, 4.0, "profile"),
      gpu = GpuHotPathMetrics("gpu", "gpu", "ok", Vector("matMul"), "profile", fallbackDetected = false),
      memoryStart = sampleMemory(1),
      memoryEnd = sampleMemory(2),
      memoryPeak = sampleMemory(3),
      notes = Map("k" -> "v")
    )

  test("toJson serializes key fields and nested structures") {
    val json = RunObservability.toJson(sampleRun("run-1"))
    assert(json.contains("\"runId\":\"run-1\""))
    assert(json.contains("\"mode\":\"train\""))
    assert(json.contains("\"domainValLosses\""))
    assert(json.contains("\"gpu\""))
    assert(json.contains("\"memoryStart\""))
  }

  test("throughputRegression warns on slowdown beyond threshold") {
    val baseline = RunObservability.StoredRunIndex("b1", 1L, "train", None, 100.0, "p")
    val current = RunObservability.StoredRunIndex("c1", 2L, "train", None, 90.0, "p")
    val reg = RunObservability.throughputRegression(baseline, current, warnPct = 5.0)
    assert(reg.nonEmpty)
    assertEquals(reg.get.status, "warn")
  }

  test("selectBaseline resolves latest/id/label") {
    val current = RunObservability.StoredRunIndex("r3", 30L, "train", Some("new"), 100.0, "p")
    val index = Vector(
      RunObservability.StoredRunIndex("r1", 10L, "train", Some("exp-a"), 90.0, "p"),
      RunObservability.StoredRunIndex("r2", 20L, "train", Some("exp-b"), 95.0, "p"),
      current
    )

    val latest = RunObservability.selectBaseline(index, current, Some("latest"))
    assertEquals(latest.map(_.runId), Some("r2"))

    val byId = RunObservability.selectBaseline(index, current, Some("r1"))
    assertEquals(byId.map(_.runId), Some("r1"))

    val byLabel = RunObservability.selectBaseline(index, current, Some("exp-b"))
    assertEquals(byLabel.map(_.runId), Some("r2"))
  }

  test("persistAndReport writes history/index/summary files") {
    val dir = Files.createTempDirectory("metrics-suite")
    val run = sampleRun("persist-1", throughput = 123.0)
    val out = RunObservability.persistAndReport(
      run,
      RunObservability.Settings(recordMetrics = true, metricsDir = dir, runLabel = run.runLabel, compareTo = None, regressionWarnPct = 5.0)
    )
    assert(out.nonEmpty)

    val historyPath = dir.resolve("runs.jsonl")
    val indexPath = dir.resolve("runs-index.tsv")
    val summaryPath = dir.resolve("latest-summary.txt")
    assert(Files.isRegularFile(historyPath))
    assert(Files.isRegularFile(indexPath))
    assert(Files.isRegularFile(summaryPath))

    val history = Files.readString(historyPath, StandardCharsets.UTF_8)
    assert(history.contains("\"runId\":\"persist-1\""))
  }
