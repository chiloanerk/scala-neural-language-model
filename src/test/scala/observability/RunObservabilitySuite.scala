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
      profile = RunProfile(10.0, 9.0, 3.0, 2.0, 4.0, "matMul: calls=10 timeMs=20.00, softmaxBatch: calls=10 timeMs=5.00"),
      gpu = GpuHotPathMetrics("gpu", "gpu", "ok", Vector("matMul"), "profile", fallbackDetected = false),
      platform = PlatformInfo("macOS", "14.0", "aarch64", "21.0.0", "Apple M1"),
      fallbackOps = Vector.empty,
      memoryStart = sampleMemory(1),
      memoryEnd = sampleMemory(2),
      memoryPeak = sampleMemory(3),
      gcCountDelta = 2,
      gcTimeMsDelta = 17,
      benchmarkMatrix = Vector.empty,
      notes = Map("k" -> "v")
    )

  test("toJson serializes key fields and nested structures") {
    val json = RunObservability.toJson(sampleRun("run-1"))
    assert(json.contains("\"runId\":\"run-1\""))
    assert(json.contains("\"mode\":\"train\""))
    assert(json.contains("\"domainValLosses\""))
    assert(json.contains("\"gpu\""))
    assert(json.contains("\"platform\""))
    assert(json.contains("\"memoryStart\""))
    assert(json.contains("\"gcCountDelta\":2"))
    assert(json.contains("\"topStages\""))
  }

  test("toJson uses dot as decimal separator regardless of locale") {
    val previous = java.util.Locale.getDefault
    try
      java.util.Locale.setDefault(java.util.Locale.GERMANY)
      val json = RunObservability.toJson(sampleRun("run-locale", throughput = 1234.56))
      assert(json.contains("\"throughputExPerSec\":1234.560000000000"))
      assert(!json.contains("1234,56"))
    finally java.util.Locale.setDefault(previous)
  }

  test("render summaries use dot as decimal separator regardless of locale") {
    val previous = java.util.Locale.getDefault
    try
      java.util.Locale.setDefault(java.util.Locale.GERMANY)
      val baseline = RunObservability.StoredRunIndex("base", 1L, "train", Some("b"), 100.0, "runs.jsonl")
      val current = RunObservability.StoredRunIndex("curr", 2L, "train", Some("c"), 1234.56, "runs.jsonl")
      val reg = RunObservability.throughputRegression(baseline, current, warnPct = 5.0)
      val summary = RunObservability.renderRunSummary(sampleRun("run-summary", throughput = 1234.56), Some(baseline), reg)
      val diff = RunObservability.renderDiffSummary(current, baseline, reg)

      assert(summary.contains("throughput_ex_per_sec: 1234.56"))
      assert(!summary.contains("1234,56"))
      assert(diff.contains("delta_pct: 1134.56"))
      assert(!diff.contains("1134,56"))
    finally java.util.Locale.setDefault(previous)
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
    val snapshots = RunObservability.loadRecentSnapshots(dir, Some("train"))
    assert(snapshots.nonEmpty)
    assertEquals(snapshots.last.runId, "persist-1")
  }

  test("loadRecentSnapshots parses legacy comma-decimal throughput") {
    val dir = Files.createTempDirectory("metrics-legacy")
    val historyPath = dir.resolve("runs.jsonl")
    val line =
      """{"runId":"legacy-1","timestampEpochMs":1700000000000,"mode":"benchmark","runLabel":"legacy","throughputExPerSec":123,45,"fallbackOps":["softmax"],"profile":{"topStages":[{"stage":"matMul","timeMs":12.0,"sharePct":60.0}]},"gcCountDelta":1,"gcTimeMsDelta":2,"benchmarkMatrix":[{"backend":"cpu","precision":"fp64","effectiveBackend":"cpu","exPerSec":100.0,"estSec":1.0,"fallbackOps":[],"diagnostics":"CPU"}]}"""
    Files.writeString(historyPath, line + "\n", StandardCharsets.UTF_8)

    val snapshots = RunObservability.loadRecentSnapshots(dir, Some("benchmark"))
    assertEquals(snapshots.length, 1)
    val snap = snapshots.head
    assertEquals(snap.runId, "legacy-1")
    assertEqualsDouble(snap.throughputExPerSec, 123.45, 1e-9)
    assertEquals(snap.fallbackOps, Vector("softmax"))
    assertEquals(snap.topStages.headOption.map(_.stage), Some("matMul"))
    assertEquals(snap.gcCountDelta, 1L)
    assertEquals(snap.gcTimeMsDelta, 2L)
    assertEquals(snap.benchmarkMatrix.headOption.map(_.backend), Some("cpu"))
  }

  test("topStagesFromProfile parses and ranks top stages") {
    val profile = "matMul: calls=530 timeMs=5164.99, linearBatch: calls=122 timeMs=104.14, softmaxBatch: calls=122 timeMs=2359.34"
    val stages = RunObservability.topStagesFromProfile(profile, limit = 2)
    assertEquals(stages.length, 2)
    assertEquals(stages.head.stage, "matMul")
    assert(stages.head.sharePct > stages(1).sharePct)
  }

  test("fallbackOpsFromDiagnostics extracts disabled ops") {
    val d = "Metal backend (device=Apple M1, precision=fp32, disabled=batchSoftmax:SIM_FAIL;batchCrossEntropy:SIM_FAIL)"
    val ops = RunObservability.fallbackOpsFromDiagnostics("gpu", "gpu", d)
    assertEquals(ops, Vector("batchCrossEntropy", "batchSoftmax"))
  }
