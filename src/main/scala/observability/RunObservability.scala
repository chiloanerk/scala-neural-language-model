package observability

import java.lang.management.ManagementFactory
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, StandardOpenOption}
import java.security.MessageDigest
import java.time.Instant

final case class MemoryMetrics(
    timestampEpochMs: Long,
    heapUsedBytes: Long,
    heapCommittedBytes: Long,
    nonHeapUsedBytes: Long,
    rssBytes: Long
)

final case class GpuHotPathMetrics(
    requestedBackend: String,
    effectiveBackend: String,
    diagnostics: String,
    enabledOps: Vector[String],
    profileSummary: String,
    fallbackDetected: Boolean
)

final case class RunProfile(
    totalSeconds: Double,
    epochSecondsTotal: Double,
    epochSecondsAvg: Double,
    batchMillisAvg: Double,
    batchMillisP95: Double,
    backendProfileSummary: String
)

final case class RegressionCheckResult(
    metric: String,
    baselineValue: Double,
    currentValue: Double,
    deltaPct: Double,
    thresholdPct: Double,
    status: String,
    message: String
)

final case class RunMetrics(
    runId: String,
    timestampEpochMs: Long,
    mode: String,
    runLabel: Option[String],
    configFingerprint: String,
    dataset: String,
    requestedBackend: String,
    precision: String,
    throughputExPerSec: Double,
    trainLoss: Option[Double],
    valLoss: Option[Double],
    valPerplexity: Option[Double],
    epochs: Int,
    domainValLosses: Vector[(String, Double)],
    retentionByDomainPct: Vector[(String, Double)],
    profile: RunProfile,
    gpu: GpuHotPathMetrics,
    memoryStart: MemoryMetrics,
    memoryEnd: MemoryMetrics,
    memoryPeak: MemoryMetrics,
    notes: Map[String, String]
)

object MemoryProbe:
  def snapshot(nowMs: Long = System.currentTimeMillis()): MemoryMetrics =
    val mx = ManagementFactory.getMemoryMXBean
    val heap = mx.getHeapMemoryUsage
    val nonHeap = mx.getNonHeapMemoryUsage
    MemoryMetrics(
      timestampEpochMs = nowMs,
      heapUsedBytes = heap.getUsed,
      heapCommittedBytes = heap.getCommitted,
      nonHeapUsedBytes = nonHeap.getUsed,
      rssBytes = processRssBytes()
    )

  private def processRssBytes(): Long =
    try
      val pid = ProcessHandle.current().pid().toString
      val pb = ProcessBuilder("/bin/zsh", "-lc", s"ps -o rss= -p $pid")
      val p = pb.start()
      val out = new String(p.getInputStream.readAllBytes(), StandardCharsets.UTF_8).trim
      val _ = p.waitFor()
      out.toLongOption.map(_ * 1024L).getOrElse(-1L)
    catch
      case _: Exception => -1L

object RunObservability:
  val DefaultMetricsDir: Path = Path.of("data/metrics")
  private val HistoryFile = "runs.jsonl"
  private val IndexFile = "runs-index.tsv"
  private val LatestSummaryFile = "latest-summary.txt"
  private val LatestDiffSummaryFile = "latest-diff-summary.txt"

  final case class Settings(
      recordMetrics: Boolean = true,
      metricsDir: Path = DefaultMetricsDir,
      runLabel: Option[String] = None,
      compareTo: Option[String] = None,
      regressionWarnPct: Double = 5.0
  )

  final case class StoredRunIndex(
      runId: String,
      timestampEpochMs: Long,
      mode: String,
      runLabel: Option[String],
      throughputExPerSec: Double,
      pathHint: String
  )

  final case class PersistOutcome(
      current: StoredRunIndex,
      baseline: Option[StoredRunIndex],
      regression: Option[RegressionCheckResult],
      latestSummaryPath: Path,
      latestDiffSummaryPath: Option[Path]
  )

  def buildRunId(mode: String): String =
    val now = Instant.now().toString.replace(":", "-")
    val suffix = java.util.UUID.randomUUID().toString.take(8)
    s"${mode.toLowerCase}-$now-$suffix"

  def configFingerprint(entries: Vector[(String, String)]): String =
    val payload = entries.sortBy(_._1).map { case (k, v) => s"$k=$v" }.mkString("|")
    val digest = MessageDigest.getInstance("SHA-256").digest(payload.getBytes(StandardCharsets.UTF_8))
    digest.take(8).map("%02x".format(_)).mkString

  def persistAndReport(run: RunMetrics, settings: Settings, log: String => Unit = s => println(s)): Option[PersistOutcome] =
    if !settings.recordMetrics then None
    else
      Files.createDirectories(settings.metricsDir)
      val historyPath = settings.metricsDir.resolve(HistoryFile)
      val indexPath = settings.metricsDir.resolve(IndexFile)
      val latestSummaryPath = settings.metricsDir.resolve(LatestSummaryFile)
      val latestDiffPath = settings.metricsDir.resolve(LatestDiffSummaryFile)

      appendLine(historyPath, toJson(run))
      val current = StoredRunIndex(
        runId = run.runId,
        timestampEpochMs = run.timestampEpochMs,
        mode = run.mode,
        runLabel = run.runLabel,
        throughputExPerSec = run.throughputExPerSec,
        pathHint = historyPath.toString
      )
      appendLine(indexPath, toIndexLine(current))

      val index = loadIndex(settings.metricsDir)
      val baseline = selectBaseline(index, current, settings.compareTo)
      val regression = baseline.flatMap(b => throughputRegression(b, current, settings.regressionWarnPct))

      val summary = renderRunSummary(run, baseline, regression)
      Files.writeString(latestSummaryPath, summary, StandardCharsets.UTF_8)
      log(s"[metrics] wrote summary: $latestSummaryPath")

      val diffPath = baseline.map { b =>
        val diff = renderDiffSummary(current, b, regression)
        Files.writeString(latestDiffPath, diff, StandardCharsets.UTF_8)
        log(s"[metrics] wrote diff summary: $latestDiffPath")
        latestDiffPath
      }

      regression.foreach { r =>
        if r.status == "warn" then log(s"[metrics] warning: ${r.message}")
      }

      Some(PersistOutcome(current, baseline, regression, latestSummaryPath, diffPath))

  def loadIndex(metricsDir: Path): Vector[StoredRunIndex] =
    val path = metricsDir.resolve(IndexFile)
    if !Files.isRegularFile(path) then Vector.empty
    else
      Files.readAllLines(path, StandardCharsets.UTF_8).toArray(new Array[String](0)).toVector.flatMap(parseIndexLine)

  def latestByMode(index: Vector[StoredRunIndex], mode: String): Option[StoredRunIndex] =
    index.filter(_.mode == mode).sortBy(_.timestampEpochMs).lastOption

  def selectBaseline(index: Vector[StoredRunIndex], current: StoredRunIndex, compareTo: Option[String]): Option[StoredRunIndex] =
    val others = index.filterNot(_.runId == current.runId)
    compareTo.map(_.trim).filter(_.nonEmpty) match
      case None | Some("latest") =>
        others.filter(_.mode == current.mode).sortBy(_.timestampEpochMs).lastOption
      case Some(raw) =>
        others.find(_.runId == raw)
          .orElse(others.filter(_.runLabel.contains(raw)).sortBy(_.timestampEpochMs).lastOption)

  def throughputRegression(baseline: StoredRunIndex, current: StoredRunIndex, warnPct: Double): Option[RegressionCheckResult] =
    if baseline.throughputExPerSec <= 0 || current.throughputExPerSec <= 0 then None
    else
      val deltaPct = ((current.throughputExPerSec - baseline.throughputExPerSec) / baseline.throughputExPerSec) * 100.0
      val status = if deltaPct <= -math.abs(warnPct) then "warn" else "ok"
      val message =
        if status == "warn" then
          f"Throughput regressed ${-deltaPct}%.2f%% vs baseline ${baseline.runId} (threshold=${math.abs(warnPct)}%.2f%%)."
        else
          f"Throughput delta vs baseline ${baseline.runId}: ${deltaPct}%.2f%% (threshold=${math.abs(warnPct)}%.2f%%)."
      Some(
        RegressionCheckResult(
          metric = "throughput_ex_per_sec",
          baselineValue = baseline.throughputExPerSec,
          currentValue = current.throughputExPerSec,
          deltaPct = deltaPct,
          thresholdPct = math.abs(warnPct),
          status = status,
          message = message
        )
      )

  def renderRunSummary(
      run: RunMetrics,
      baseline: Option[StoredRunIndex],
      regression: Option[RegressionCheckResult]
  ): String =
    val domains =
      if run.domainValLosses.isEmpty then "-"
      else run.domainValLosses.map { case (d, v) => f"$d=$v%.4f" }.mkString(", ")
    val retention =
      if run.retentionByDomainPct.isEmpty then "-"
      else run.retentionByDomainPct.map { case (d, v) => f"$d=$v%.1f%%" }.mkString(", ")

    val baselineLine = baseline.map(b => s"baseline: ${b.runId} (${b.mode}, ${f"${b.throughputExPerSec}%.2f"} ex/s)").getOrElse("baseline: none")
    val regressionLine = regression.map(r => s"regression: ${r.status} (${r.message})").getOrElse("regression: n/a")

    s"""Run summary
|run_id: ${run.runId}
|mode: ${run.mode}
|timestamp_ms: ${run.timestampEpochMs}
|label: ${run.runLabel.getOrElse("-")}
|dataset: ${run.dataset}
|config_fingerprint: ${run.configFingerprint}
|backend: requested=${run.requestedBackend} effective=${run.gpu.effectiveBackend} precision=${run.precision}
|throughput_ex_per_sec: ${f"${run.throughputExPerSec}%.2f"}
|losses: train=${run.trainLoss.map(v => f"$v%.4f").getOrElse("-")} val=${run.valLoss.map(v => f"$v%.4f").getOrElse("-")} ppl=${run.valPerplexity.map(v => f"$v%.2f").getOrElse("-")}
|domains: $domains
|retention: $retention
|profile: total_s=${f"${run.profile.totalSeconds}%.2f"} epoch_total_s=${f"${run.profile.epochSecondsTotal}%.2f"} epoch_avg_s=${f"${run.profile.epochSecondsAvg}%.2f"} batch_avg_ms=${f"${run.profile.batchMillisAvg}%.3f"} batch_p95_ms=${f"${run.profile.batchMillisP95}%.3f"}
|gpu: fallback=${run.gpu.fallbackDetected} enabled_ops=${if run.gpu.enabledOps.isEmpty then "-" else run.gpu.enabledOps.mkString(",")}
|memory_bytes: start_heap=${run.memoryStart.heapUsedBytes} end_heap=${run.memoryEnd.heapUsedBytes} peak_heap=${run.memoryPeak.heapUsedBytes} start_rss=${run.memoryStart.rssBytes} end_rss=${run.memoryEnd.rssBytes} peak_rss=${run.memoryPeak.rssBytes}
|$baselineLine
|$regressionLine
|""".stripMargin

  def renderDiffSummary(current: StoredRunIndex, baseline: StoredRunIndex, regression: Option[RegressionCheckResult]): String =
    val deltaPct =
      if baseline.throughputExPerSec <= 0 then 0.0
      else ((current.throughputExPerSec - baseline.throughputExPerSec) / baseline.throughputExPerSec) * 100.0
    val verdict = regression.map(_.status).getOrElse("n/a")
    s"""Run comparison
|current: ${current.runId} (${f"${current.throughputExPerSec}%.2f"} ex/s)
|baseline: ${baseline.runId} (${f"${baseline.throughputExPerSec}%.2f"} ex/s)
|delta_pct: ${f"$deltaPct%.2f"}
|status: $verdict
|""".stripMargin

  def toJson(run: RunMetrics): String =
    val domainJson = run.domainValLosses.map { case (d, v) => s"""{"domain":${j(d)},"valLoss":${jd(v)}}""" }.mkString("[", ",", "]")
    val retentionJson = run.retentionByDomainPct.map { case (d, v) => s"""{"domain":${j(d)},"retentionPct":${jd(v)}}""" }.mkString("[", ",", "]")
    val notesJson = run.notes.toVector.sortBy(_._1).map { case (k, v) => s"${j(k)}:${j(v)}" }.mkString("{", ",", "}")
    s"{" +
      s""""runId":${j(run.runId)},""" +
      s""""timestampEpochMs":${run.timestampEpochMs},""" +
      s""""mode":${j(run.mode)},""" +
      s""""runLabel":${jo(run.runLabel)},""" +
      s""""configFingerprint":${j(run.configFingerprint)},""" +
      s""""dataset":${j(run.dataset)},""" +
      s""""requestedBackend":${j(run.requestedBackend)},""" +
      s""""precision":${j(run.precision)},""" +
      s""""throughputExPerSec":${jd(run.throughputExPerSec)},""" +
      s""""trainLoss":${jod(run.trainLoss)},""" +
      s""""valLoss":${jod(run.valLoss)},""" +
      s""""valPerplexity":${jod(run.valPerplexity)},""" +
      s""""epochs":${run.epochs},""" +
      s""""domainValLosses":$domainJson,""" +
      s""""retentionByDomainPct":$retentionJson,""" +
      s""""profile":${profileJson(run.profile)},""" +
      s""""gpu":${gpuJson(run.gpu)},""" +
      s""""memoryStart":${memoryJson(run.memoryStart)},""" +
      s""""memoryEnd":${memoryJson(run.memoryEnd)},""" +
      s""""memoryPeak":${memoryJson(run.memoryPeak)},""" +
      s""""notes":$notesJson""" +
      s"}"

  private def profileJson(p: RunProfile): String =
    s"{" +
      s""""totalSeconds":${jd(p.totalSeconds)},""" +
      s""""epochSecondsTotal":${jd(p.epochSecondsTotal)},""" +
      s""""epochSecondsAvg":${jd(p.epochSecondsAvg)},""" +
      s""""batchMillisAvg":${jd(p.batchMillisAvg)},""" +
      s""""batchMillisP95":${jd(p.batchMillisP95)},""" +
      s""""backendProfileSummary":${j(p.backendProfileSummary)}""" +
      s"}"

  private def gpuJson(g: GpuHotPathMetrics): String =
    val ops = g.enabledOps.map(j).mkString("[", ",", "]")
    s"{" +
      s""""requestedBackend":${j(g.requestedBackend)},""" +
      s""""effectiveBackend":${j(g.effectiveBackend)},""" +
      s""""diagnostics":${j(g.diagnostics)},""" +
      s""""enabledOps":$ops,""" +
      s""""profileSummary":${j(g.profileSummary)},""" +
      s""""fallbackDetected":${if g.fallbackDetected then "true" else "false"}""" +
      s"}"

  private def memoryJson(m: MemoryMetrics): String =
    s"{" +
      s""""timestampEpochMs":${m.timestampEpochMs},""" +
      s""""heapUsedBytes":${m.heapUsedBytes},""" +
      s""""heapCommittedBytes":${m.heapCommittedBytes},""" +
      s""""nonHeapUsedBytes":${m.nonHeapUsedBytes},""" +
      s""""rssBytes":${m.rssBytes}""" +
      s"}"

  private def appendLine(path: Path, line: String): Unit =
    Files.writeString(
      path,
      line + "\n",
      StandardCharsets.UTF_8,
      StandardOpenOption.CREATE,
      StandardOpenOption.APPEND
    )

  private def toIndexLine(idx: StoredRunIndex): String =
    val label = idx.runLabel.getOrElse("")
    s"${idx.runId}\t${idx.timestampEpochMs}\t${idx.mode}\t${label.replace("\t", " ")}\t${idx.throughputExPerSec}\t${idx.pathHint}"

  private def parseIndexLine(line: String): Option[StoredRunIndex] =
    val parts = line.split("\t", -1).toVector
    if parts.length < 6 then None
    else
      Some(
        StoredRunIndex(
          runId = parts(0),
          timestampEpochMs = parts(1).toLongOption.getOrElse(0L),
          mode = parts(2),
          runLabel = Option(parts(3)).filter(_.nonEmpty),
          throughputExPerSec = parts(4).toDoubleOption.getOrElse(0.0),
          pathHint = parts(5)
        )
      )

  private def j(raw: String): String =
    val escaped = raw
      .replace("\\", "\\\\")
      .replace("\"", "\\\"")
      .replace("\n", "\\n")
      .replace("\r", "\\r")
      .replace("\t", "\\t")
    s""""$escaped""""

  private def jo(raw: Option[String]): String = raw.map(j).getOrElse("null")
  private def jod(raw: Option[Double]): String = raw.map(jd).getOrElse("null")
  private def jd(v: Double): String =
    if v.isNaN || v.isInfinity then "null" else f"$v%.12f"
