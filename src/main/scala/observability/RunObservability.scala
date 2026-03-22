package observability

import java.lang.management.ManagementFactory
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, StandardOpenOption}
import java.security.MessageDigest
import java.time.Instant
import java.util.Locale
import scala.util.matching.Regex

final case class MemoryMetrics(
    timestampEpochMs: Long,
    heapUsedBytes: Long,
    heapCommittedBytes: Long,
    nonHeapUsedBytes: Long,
    rssBytes: Long
)

final case class GcMetrics(
    collectionCount: Long,
    collectionTimeMs: Long
)

final case class TopStage(
    stage: String,
    timeMs: Double,
    sharePct: Double
)

final case class BenchmarkCell(
    backend: String,
    precision: String,
    effectiveBackend: String,
    exPerSec: Double,
    estSec: Double,
    fallbackOps: Vector[String],
    diagnostics: String
)

final case class GpuHotPathMetrics(
    requestedBackend: String,
    effectiveBackend: String,
    diagnostics: String,
    enabledOps: Vector[String],
    profileSummary: String,
    fallbackDetected: Boolean
)

final case class PlatformInfo(
    osName: String,
    osVersion: String,
    arch: String,
    javaVersion: String,
    deviceName: String
)

final case class RunProfile(
    totalSeconds: Double,
    epochSecondsTotal: Double,
    epochSecondsAvg: Double,
    batchMillisAvg: Double,
    batchMillisP95: Double,
    backendProfileSummary: String,
    topStages: Vector[TopStage] = Vector.empty
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
    platform: PlatformInfo,
    fallbackOps: Vector[String],
    memoryStart: MemoryMetrics,
    memoryEnd: MemoryMetrics,
    memoryPeak: MemoryMetrics,
    gcCountDelta: Long,
    gcTimeMsDelta: Long,
    benchmarkMatrix: Vector[BenchmarkCell] = Vector.empty,
    notes: Map[String, String]
)

final case class RunSnapshot(
    runId: String,
    timestampEpochMs: Long,
    mode: String,
    runLabel: Option[String],
    throughputExPerSec: Double,
    fallbackOps: Vector[String],
    topStages: Vector[TopStage],
    benchmarkMatrix: Vector[BenchmarkCell],
    gcCountDelta: Long,
    gcTimeMsDelta: Long
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

object GcProbe:
  def snapshot(): GcMetrics =
    val beans = ManagementFactory.getGarbageCollectorMXBeans
    val count = beans.toArray.map(_.asInstanceOf[java.lang.management.GarbageCollectorMXBean].getCollectionCount).filter(_ >= 0).sum
    val time = beans.toArray.map(_.asInstanceOf[java.lang.management.GarbageCollectorMXBean].getCollectionTime).filter(_ >= 0).sum
    GcMetrics(count, time)

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

  private val StageRegex: Regex = raw"([A-Za-z][A-Za-z0-9]*): calls=\d+ timeMs=([0-9]+(?:[.,][0-9]+)?)".r
  private val DisabledRegex: Regex = raw"disabled=([^)]*)".r

  def buildRunId(mode: String): String =
    val now = Instant.now().toString.replace(":", "-")
    val suffix = java.util.UUID.randomUUID().toString.take(8)
    s"${mode.toLowerCase}-$now-$suffix"

  def configFingerprint(entries: Vector[(String, String)]): String =
    val payload = entries.sortBy(_._1).map { case (k, v) => s"$k=$v" }.mkString("|")
    val digest = MessageDigest.getInstance("SHA-256").digest(payload.getBytes(StandardCharsets.UTF_8))
    digest.take(8).map("%02x".format(_)).mkString

  def topStagesFromProfile(profileSummary: String, limit: Int = 3): Vector[TopStage] =
    val rows = StageRegex.findAllMatchIn(profileSummary).toVector.flatMap { m =>
      val stage = m.group(1)
      parseDoubleFlexible(m.group(2)).map(v => stage -> v)
    }
    val total = rows.map(_._2).sum
    rows.sortBy(-_._2).take(math.max(0, limit)).map { case (stage, timeMs) =>
      val pct = if total <= 0 then 0.0 else (timeMs / total) * 100.0
      TopStage(stage, timeMs, pct)
    }

  def fallbackOpsFromDiagnostics(requestedBackend: String, effectiveBackend: String, diagnostics: String): Vector[String] =
    if requestedBackend == "gpu" && effectiveBackend != "gpu" then Vector("all")
    else
      DisabledRegex.findFirstMatchIn(diagnostics).toVector
        .flatMap(_.group(1).split(";").toVector.map(_.trim).filter(_.nonEmpty))
        .map(_.takeWhile(_ != ':'))
        .filter(_.nonEmpty)
        .distinct
        .sorted

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

  def loadRecentSnapshots(metricsDir: Path, modeFilter: Option[String], limit: Int = 20): Vector[RunSnapshot] =
    val path = metricsDir.resolve(HistoryFile)
    if !Files.isRegularFile(path) then Vector.empty
    else
      val lines = Files.readAllLines(path, StandardCharsets.UTF_8).toArray(new Array[String](0)).toVector
      val parsed = lines.flatMap(parseRunSnapshotLine)
      val filtered = modeFilter match
        case Some(m) => parsed.filter(_.mode == m)
        case None    => parsed
      filtered.sortBy(_.timestampEpochMs).takeRight(math.max(1, limit))

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
          s"Throughput regressed ${fmtPct(-deltaPct)} vs baseline ${baseline.runId} (threshold=${fmtPct(math.abs(warnPct))})."
        else
          s"Throughput delta vs baseline ${baseline.runId}: ${fmtSignedPct(deltaPct)} (threshold=${fmtPct(math.abs(warnPct))})."
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
      else run.domainValLosses.map { case (d, v) => s"$d=${fmt(v, 4)}" }.mkString(", ")
    val retention =
      if run.retentionByDomainPct.isEmpty then "-"
      else run.retentionByDomainPct.map { case (d, v) => s"$d=${fmt(v, 1)}%" }.mkString(", ")
    val topStages =
      if run.profile.topStages.isEmpty then "-"
      else run.profile.topStages.map(s => s"${s.stage}:${fmt(s.timeMs, 2)}ms(${fmt(s.sharePct, 1)}%)").mkString(", ")
    val fallback = if run.fallbackOps.isEmpty then "-" else run.fallbackOps.mkString(",")
    val matrix =
      if run.benchmarkMatrix.isEmpty then "-"
      else
        run.benchmarkMatrix
          .sortBy(c => (c.backend, c.precision))
          .map(c => s"${c.backend}/${c.precision}=${fmt(c.exPerSec, 2)}ex/s(eff=${c.effectiveBackend},fb=${if c.fallbackOps.isEmpty then "-" else c.fallbackOps.mkString("+")})")
          .mkString("; ")

    val baselineLine = baseline.map(b => s"baseline: ${b.runId} (${b.mode}, ${fmt(b.throughputExPerSec, 2)} ex/s)").getOrElse("baseline: none")
    val regressionLine = regression.map(r => s"regression: ${r.status} (${r.message})").getOrElse("regression: n/a")

    s"""Run summary
|run_id: ${run.runId}
|mode: ${run.mode}
|timestamp_ms: ${run.timestampEpochMs}
|label: ${run.runLabel.getOrElse("-")}
|dataset: ${run.dataset}
|config_fingerprint: ${run.configFingerprint}
|platform: os=${run.platform.osName} ${run.platform.osVersion} arch=${run.platform.arch} java=${run.platform.javaVersion} device=${run.platform.deviceName}
|backend: requested=${run.requestedBackend} effective=${run.gpu.effectiveBackend} precision=${run.precision}
|throughput_ex_per_sec: ${fmt(run.throughputExPerSec, 2)}
|losses: train=${run.trainLoss.map(v => fmt(v, 4)).getOrElse("-")} val=${run.valLoss.map(v => fmt(v, 4)).getOrElse("-")} ppl=${run.valPerplexity.map(v => fmt(v, 2)).getOrElse("-")}
|domains: $domains
|retention: $retention
|top_stages: $topStages
|fallback_ops: $fallback
|benchmark_matrix: $matrix
|profile: total_s=${fmt(run.profile.totalSeconds, 2)} epoch_total_s=${fmt(run.profile.epochSecondsTotal, 2)} epoch_avg_s=${fmt(run.profile.epochSecondsAvg, 2)} batch_avg_ms=${fmt(run.profile.batchMillisAvg, 3)} batch_p95_ms=${fmt(run.profile.batchMillisP95, 3)}
|gpu: fallback=${run.gpu.fallbackDetected} enabled_ops=${if run.gpu.enabledOps.isEmpty then "-" else run.gpu.enabledOps.mkString(",")}
|memory_bytes: start_heap=${run.memoryStart.heapUsedBytes} end_heap=${run.memoryEnd.heapUsedBytes} peak_heap=${run.memoryPeak.heapUsedBytes} start_rss=${run.memoryStart.rssBytes} end_rss=${run.memoryEnd.rssBytes} peak_rss=${run.memoryPeak.rssBytes}
|gc: count_delta=${run.gcCountDelta} time_ms_delta=${run.gcTimeMsDelta}
|$baselineLine
|$regressionLine
|""".stripMargin

  def renderDiffSummary(current: StoredRunIndex, baseline: StoredRunIndex, regression: Option[RegressionCheckResult]): String =
    val deltaPct =
      if baseline.throughputExPerSec <= 0 then 0.0
      else ((current.throughputExPerSec - baseline.throughputExPerSec) / baseline.throughputExPerSec) * 100.0
    val verdict = regression.map(_.status).getOrElse("n/a")
    s"""Run comparison
|current: ${current.runId} (${fmt(current.throughputExPerSec, 2)} ex/s)
|baseline: ${baseline.runId} (${fmt(baseline.throughputExPerSec, 2)} ex/s)
|delta_pct: ${fmt(deltaPct, 2)}
|status: $verdict
|""".stripMargin

  def toJson(run: RunMetrics): String =
    val domainJson = run.domainValLosses.map { case (d, v) => s"""{"domain":${j(d)},"valLoss":${jd(v)}}""" }.mkString("[", ",", "]")
    val retentionJson = run.retentionByDomainPct.map { case (d, v) => s"""{"domain":${j(d)},"retentionPct":${jd(v)}}""" }.mkString("[", ",", "]")
    val notesJson = run.notes.toVector.sortBy(_._1).map { case (k, v) => s"${j(k)}:${j(v)}" }.mkString("{", ",", "}")
    val fallbackJson = run.fallbackOps.map(j).mkString("[", ",", "]")
    val topStagesJson = run.profile.topStages.map(s => s"""{"stage":${j(s.stage)},"timeMs":${jd(s.timeMs)},"sharePct":${jd(s.sharePct)}}""").mkString("[", ",", "]")
    val benchJson = run.benchmarkMatrix
      .map { c =>
        s"""{"backend":${j(c.backend)},"precision":${j(c.precision)},"effectiveBackend":${j(c.effectiveBackend)},"exPerSec":${jd(c.exPerSec)},"estSec":${jd(c.estSec)},"fallbackOps":${c.fallbackOps.map(j).mkString("[", ",", "]")},"diagnostics":${j(c.diagnostics)}}"""
      }
      .mkString("[", ",", "]")

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
      s""""profile":${profileJson(run.profile, topStagesJson)},""" +
      s""""gpu":${gpuJson(run.gpu)},""" +
      s""""platform":${platformJson(run.platform)},""" +
      s""""fallbackOps":$fallbackJson,""" +
      s""""memoryStart":${memoryJson(run.memoryStart)},""" +
      s""""memoryEnd":${memoryJson(run.memoryEnd)},""" +
      s""""memoryPeak":${memoryJson(run.memoryPeak)},""" +
      s""""gcCountDelta":${run.gcCountDelta},""" +
      s""""gcTimeMsDelta":${run.gcTimeMsDelta},""" +
      s""""benchmarkMatrix":$benchJson,""" +
      s""""notes":$notesJson""" +
      s"}"

  private def profileJson(p: RunProfile, topStagesJson: String): String =
    s"{" +
      s""""totalSeconds":${jd(p.totalSeconds)},""" +
      s""""epochSecondsTotal":${jd(p.epochSecondsTotal)},""" +
      s""""epochSecondsAvg":${jd(p.epochSecondsAvg)},""" +
      s""""batchMillisAvg":${jd(p.batchMillisAvg)},""" +
      s""""batchMillisP95":${jd(p.batchMillisP95)},""" +
      s""""backendProfileSummary":${j(p.backendProfileSummary)},""" +
      s""""topStages":$topStagesJson""" +
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

  private def platformJson(p: PlatformInfo): String =
    s"{" +
      s""""osName":${j(p.osName)},""" +
      s""""osVersion":${j(p.osVersion)},""" +
      s""""arch":${j(p.arch)},""" +
      s""""javaVersion":${j(p.javaVersion)},""" +
      s""""deviceName":${j(p.deviceName)}""" +
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

  private def parseRunSnapshotLine(line: String): Option[RunSnapshot] =
    for
      runId <- extractString(line, "runId")
      mode <- extractString(line, "mode")
      ts <- extractLong(line, "timestampEpochMs")
      throughput <- extractDouble(line, "throughputExPerSec")
    yield
      val label = extractNullableString(line, "runLabel")
      val fallbackOps = extractStringArray(line, "fallbackOps")
      val topStages = extractTopStages(line)
      val matrix = extractBenchmarkMatrix(line)
      val gcCount = extractLong(line, "gcCountDelta").getOrElse(0L)
      val gcTime = extractLong(line, "gcTimeMsDelta").getOrElse(0L)
      RunSnapshot(runId, ts, mode, label, throughput, fallbackOps, topStages, matrix, gcCount, gcTime)

  private def extractString(line: String, key: String): Option[String] =
    raw""""\Q$key\E":"([^"]*)"""".r.findFirstMatchIn(line).map(_.group(1))

  private def extractNullableString(line: String, key: String): Option[String] =
    raw""""\Q$key\E":null""".r.findFirstMatchIn(line) match
      case Some(_) => None
      case None    => extractString(line, key)

  private def extractDouble(line: String, key: String): Option[Double] =
    raw""""\Q$key\E":(-?[0-9]+(?:[.,][0-9]+)?)""".r.findFirstMatchIn(line).flatMap(m => parseDoubleFlexible(m.group(1)))

  private def extractLong(line: String, key: String): Option[Long] =
    raw""""\Q$key\E":(-?[0-9]+)""".r.findFirstMatchIn(line).flatMap(m => m.group(1).toLongOption)

  private def extractStringArray(line: String, key: String): Vector[String] =
    val body = raw""""\Q$key\E":\[(.*?)\]""".r.findFirstMatchIn(line).map(_.group(1)).getOrElse("")
    raw""""([^"]+)"""".r.findAllMatchIn(body).map(_.group(1)).toVector

  private def extractTopStages(line: String): Vector[TopStage] =
    raw"""\{"stage":"([^"]+)","timeMs":(-?[0-9]+(?:\.[0-9]+)?),"sharePct":(-?[0-9]+(?:\.[0-9]+)?)\}""".r
      .findAllMatchIn(line)
      .flatMap { m =>
        for
          t <- parseDoubleFlexible(m.group(2))
          s <- parseDoubleFlexible(m.group(3))
        yield TopStage(m.group(1), t, s)
      }
      .toVector

  private def extractBenchmarkMatrix(line: String): Vector[BenchmarkCell] =
    raw"""\{"backend":"([^"]+)","precision":"([^"]+)","effectiveBackend":"([^"]+)","exPerSec":(-?[0-9]+(?:\.[0-9]+)?),"estSec":(-?[0-9]+(?:\.[0-9]+)?),"fallbackOps":\[(.*?)\],"diagnostics":"([^"]*)"\}""".r
      .findAllMatchIn(line)
      .flatMap { m =>
        for
          ex <- parseDoubleFlexible(m.group(4))
          est <- parseDoubleFlexible(m.group(5))
        yield
          val fallbackOps = raw""""([^"]+)"""".r.findAllMatchIn(m.group(6)).map(_.group(1)).toVector
          BenchmarkCell(m.group(1), m.group(2), m.group(3), ex, est, fallbackOps, m.group(7))
      }
      .toVector

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
    if v.isNaN || v.isInfinity then "null" else String.format(Locale.US, "%.12f", Double.box(v))

  private def fmt(v: Double, decimals: Int): String =
    String.format(Locale.US, s"%.${math.max(0, decimals)}f", Double.box(v))

  private def fmtPct(v: Double): String = s"${fmt(v, 2)}%"
  private def fmtSignedPct(v: Double): String =
    val prefix = if v > 0 then "+" else ""
    s"$prefix${fmt(v, 2)}%"

  private def parseDoubleFlexible(raw: String): Option[Double] =
    raw.replace(',', '.').toDoubleOption
