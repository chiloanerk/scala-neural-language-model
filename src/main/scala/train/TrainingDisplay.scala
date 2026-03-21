package train

final case class BatchProgress(
    epoch: Int,
    totalEpochs: Int,
    percent: Int,
    elapsedSec: Double,
    remainingSec: Double,
    examplesPerSec: Double,
    avgLoss: Double
)

trait TrainingDisplay:
  def onTrainingStart(estimateSec: Double): Unit
  def onEpochStart(epoch: Int, totalEpochs: Int, lr: Double, totalExamples: Int): Unit
  def onBatchProgress(p: BatchProgress): Unit
  def onEpochComplete(metrics: EpochMetrics, isBest: Boolean, patienceCounter: Int, patience: Int): Unit
  def onCancellationRequested(epoch: Int): Unit
  def onTrainingComplete(interrupted: Boolean): Unit

object TrainingDisplay:
  def isInteractiveTerminal: Boolean =
    val forceRaw = Option(System.getenv("TRAIN_PROGRESS_FORCE_TTY")).map(_.trim.toLowerCase)
    val forceOn = forceRaw.exists(v => Set("1", "true", "yes", "on").contains(v))
    val forceOff = forceRaw.exists(v => Set("0", "false", "no", "off").contains(v))
    if forceOn then true
    else if forceOff then false
    else
      val term = Option(System.getenv("TERM")).getOrElse("").toLowerCase
      val ci = Option(System.getenv("CI")).exists(_.nonEmpty)
      val noColor = Option(System.getenv("NO_COLOR")).exists(_.nonEmpty)
      val termLooksInteractive = term.nonEmpty && term != "dumb"
      val consoleAttached = System.console() != null
      (termLooksInteractive || consoleAttached) && !ci && !noColor

  def create(interactive: Boolean = isInteractiveTerminal): TrainingDisplay =
    if interactive then AnsiEpochBoardDisplay() else PlainLogDisplay()

final class PlainLogDisplay(log: String => Unit = s => println(s)) extends TrainingDisplay:
  override def onTrainingStart(estimateSec: Double): Unit =
    if estimateSec > 0 then
      log(f"Estimated epoch time: ${estimateSec / 60.0}%.1f min (${estimateSec}%.0f sec)")
      if estimateSec > 120 then
        log("  Tip: long epoch detected. Consider chunking input, contextSize<=3, and lower maxVocab for faster iteration.")

  override def onEpochStart(epoch: Int, totalEpochs: Int, lr: Double, totalExamples: Int): Unit =
    log(f"Epoch $epoch/$totalEpochs starting (lr=$lr%.4f, examples=$totalExamples)...")

  override def onBatchProgress(p: BatchProgress): Unit =
    val bar = Trainer.createProgressBar(p.percent)
    log(f"  $bar ${p.percent}%3d%% | ${p.elapsedSec}%.1fs/${p.remainingSec}%.1fs | ${p.examplesPerSec}%.0f ex/s | loss=${p.avgLoss}%.4f")

  override def onEpochComplete(metrics: EpochMetrics, isBest: Boolean, patienceCounter: Int, patience: Int): Unit =
    val bestMarker = if isBest then " | best" else ""
    val patienceInfo = if patience > 0 then s" | patience=${patienceCounter}/${patience}" else ""
    val detail = f"delta=${metrics.bestDeltaPct}%.2f%% gap=${metrics.generalizationGap}%.3f"
    log(
      f"  Epoch ${metrics.epoch}%2d done in ${metrics.epochSeconds}%.1fs | train=${metrics.trainLoss}%.4f val=${metrics.valLoss}%.4f ppl=${metrics.valPerplexity}%.2f | ${metrics.status}: ${metrics.statusReason} | $detail$bestMarker$patienceInfo"
    )

  override def onCancellationRequested(epoch: Int): Unit =
    log(s"\nCancellation requested at epoch $epoch. Wrapping up safely...")

  override def onTrainingComplete(interrupted: Boolean): Unit =
    if interrupted then log("Training stopped by user.")
    else log("Training complete!")

final class AnsiEpochBoardDisplay(
    write: String => Unit = s => print(s),
    log: String => Unit = s => println(s)
) extends TrainingDisplay:
  private val clearLine = "\u001b[2K"
  private var liveLineOpen = false

  override def onTrainingStart(estimateSec: Double): Unit =
    if estimateSec > 0 then
      log(f"Estimated epoch time: ${estimateSec / 60.0}%.1f min (${estimateSec}%.0f sec)")
      if estimateSec > 120 then
        log("  Tip: long epoch detected. Consider chunking input, contextSize<=3, and lower maxVocab for faster iteration.")

  override def onEpochStart(epoch: Int, totalEpochs: Int, lr: Double, totalExamples: Int): Unit =
    liveLineOpen = true
    val line = f"Epoch $epoch/$totalEpochs | lr=$lr%.4f | examples=$totalExamples | starting..."
    write(s"\r$clearLine$line")

  override def onBatchProgress(p: BatchProgress): Unit =
    val bar = Trainer.createProgressBar(p.percent)
    val line =
      f"Epoch ${p.epoch}%2d/${p.totalEpochs}%2d $bar ${p.percent}%3d%% | ${p.elapsedSec}%.1fs/${p.remainingSec}%.1fs | ${p.examplesPerSec}%.0f ex/s | loss=${p.avgLoss}%.4f"
    write(s"\r$clearLine$line")

  override def onEpochComplete(metrics: EpochMetrics, isBest: Boolean, patienceCounter: Int, patience: Int): Unit =
    val bestMarker = if isBest then " | best" else ""
    val patienceInfo = if patience > 0 then s" | patience=${patienceCounter}/${patience}" else ""
    val detail = f"delta=${metrics.bestDeltaPct}%.2f%% gap=${metrics.generalizationGap}%.3f"
    val finalLine =
      f"Epoch ${metrics.epoch}%2d done ${metrics.epochSeconds}%.1fs | train=${metrics.trainLoss}%.4f val=${metrics.valLoss}%.4f ppl=${metrics.valPerplexity}%.2f | ${metrics.status}: ${metrics.statusReason} | $detail$bestMarker$patienceInfo"
    write(s"\r$clearLine$finalLine\n")
    liveLineOpen = false

  override def onCancellationRequested(epoch: Int): Unit =
    if liveLineOpen then write(s"\r$clearLine")
    log(s"Cancellation requested at epoch $epoch. Wrapping up safely...")
    liveLineOpen = false

  override def onTrainingComplete(interrupted: Boolean): Unit =
    if liveLineOpen then write(s"\r$clearLine")
    if interrupted then log("Training stopped by user.")
    else log("Training complete!")
    liveLineOpen = false
