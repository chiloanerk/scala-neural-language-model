package app

import compute.BackendSelector
import data.{TextPipeline, VocabIO}
import eval.Metrics
import linalg.LinearAlgebra
import nn.{LanguageModel, ModelConfig}
import train.{CheckpointIO, TrainConfig, Trainer}

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.io.StdIn
import scala.jdk.CollectionConverters._
import scala.util.Using

object Main:

  // Persistent model paths
  val ModelPath = Path.of("data/models/latest.ckpt")
  val VocabPath = Path.of("data/models/latest.vocab")

  // Training presets
  case class Preset(
      name: String,
      epochs: Int,
      patience: Int,
      hiddenDim: Int,
      embedDim: Int,
      learningRate: Double,
      description: String
  )

  val presets = Vector(
    Preset("quick", 5, 3, 32, 16, 0.05, "Test (~30 sec)"),
    Preset("balanced", 20, 5, 64, 24, 0.02, "Standard (~2 min)"),
    Preset("thorough", 50, 10, 128, 48, 0.01, "Best quality (~10 min)")
  )

  private def readLineWithPrompt(prompt: String): String | Null =
    // sbt forked runs can buffer non-newline output; print prompt as its own line.
    println(prompt)
    Console.out.flush()
    StdIn.readLine()

  private def defaultPrecisionForBackend(backend: String): String =
    "fp64"

  private[app] def benchmarkMatrix(backendFlag: Option[String], precisionFlag: Option[String]): Vector[(String, String)] =
    val backends = backendFlag match
      case Some(b) => Vector(BackendSelector.normalizeBackend(b))
      case None    => Vector("cpu", "gpu")
    val precisions = precisionFlag match
      case Some(p) => Vector(BackendSelector.normalizePrecision(p))
      case None    => Vector("fp64", "fp32")
    for
      b <- backends
      p <- precisions
    yield (b, p)

  private def readTrimmedRequired(prompt: String, field: String): String =
    Option(readLineWithPrompt(prompt)) match
      case Some(v) => v.trim
      case None =>
        println(s"\nNo interactive input available for '$field'.")
        println("Run with explicit flags for non-interactive mode (and add --yes to auto-confirm).")
        sys.exit(1)
        ""

  def main(args: Array[String]): Unit =
    if args.isEmpty then
      showMainMenu()
      sys.exit(0)

    args(0) match
      case "train"   => runTrain(CliHelpers.parseArgs(args.drop(1)))
      case "predict" => runPredict(CliHelpers.parseArgs(args.drop(1)))
      case "chunk"   => runChunker(CliHelpers.parseArgs(args.drop(1)))
      case "gpu-info" => runGpuInfo(CliHelpers.parseArgs(args.drop(1)))
      case "benchmark" => runBenchmark(CliHelpers.parseArgs(args.drop(1)))
      case "test"    => TestRunner.main(Array.empty)
      case _ =>
        printUsage()
        sys.exit(1)

  private def showMainMenu(): Unit =
    val modelStatus = if Files.isRegularFile(ModelPath) then
      val size = Files.size(ModelPath) / 1024
      s"✓ Model exists (${size} KB)"
    else "✗ No model yet"

    println(
      s"""
      |=== Scala Neural Language Model (NLM) ===
      |
      |Status: $modelStatus
      |
      |Commands:
      |  train   - Train or continue training your model
      |  predict - Predict next words
      |  chunk   - Split large files into training chunks
      |  gpu-info - Check Metal/JNI status
      |  benchmark - Compare CPU/GPU throughput estimate
      |  test    - Run built-in tests
      |
      |Quick start:
      |  sbt "runMain app.Main train --input data/corpus/text.txt"
      |  sbt "runMain app.Main predict --context 'the cat'"
      |  sbt "runMain app.Main chunk --input large-file.txt --lines 1000"
      |  sbt "runMain app.Main gpu-info --precision fp32"
      |""".stripMargin)

  private def runTrain(flags: Map[String, String]): Unit =
    println("\n=== Training ===\n")

    // Check for existing model
    val freshTraining = flags.get("fresh").exists(CliHelpers.isTruthy)
    val hasExistingModel = Files.isRegularFile(ModelPath)

    // Ask if user wants fresh start when model exists AND no flag provided
    val actuallyFresh = if hasExistingModel && !freshTraining && !flags.contains("input") then
      println(s"Found existing model: $ModelPath (${Files.size(ModelPath) / 1024} KB)")
      println()
      println("Choose action:")
      println("  1. Continue training (default)")
      println("  2. Start fresh (new model, ignores existing)")
      println()
      val raw = readTrimmedRequired("Select [1]: ", "select action")
      CliHelpers.parseMenuChoice(raw, optionCount = 2, defaultIndex = 0).contains(1)
    else if freshTraining then
      println("Starting fresh training (ignoring any existing model).\n")
      true
    else if hasExistingModel then
      val size = Files.size(ModelPath) / 1024
      println(s"Found existing model: $ModelPath (${size} KB)")
      println("This will continue training from where you left off.\n")
      false
    else
      println("No existing model found. Starting fresh training.\n")
      false

    // Get input file
    val inputPath = flags.get("input") match
      case Some(path) => Path.of(path)
      case None =>
        // Discover text files
        val candidates = discoverTextFiles(Path.of(".").toAbsolutePath.normalize, maxDepth = 6)
          .filter(p => !p.toString.contains("/chunks/")) // Skip chunked files
        
        // Count lines for each file (with fallback for encoding issues)
        val candidatesWithLines = candidates.map { p =>
          val lines = countLines(p)
          val size = Files.size(p) / 1024
          (p, lines, size)
        }.filter { case (_, lines, size) => lines > 0 || size > 0 } // Keep files with lines OR size > 0
        
        if candidatesWithLines.isEmpty then
          println("Error: No readable .txt files found. Specify --input <file.txt>")
          sys.exit(1)
        
        val (recommended, other) = CliHelpers.classifyTrainingFiles(candidatesWithLines.map(_._1))
        val orderedPaths = recommended ++ other
        val byPath = candidatesWithLines.map { case (p, lines, size) => p -> (lines, size) }.toMap

        println("Found text files (excluding chunks):")
        if recommended.nonEmpty then println("  Recommended training files:")
        recommended.zipWithIndex.foreach { case (p, idx) =>
          val (lines, size) = byPath(p)
          val linesStr = if lines > 0 then s"$lines lines" else "unknown lines"
          println(s"    ${idx + 1}. $p ($linesStr, ${size} KB)")
        }
        if other.nonEmpty then println("  Other text files (likely derived; usually avoid):")
        other.zipWithIndex.foreach { case (p, idx) =>
          val (lines, size) = byPath(p)
          val linesStr = if lines > 0 then s"$lines lines" else "unknown lines"
          println(s"    ${recommended.length + idx + 1}. $p ($linesStr, ${size} KB)")
        }
        println()
        
        var selected: Option[Path] = None
        while selected.isEmpty do
          val raw = readTrimmedRequired(s"Choose file number [1]: ", "choose training file")
          CliHelpers.parseMenuChoice(raw, optionCount = orderedPaths.length, defaultIndex = 0) match
            case Some(idx) => selected = Some(orderedPaths(idx))
            case None      => println(s"Please choose 1-${orderedPaths.length}.")
        selected.get

    require(Files.isRegularFile(inputPath), s"Input file not found: $inputPath")

    // Check if we can reuse existing model's architecture
    val existingCfg = if hasExistingModel && !actuallyFresh && Files.isRegularFile(ModelPath) then
      Some(CheckpointIO.load(ModelPath)._2)
    else
      None

    // Get preset
    val presetName = flags.get("preset").getOrElse {
      println("\nChoose training quality:")
      presets.zipWithIndex.foreach { case (p, idx) =>
        println(s"  ${idx + 1}. ${p.name.capitalize} - ${p.description}")
      }
      val raw = readTrimmedRequired("\nSelect preset [2]: ", "select preset")
      val idx = CliHelpers.parseMenuChoice(raw, optionCount = presets.length, defaultIndex = 1).getOrElse(1)
      presets(idx).name
    }
    val preset = presets.find(_.name == presetName).getOrElse {
      println(s"Unknown preset '$presetName'. Using 'balanced'.")
      presets(1)
    }
    val backend = BackendSelector.normalizeBackend(flags.getOrElse("backend", "gpu"))
    val precision = BackendSelector.normalizePrecision(flags.getOrElse("precision", defaultPrecisionForBackend(backend)))
    val learningRate = flags.get("lr").flatMap(_.toDoubleOption).getOrElse(preset.learningRate)
    val lrDecay = flags.get("lrDecay").flatMap(_.toDoubleOption).getOrElse(1.0)
    val batchSize = flags.get("batchSize").flatMap(_.toIntOption).getOrElse(0)
    val prefetch = flags.get("prefetch").flatMap(_.toIntOption).getOrElse(1)
    val profileGpu = flags.get("profileGpu").exists(CliHelpers.isTruthy)
    if flags.get("gpuInfo").exists(CliHelpers.isTruthy) then
      println(s"GPU info: ${BackendSelector.gpuInfo(precision)}")

    // Get advanced options - use existing model's architecture if available
    val contextSize = flags.get("contextSize").map(_.toInt).getOrElse {
      existingCfg match
        case Some(cfg) =>
          println(s"\nUsing existing model architecture: context=${cfg.contextSize}")
          cfg.contextSize
        case None =>
          val raw = readTrimmedRequired("\nContext size (words to look back) [3]: ", "context size")
          raw.toIntOption.getOrElse(3)
    }
    val maxVocab = flags.get("maxVocab").map(_.toInt).getOrElse {
      existingCfg match
        case Some(cfg) =>
          println(s"Using existing model vocabulary size: ${cfg.vocabSize}")
          cfg.vocabSize
        case None =>
          val raw = readTrimmedRequired("Max vocabulary (unique words) [3000]: ", "max vocabulary")
          raw.toIntOption.getOrElse(3000)
    }

    // Ensure output directory exists
    Files.createDirectories(ModelPath.getParent)

    // Show summary and confirm
    println(s"\n=== Summary ===")
    println(s"  Input: $inputPath")
    println(s"  Preset: ${preset.name.capitalize} (${preset.epochs} epochs, patience=${preset.patience})")
    println(s"  Context size: $contextSize")
    println(s"  Max vocab: $maxVocab")
    println(s"  Backend: $backend ($precision)")
    println(f"  Learning rate: $learningRate%.4f (decay=$lrDecay%.4f)")
    if batchSize > 0 then println(s"  Batch size: $batchSize")
    if profileGpu then println("  GPU profile: enabled")
    if actuallyFresh then
      println(s"  🆕 Starting fresh model (new architecture)")
    else if existingCfg.isDefined then
      println(s"  ✓ Continuing from existing model")
    else
      println(s"  🆕 Starting fresh model")
    println()

    val autoConfirm = flags.get("yes").exists(CliHelpers.isTruthy)
    val confirm = if autoConfirm then
      println("Auto-confirming (--yes flag)")
      true
    else
      promptYesNo("Start training?", true)
    
    if !confirm then
      println("Training canceled.")
      sys.exit(0)

    println()

    // Load or create model - try multiple encodings
    val rawText = 
      try
        Files.readString(inputPath, StandardCharsets.UTF_8)
      catch
        case _: Exception =>
          println("UTF-8 read failed, trying with system default encoding...")
          // Use BufferedReader with default charset which is more lenient
          import java.io._
          val reader = new BufferedReader(new InputStreamReader(Files.newInputStream(inputPath)))
          try
            val text = new StringBuilder
            var line = reader.readLine()
            while line != null do
              text.append(line).append("\n")
              line = reader.readLine()
            text.toString
          finally
            reader.close()
    val tokens = TextPipeline.tokenize(rawText)
    
    val vocab = if actuallyFresh || !Files.isRegularFile(VocabPath) then
      TextPipeline.buildVocab(tokens, maxVocab)
    else
      VocabIO.load(VocabPath)

    val ids = TextPipeline.tokensToIds(tokens, vocab)
    val examples = TextPipeline.buildExamples(ids, contextSize)
    val (trainSet, valSet) = TextPipeline.splitDeterministic(examples, 0.9, 42)

    require(trainSet.nonEmpty, "Training set is empty. Provide more text or reduce contextSize.")

    val cfg = ModelConfig(
      contextSize = contextSize,
      embedDim = preset.embedDim,
      hiddenDim = preset.hiddenDim,
      vocabSize = vocab.size,
      activation = "tanh"
    )

    val params0 = if hasExistingModel && !actuallyFresh && Files.isRegularFile(ModelPath) then
      val (existingParams, existingCfg) = CheckpointIO.load(ModelPath)
      println(s"Loaded existing model (vocab: ${existingCfg.vocabSize} words, context: ${existingCfg.contextSize}, embed: ${existingCfg.embedDim}, hidden: ${existingCfg.hiddenDim})")
      existingParams
    else
      println("Initializing new model...")
      LanguageModel.initParams(cfg, 42)

    val trainCfg = TrainConfig(
      epochs = preset.epochs,
      learningRate = learningRate,
      lrDecay = lrDecay,
      l2 = 0.0,
      clipNorm = None,
      shuffleEachEpoch = true,
      seed = 42,
      patience = preset.patience,
      activation = "tanh",
      backend = backend,
      precision = precision,
      batchSize = batchSize,
      prefetch = prefetch,
      profileGpu = profileGpu
    )

    // Train
    val result = Trainer.train(params0, trainSet, valSet, trainCfg)

    // Save
    CheckpointIO.save(result.params, cfg, ModelPath)
    VocabIO.save(vocab, VocabPath)

    println("\n=== Results ===")
    val bestMetric = result.history.minBy(_.valLoss)
    val finalMetric = if trainCfg.patience > 0 then bestMetric else result.history.last
    val restoredNote = if trainCfg.patience > 0 then " (restored best)" else ""
    result.history.foreach { m =>
      val indicator = if m.epoch == bestMetric.epoch then " ← best" else ""
      println(f"  Epoch ${m.epoch}%2d: train=${m.trainLoss}%.4f val=${m.valLoss}%.4f ppl=${m.valPerplexity}%.1f$indicator")
    }

    println(s"\n✓ Model saved to $ModelPath")
    println(s"  Vocab saved to $VocabPath")
    println(f"  Final$restoredNote: val_loss=${finalMetric.valLoss}%.4f val_ppl=${finalMetric.valPerplexity}%.1f")
    if trainCfg.patience > 0 then
      val last = result.history.last
      println(f"  Last epoch (pre-restore): val_loss=${last.valLoss}%.4f val_ppl=${last.valPerplexity}%.1f")
    println()
    println("Continue with: sbt \"runMain app.Main train --input <more-data.txt>\"")
    println("Or predict:    sbt \"runMain app.Main predict --context 'your text'\"")

  private def runChunker(flags: Map[String, String]): Unit =
    println("\n=== File Chunker ===\n")

    // Get input file
    val inputPath = flags.get("input") match
      case Some(path) => Path.of(path)
      case None =>
        // Discover text files
        val candidates = discoverTextFiles(Path.of(".").toAbsolutePath.normalize, maxDepth = 6)
        if candidates.isEmpty then
          println("No .txt files found. Specify --input <file.txt>")
          sys.exit(1)
        
        println("Found text files:")
        candidates.zipWithIndex.foreach { case (p, idx) =>
          val lines = countLines(p)
          val size = Files.size(p) / 1024
          println(s"  ${idx + 1}. $p (${lines} lines, ${size} KB)")
        }
        println()
        
        var selected: Option[Path] = None
        while selected.isEmpty do
          val raw = readTrimmedRequired("Choose file number: ", "choose file number")
          CliHelpers.parseMenuChoice(raw, optionCount = candidates.length, defaultIndex = 0) match
            case Some(idx) => selected = Some(candidates(idx))
            case None      => println(s"Please choose 1-${candidates.length}.")
        selected.get

    require(Files.isRegularFile(inputPath), s"Input file not found: $inputPath")
    
    val totalLines = countLines(inputPath)
    val sizeKB = Files.size(inputPath) / 1024
    println(s"\nSelected: $inputPath")
    println(s"Size: ${totalLines} lines, ${sizeKB} KB\n")

    // Get chunk size
    val chunkSize = flags.get("lines").map(_.toInt).getOrElse {
      // Recommend based on file size
      val recommended = CliHelpers.recommendChunkSize(totalLines)
      println(s"Recommended chunk size: $recommended lines")
      println(s"  - This will create ${(totalLines + recommended - 1) / recommended} chunks\n")
      
      var size: Option[Int] = None
      while size.isEmpty do
        val raw = readTrimmedRequired(s"Lines per chunk [$recommended]: ", "lines per chunk")
        size = if raw.isEmpty then Some(recommended) else raw.toIntOption
        if size.isEmpty then println("Please enter a number.")
      size.get
    }

    // Get output directory
    val outputDir = flags.get("output") match
      case Some(path) => Path.of(path)
      case None =>
        val parent = Option(inputPath.getParent).getOrElse(Path.of("."))
        parent.resolve("chunks")

    Files.createDirectories(outputDir)

    // Get base name for chunks
    val baseName = flags.get("name").getOrElse {
      inputPath.getFileName.toString.replace(".txt", "")
    }

    println(s"\nChunking plan:")
    println(s"  Input: $inputPath (${totalLines} lines)")
    println(s"  Output: $outputDir/")
    println(s"  Chunk size: $chunkSize lines")
    val autoConfirm = flags.get("yes").exists(CliHelpers.isTruthy)
    
    println(s"  Files: ${baseName}-part1.txt, ${baseName}-part2.txt, ...")
    println()

    val confirm = if autoConfirm then
      println("Auto-confirming (--yes flag)")
      true
    else
      promptYesNo("Create chunks?", true)
      
    if !confirm then
      println("Canceled.")
      sys.exit(0)

    // Read and split
    val lines = Files.readAllLines(inputPath, StandardCharsets.UTF_8)
    val totalChunks = (lines.size() + chunkSize - 1) / chunkSize

    println(s"\nCreating $totalChunks chunks...")

    var chunkNum = 1
    var lineIdx = 0
    while lineIdx < lines.size() do
      val endIdx = math.min(lineIdx + chunkSize, lines.size())
      val chunkLines = lines.subList(lineIdx, endIdx)
      
      val chunkPath = outputDir.resolve(s"${baseName}-part$chunkNum.txt")
      Files.write(chunkPath, chunkLines, StandardCharsets.UTF_8)
      
      val chunkLinesCount = endIdx - lineIdx
      println(s"  ✓ ${chunkPath.getFileName} ($chunkLinesCount lines)")
      
      chunkNum += 1
      lineIdx = endIdx

    println(s"\n✓ Created $totalChunks chunks in $outputDir/")
    println()
    println("Next steps:")
    println(s"  sbt \"runMain app.Main train --input $outputDir/${baseName}-part1.txt --preset quick\"")
    println(s"  sbt \"runMain app.Main train --input $outputDir/${baseName}-part2.txt --preset quick\"")
    println("  ... continue with remaining parts")

  private def runPredict(flags: Map[String, String]): Unit =
    println("\n=== Prediction ===\n")

    // Check model exists
    if !Files.isRegularFile(ModelPath) then
      println("No model found. Train first:")
      println("  sbt \"runMain app.Main train --input data/corpus/text.txt\"")
      sys.exit(1)

    val (params, cfg) = CheckpointIO.load(ModelPath)
    val vocab = VocabIO.load(VocabPath)
    val backend = BackendSelector.fromConfig(
      flags.getOrElse("backend", "gpu"),
      flags.getOrElse("precision", defaultPrecisionForBackend(flags.getOrElse("backend", "gpu"))),
      warn = msg => println(s"[backend] $msg")
    )

    println(s"Model: ${vocab.size} words, context=${cfg.contextSize}")
    println()

    // Get context
    val contextText = flags.get("context") match
      case Some(text) => text
      case None =>
        println("Enter text to predict next word (or 'quit'):")
        val raw = StdIn.readLine("> ")
        if raw == null || raw.trim.toLowerCase == "quit" then sys.exit(0)
        raw.trim

    val topK = CliHelpers.boundedTopK(flags.get("topK").flatMap(_.toIntOption).getOrElse(5))

    val tokens = TextPipeline.tokenize(contextText)
    val contextIds = adaptContext(tokens.map(vocab.toId), cfg.contextSize, vocab.unkId)

    val cache = LanguageModel.forward(params, contextIds, cfg.activation, backend)
    val top = LinearAlgebra.argTopK(cache.probs, topK)

    println("\nTop predictions:")
    top.zipWithIndex.foreach { case ((id, prob), idx) =>
      val word = vocab.toToken(id)
      val bar = "█" * ((prob * 20).toInt max 1)
      println(f"  ${idx + 1}. $word%-12s $bar%-20s ${prob * 100}%.1f%%")
    }

  private def runGpuInfo(flags: Map[String, String]): Unit =
    val backendName = BackendSelector.normalizeBackend(flags.getOrElse("backend", "gpu"))
    val precision = BackendSelector.normalizePrecision(flags.getOrElse("precision", defaultPrecisionForBackend(backendName)))
    println(s"\nGPU probe ($precision): ${BackendSelector.gpuInfo(precision)}")

  private def runBenchmark(flags: Map[String, String]): Unit =
    println("\n=== Benchmark ===")
    val inputPath = Path.of(flags.getOrElse("input", "data/corpus/example-corpus.txt"))
    if !Files.isRegularFile(inputPath) then
      println(s"Input file not found: $inputPath")
      sys.exit(1)

    val contextSize = flags.get("contextSize").flatMap(_.toIntOption).getOrElse(3)
    val maxVocab = flags.get("maxVocab").flatMap(_.toIntOption).getOrElse(3000)
    val sample = flags.get("sample").flatMap(_.toIntOption).getOrElse(2000)
    val benchmarkBackendFlag = flags.get("backend")
    val precisionFlag = flags.get("precision")
    val activation = flags.getOrElse("activation", "tanh")
    val batchSize = flags.get("batchSize").flatMap(_.toIntOption).getOrElse(0)

    val raw = Files.readString(inputPath, StandardCharsets.UTF_8)
    val tokens = TextPipeline.tokenize(raw)
    val vocab = TextPipeline.buildVocab(tokens, maxVocab)
    val ids = TextPipeline.tokensToIds(tokens, vocab)
    val examples = TextPipeline.buildExamples(ids, contextSize).take(sample)
    if examples.isEmpty then
      println("No examples to benchmark. Adjust context/sample.")
      return

    val cfg = ModelConfig(contextSize = contextSize, embedDim = 24, hiddenDim = 64, vocabSize = vocab.size, activation = activation)
    val params = LanguageModel.initParams(cfg, seed = 42)

    def bench(label: String, backendName: String, precision: String): Double =
      val backend = BackendSelector.fromConfig(backendName, precision, warn = _ => ())
      backend.resetProfile()
      val trainCfg = TrainConfig(epochs = 1, activation = activation, backend = backendName, precision = precision, batchSize = batchSize, profileGpu = true)
      val sec = Trainer.estimateEpochSeconds(params, examples, trainCfg, backend, sampleSize = math.min(500, examples.length))
      val exPerSec = if sec <= 0 then 0.0 else examples.length / sec
      println(f"$label%-4s backend=${backend.diagnostics} | est=${sec}%.2fs for ${examples.length} ex | ${exPerSec}%.1f ex/s")
      if backend.isGpu then println(s"     profile=${backend.profileSummary}")
      exPerSec

    val results = scala.collection.mutable.Map.empty[(String, String), Double]
    val combos = benchmarkMatrix(benchmarkBackendFlag, precisionFlag)
    combos.foreach { case (backendName, precision) =>
      val label = s"${backendName.toUpperCase}[$precision]"
      val exPerSec = bench(label, backendName, precision)
      results((backendName, precision)) = exPerSec
    }

    Vector("fp64", "fp32").foreach { precision =>
      val cpu = results.get(("cpu", precision))
      val gpu = results.get(("gpu", precision))
      (cpu, gpu) match
        case (Some(c), Some(g)) if c > 0 && g > 0 =>
          println(f"Speedup (GPU/CPU, $precision): ${g / c}%.2fx")
        case _ => ()
    }

    val opPrecisions = precisionFlag.map(BackendSelector.normalizePrecision).map(Vector(_)).getOrElse(Vector("fp64", "fp32"))
    opPrecisions.foreach { precision =>
      println(s"\nPer-op timing (synthetic, $precision)")
      def benchOps(label: String, backendName: String): Unit =
        val backend = BackendSelector.fromConfig(backendName, precision, warn = _ => ())
        val rows = 256
        val cols = 256
        val m = linalg.Matrix.fromFunction(rows, cols)((r, c) => ((r + c) % 17) * 0.01)
        val x = Vector.tabulate(cols)(i => (i % 13) * 0.01)
        val b = Vector.tabulate(rows)(i => (i % 7) * 0.01)
        val a = Vector.tabulate(rows)(i => (i % 11) * 0.01)
        val o = Vector.tabulate(cols)(i => (i % 5) * 0.01)
        val iters = 100

        def timeMs(fn: => Unit): Double =
          val t0 = System.nanoTime()
          var i = 0
          while i < iters do
            fn
            i += 1
          (System.nanoTime() - t0).toDouble / 1e6

        val matVecMs = timeMs { backend.matVecMul(m, x); () }
        val outerMs = timeMs { backend.outer(a, o); () }
        val fusedMs = timeMs { backend.linearActivation(m, x, b, activation); () }
        println(f"$label%-4s matVec=${matVecMs / iters}%.3fms outer=${outerMs / iters}%.3fms fused=${fusedMs / iters}%.3fms")

      val opBackends = benchmarkBackendFlag.map(BackendSelector.normalizeBackend).map(Vector(_)).getOrElse(Vector("cpu", "gpu"))
      opBackends.foreach { b =>
        benchOps(b.toUpperCase, b)
      }
    }

  private def adaptContext(ids: Vector[Int], contextSize: Int, padId: Int): Vector[Int] =
    if ids.length >= contextSize then ids.takeRight(contextSize)
    else Vector.fill(contextSize - ids.length)(padId) ++ ids

  private def countLines(path: Path): Int =
    try
      val lines = Files.readAllLines(path, StandardCharsets.UTF_8)
      lines.size()
    catch
      case _: Exception =>
        // Try with default charset, or return 0 if that fails too
        try
          val lines = Files.readAllLines(path)
          lines.size()
        catch
          case _: Exception => 0

  private def discoverTextFiles(root: Path, maxDepth: Int = 6, maxResults: Int = 20): Vector[Path] =
    if !Files.exists(root) then Vector.empty
    else
      Using.resource(Files.walk(root, maxDepth)) { stream =>
        stream.iterator().asScala
          .filter(Files.isRegularFile(_))
          .filter(p => p.getFileName.toString.toLowerCase.endsWith(".txt"))
          .toVector
          .sortBy(_.toString)
          .take(maxResults)
      }

  private def promptYesNo(label: String, default: Boolean): Boolean =
    val defaultStr = if default then "Y/n" else "y/N"
    val raw = readLineWithPrompt(s"$label [$defaultStr]: ")
    val resolved = CliHelpers.resolveConfirmation(raw, default, autoConfirm = false)
    if raw == null then
      println("\nNo interactive input available for confirmation; canceling for safety.")
    resolved

  private def printUsage(): Unit =
    println(
      """Usage:
        |
        |  sbt "runMain app.Main"              Show this menu
        |  sbt "runMain app.Main train"        Train/continue training
        |  sbt "runMain app.Main predict"      Predict next words
        |  sbt "runMain app.Main chunk"        Split large files into chunks
        |  sbt "runMain app.Main gpu-info"     Show Metal/JNI status
        |  sbt "runMain app.Main benchmark"    Compare CPU/GPU throughput estimate
        |
        |Training:
        |  --input FILE      Training text file (required)
        |  --preset NAME     quick/balanced/thorough (default: balanced)
        |  --fresh           Start fresh (ignore existing model)
        |  --contextSize N   Words to look back (default: 3)
        |  --maxVocab N      Max unique words (default: 3000)
        |  --backend NAME    cpu|gpu (default: gpu)
        |  --precision MODE  fp64|fp32 (default: fp64)
        |  --lr VALUE        Learning rate override (default: preset-specific)
        |  --lrDecay VALUE   Learning-rate decay per epoch (default: 1.0)
        |  --batchSize N     Mini-batch size (default: 128 gpu / 32 cpu)
        |  --prefetch N      Prefetch window (1 or 2; default: 1)
        |  --profileGpu      Print per-op GPU profile summary after training
        |  --gpuInfo         Print GPU probe details before training
        |
        |Chunking:
        |  --input FILE      File to split (required, or select interactively)
        |  --lines N         Lines per chunk (default: auto-recommend)
        |  --output DIR      Output directory (default: <input>/chunks/)
        |  --name BASE       Base name for chunks (default: input filename)
        |
        |Prediction:
        |  --context TEXT    Words to continue from
        |  --topK N          Predictions to show (default: 5)
        |  --backend NAME    cpu|gpu (default: gpu)
        |  --precision MODE  fp64|fp32 (default: fp64)

        |GPU:
        |  sbt "runMain app.Main gpu-info --precision fp64"
        |
        |Benchmark:
        |  --input FILE      Corpus text path
        |  --backend NAME    cpu|gpu (default: run both)
        |  --contextSize N   Context size (default: 3)
        |  --maxVocab N      Max vocab (default: 3000)
        |  --sample N        Number of examples (default: 2000)
        |  --precision MODE  fp64|fp32 (default: run both)
        |  --batchSize N     Batch size override for benchmark estimate
        |
        |Examples:
        |  sbt "runMain app.Main train --input data/corpus/text.txt"
        |  sbt "runMain app.Main train --input more.txt --preset quick"
        |  sbt "runMain app.Main chunk --input large-file.txt --lines 1000"
        |  sbt "runMain app.Main predict --context 'the cat sat' --topK 5"
        |""".stripMargin)
