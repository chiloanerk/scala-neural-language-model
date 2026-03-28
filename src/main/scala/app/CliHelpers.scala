package app

import java.nio.file.Path

object CliHelpers:
  final case class ParsedValue[A](value: A, invalidInput: Boolean)

  def trimOrEmpty(raw: String): String =
    Option(raw).getOrElse("").trim

  def isTruthy(raw: String): Boolean =
    Set("1", "true", "yes", "y").contains(trimOrEmpty(raw).toLowerCase)

  def parseMenuChoice(raw: String, optionCount: Int, defaultIndex: Int): Option[Int] =
    if optionCount <= 0 || defaultIndex < 0 || defaultIndex >= optionCount then None
    else
      val normalized = trimOrEmpty(raw)
      val selected =
        if normalized.isEmpty then defaultIndex
        else normalized.toIntOption.map(_ - 1).getOrElse(defaultIndex)

      if selected >= 0 && selected < optionCount then Some(selected) else None

  def parseYesNo(raw: String, default: Boolean): Boolean =
    val normalized = trimOrEmpty(raw).toLowerCase
    if normalized.isEmpty then default
    else
      normalized match
        case "y" | "yes" | "1" | "true"  => true
        case "n" | "no" | "0" | "false" => false
        case _                                 => default

  /**
    * Parses CLI args with support for:
    * - key/value: --input file.txt
    * - boolean switch: --yes  (becomes "true")
    * Repeated keys use "last wins".
    */
  def parseArgs(args: Array[String]): Map[String, String] =
    val out = scala.collection.mutable.LinkedHashMap.empty[String, String]
    var i = 0
    while i < args.length do
      val token = args(i)
      if token.startsWith("--") && token.length > 2 then
        val key = token.stripPrefix("--")
        val hasValue = i + 1 < args.length && !args(i + 1).startsWith("--")
        if hasValue then
          out.update(key, args(i + 1))
          i += 2
        else
          out.update(key, "true")
          i += 1
      else
        i += 1
    out.toMap

  /**
    * Confirmation behavior:
    * - autoConfirm=true => proceed
    * - missing input (EOF/non-interactive) => cancel
    * - otherwise parse yes/no with default
    */
  def resolveConfirmation(raw: String | Null, default: Boolean, autoConfirm: Boolean): Boolean =
    if autoConfirm then true
    else if raw == null then false
    else parseYesNo(raw, default)

  def looksLikeDerivedTextFile(path: Path): Boolean =
    val name = path.getFileName.toString.toLowerCase
    name.contains("vocab") || name.contains("token") || name.contains("chunk") || name.contains("part")

  def classifyTrainingFiles(files: Vector[Path]): (Vector[Path], Vector[Path]) =
    files.partition(p => !looksLikeDerivedTextFile(p))

  def recommendChunkSize(totalLines: Int): Int =
    if totalLines > 10000 then 2000
    else if totalLines > 5000 then 1000
    else 500

  def boundedTopK(requested: Int, default: Int = 5, min: Int = 1, max: Int = 50): Int =
    val base = if requested <= 0 then default else requested
    math.max(min, math.min(max, base))

  def boundedIntOrDefault(raw: Option[String], default: Int, min: Int, max: Int): ParsedValue[Int] =
    require(min <= max, s"min must be <= max, got min=$min max=$max")
    raw match
      case None => ParsedValue(default, invalidInput = false)
      case Some(text) =>
        text.trim.toIntOption match
          case Some(v) if v >= min && v <= max => ParsedValue(v, invalidInput = false)
          case _                                => ParsedValue(default, invalidInput = true)

  def boundedDoubleOrDefault(raw: Option[String], default: Double, minInclusive: Double, maxInclusive: Double): ParsedValue[Double] =
    require(minInclusive <= maxInclusive, s"minInclusive must be <= maxInclusive, got min=$minInclusive max=$maxInclusive")
    raw match
      case None => ParsedValue(default, invalidInput = false)
      case Some(text) =>
        text.trim.toDoubleOption match
          case Some(v) if v >= minInclusive && v <= maxInclusive => ParsedValue(v, invalidInput = false)
          case _                                                  => ParsedValue(default, invalidInput = true)
