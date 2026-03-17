package linalg

type Vec = Vector[Double]

final case class Matrix(
    data: Vector[Double],
    rows: Int,
    cols: Int,
    transposed: Boolean = false,
    stride: Int = -1
):
  private val internalStride: Int = if stride == -1 then cols else stride

  require(rows >= 0 && cols >= 0, s"rows and cols must be non-negative: rows=$rows cols=$cols")
  require(data.length == rows * cols, s"data length ${data.length} must equal rows*cols ${rows * cols}")

  private def linearIndex(r: Int, c: Int): Int =
    require(r >= 0 && r < rows, s"row out of range: $r for rows=$rows")
    require(c >= 0 && c < cols, s"col out of range: $c for cols=$cols")
    if !transposed then r * internalStride + c else c * internalStride + r

  def get(r: Int, c: Int): Double = data(linearIndex(r, c))

  def updated(r: Int, c: Int, value: Double): Matrix =
    val idx = linearIndex(r, c)
    copy(data = data.updated(idx, value), stride = internalStride)

  def rowSlice(r: Int): Vec = Vector.tabulate(cols)(c => get(r, c))

  def colSlice(c: Int): Vec = Vector.tabulate(rows)(r => get(r, c))

  def map(f: Double => Double): Matrix =
    Matrix.fromFunction(rows, cols)((r, c) => f(get(r, c)))

  def zipMap(other: Matrix)(f: (Double, Double) => Double): Matrix =
    require(rows == other.rows && cols == other.cols, s"shape mismatch: ($rows,$cols) vs (${other.rows},${other.cols})")
    Matrix.fromFunction(rows, cols)((r, c) => f(get(r, c), other.get(r, c)))

  def transposeView: Matrix =
    Matrix(data = data, rows = cols, cols = rows, transposed = !transposed, stride = internalStride)

object Matrix:
  def zeros(rows: Int, cols: Int): Matrix = Matrix(Vector.fill(rows * cols)(0.0), rows, cols)

  def fromFunction(rows: Int, cols: Int)(f: (Int, Int) => Double): Matrix =
    val out = Vector.newBuilder[Double]
    out.sizeHint(rows * cols)
    var r = 0
    while r < rows do
      var c = 0
      while c < cols do
        out += f(r, c)
        c += 1
      r += 1
    Matrix(out.result(), rows, cols)
