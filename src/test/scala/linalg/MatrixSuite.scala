package linalg

import munit.FunSuite

class MatrixSuite extends FunSuite:
  test("zeros creates correct shape and values") {
    val m = Matrix.zeros(2, 3)
    assertEquals(m.rows, 2)
    assertEquals(m.cols, 3)
    assert(m.data.forall(_ == 0.0))
  }

  test("fromFunction and get access") {
    val m = Matrix.fromFunction(2, 2)((r, c) => (r + c).toDouble)
    assertEquals(m.get(0, 0), 0.0)
    assertEquals(m.get(1, 1), 2.0)
  }

  test("updated changes one cell") {
    val m = Matrix.zeros(2, 2).updated(1, 1, 4.2)
    assertEquals(m.get(1, 1), 4.2)
    assertEquals(m.get(0, 0), 0.0)
  }

  test("rowSlice and colSlice") {
    val m = Matrix.fromFunction(2, 3)((r, c) => (r * 10 + c).toDouble)
    assertEquals(m.rowSlice(1), Vector(10.0, 11.0, 12.0))
    assertEquals(m.colSlice(2), Vector(2.0, 12.0))
  }

  test("map and zipMap") {
    val a = Matrix.fromFunction(2, 2)((r, c) => (r + c).toDouble)
    val b = Matrix.fromFunction(2, 2)((r, c) => (r * c).toDouble)
    val mapped = a.map(_ + 1.0)
    val zipped = a.zipMap(b)(_ + _)

    assertEquals(mapped.get(1, 1), 3.0)
    assertEquals(zipped.get(1, 1), 3.0)
  }

  test("transposeView does not copy and indexes correctly") {
    val m = Matrix.fromFunction(2, 3)((r, c) => (r * 10 + c).toDouble)
    val t = m.transposeView
    assertEquals(t.rows, 3)
    assertEquals(t.cols, 2)
    assertEquals(t.get(2, 1), 12.0)
    assertEquals(t.transposeView.get(1, 2), 12.0)
  }
