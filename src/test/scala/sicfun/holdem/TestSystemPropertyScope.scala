package sicfun.holdem

/** Serializes temporary JVM system-property overrides across test suites.
  *
  * System properties are process-global mutable state. Running property-mutating
  * tests in parallel can cause cross-suite interference and flaky assertions.
  */
object TestSystemPropertyScope:
  private val lock = new AnyRef

  def withSystemProperties[A](updates: Seq[(String, Option[String])])(thunk: => A): A =
    lock.synchronized {
      val previous = updates.map { case (key, _) => key -> sys.props.get(key) }.toMap
      updates.foreach {
        case (key, Some(value)) => sys.props.update(key, value)
        case (key, None) => sys.props.remove(key)
      }
      try thunk
      finally
        previous.foreach {
          case (key, Some(value)) => sys.props.update(key, value)
          case (key, None) => sys.props.remove(key)
        }
    }
