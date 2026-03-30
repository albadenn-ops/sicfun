package sicfun.holdem.types

/** Scoped runtime-property overrides for tests and local tooling.
  *
  * Overrides are stored in an inheritable thread-local snapshot so child worker
  * threads created inside the scope observe the same effective values without
  * mutating process-global JVM system properties.
  */
object ScopedRuntimeProperties:
  private val overridesRef = new InheritableThreadLocal[List[Map[String, Option[String]]]]:
    override def initialValue(): List[Map[String, Option[String]]] = Nil
    override def childValue(parentValue: List[Map[String, Option[String]]]): List[Map[String, Option[String]]] =
      parentValue

  def withOverrides[A](updates: Seq[(String, Option[String])])(thunk: => A): A =
    val previous = overridesRef.get()
    val merged = updates.foldLeft(previous.headOption.getOrElse(Map.empty[String, Option[String]])) {
      case (acc, (key, value)) => acc.updated(key, value)
    }
    overridesRef.set(merged :: previous)
    try thunk
    finally
      previous match
        case Nil => overridesRef.remove()
        case _ => overridesRef.set(previous)

  def get(property: String): Option[Option[String]] =
    overridesRef.get().headOption.flatMap(_.get(property))
