package sicfun.holdem.runtime

trait PlaceholderMarker:
  def placeholderReason: String

object PlaceholderMarker:

  def scanPlaceholders(root: Any): Vector[PlaceholderMarker] =
    val visited = collection.mutable.Set.empty[Any]
    val found = collection.mutable.ArrayBuffer.empty[PlaceholderMarker]

    def visit(obj: Any): Unit =
      if obj == null then ()
      else if visited.exists(_.asInstanceOf[AnyRef] eq obj.asInstanceOf[AnyRef]) then ()
      else
        visited += obj
        obj match
          case pm: PlaceholderMarker => found += pm
          case _ => ()
        obj match
          case it: Iterable[?] => it.foreach(visit)
          case arr: Array[?]   => arr.foreach(visit)
          case p: Product      => p.productIterator.foreach(visit)
          case other =>
            val cls = other.getClass
            cls.getMethods.foreach { m =>
              if m.getParameterCount == 0 &&
                !m.getName.startsWith("$") &&
                !isJavaLangObjectMethod(m.getName) &&
                m.getReturnType != classOf[Unit] &&
                m.getReturnType != java.lang.Void.TYPE
              then
                try
                  val v = m.invoke(other)
                  if v != other then visit(v)
                catch case _: Throwable => ()
            }

    visit(root)
    found.toVector

  private def isJavaLangObjectMethod(name: String): Boolean =
    Set("hashCode", "toString", "getClass", "wait", "notify", "notifyAll", "clone").contains(name)
