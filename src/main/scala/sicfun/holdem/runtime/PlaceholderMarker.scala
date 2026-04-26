package sicfun.holdem.runtime

trait PlaceholderMarker:
  def placeholderReason: String

object PlaceholderMarker:

  def scanPlaceholders(root: Any): Vector[PlaceholderMarker] =
    val visited = java.util.IdentityHashMap[AnyRef, java.lang.Boolean]()
    val found = collection.mutable.ArrayBuffer.empty[PlaceholderMarker]

    def visit(obj: Any): Unit =
      if obj == null then ()
      else
        val ref = obj.asInstanceOf[AnyRef]
        if visited.containsKey(ref) then ()
        else
          visited.put(ref, java.lang.Boolean.TRUE)
          obj match
            case pm: PlaceholderMarker => found += pm
            case _ => ()
          obj match
            case _: String       => () // stdlib leaf: don't reflect
            case _: Number       => ()
            case _: java.lang.Boolean => ()
            case _: java.lang.Character => ()
            case it: Iterable[?] => it.foreach(visit)
            case arr: Array[?]   => arr.foreach(visit)
            case p: Product      => p.productIterator.foreach(visit)
            case other =>
              if isStdlibClass(other.getClass) then ()
              else
                val cls = other.getClass
                cls.getMethods.foreach { m =>
                  val n = m.getName
                  if m.getParameterCount == 0 &&
                    !n.startsWith("$") &&
                    !n.contains("$$") &&
                    !isJavaLangObjectMethod(n) &&
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

  private def isStdlibClass(cls: Class[?]): Boolean =
    val name = cls.getName
    name.startsWith("java.") ||
      name.startsWith("javax.") ||
      name.startsWith("sun.") ||
      name.startsWith("jdk.") ||
      name.startsWith("scala.")
