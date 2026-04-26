package sicfun.holdem.runtime

import sicfun.holdem.engine.villain.PlayerArchetype

import java.util.Locale

/** Villain-side configuration types and CLI parsing for the playing hall.
  *
  * Carved out of [[TexasHoldemPlayingHall]] as the second F6 split slice (after
  * [[HallFormat]]). Bundles every villain-related concept that was previously a
  * private member of the 2,500-LOC monolith into a cohesive sibling module:
  *
  *   - [[VillainMode]] / [[VillainProfile]] -- per-villain identity + decision strategy
  *   - [[buildVillainPool]] -- `--villainPool` CLI parsing (comma-separated styles)
  *   - [[parseVillainModeToken]] / [[parseLeakToken]] -- token-by-token parsing
  *   - [[villainModeLabel]] / [[villainModeSlug]] -- display + filename helpers
  *   - [[villainModeOpt]] -- `--villainStyle` option parser used by `parseArgs`
  *
  * Visibility is `private[runtime]` so the rest of the runtime package keeps the
  * same access it had before the move; nothing outside `sicfun.holdem.runtime` ever
  * needed these types.
  */
private[runtime] object HallVillain:
  /** How a villain player makes decisions during the simulation. */
  enum VillainMode:
    case Archetype(style: PlayerArchetype)
    case Gto
    case LeakInjected(leakId: String, severity: Double)

  /** A named villain with a decision mode and a human-readable label for logging. */
  final case class VillainProfile(
      name: String,
      mode: VillainMode,
      label: String
  )

  /** Parses the `--villainPool` option (comma-separated list of villain mode tokens) into a
    * vector of [[VillainProfile]]s. If no pool is specified, creates a single-villain pool from
    * the fallback mode. Each profile gets a unique name like `"Villain01_tag"`.
    */
  def buildVillainPool(
      fallbackMode: VillainMode,
      rawPool: Option[String]
  ): Either[String, Vector[VillainProfile]] =
    rawPool.map(_.trim).filter(_.nonEmpty) match
      case None =>
        Right(Vector(VillainProfile(name = "Villain", mode = fallbackMode, label = villainModeLabel(fallbackMode))))
      case Some(raw) =>
        val tokens = raw.split(",").toVector.map(_.trim).filter(_.nonEmpty)
        if tokens.isEmpty then Left("--villainPool must include at least one style")
        else
          tokens.zipWithIndex.foldLeft[Either[String, Vector[VillainProfile]]](Right(Vector.empty)) {
            case (Left(error), _) => Left(error)
            case (Right(acc), (token, idx)) =>
              parseVillainModeToken(token).left.map(error => s"--villainPool: $error").map { mode =>
                acc :+ VillainProfile(
                  name = f"Villain${idx + 1}%02d_${villainModeSlug(mode)}",
                  mode = mode,
                  label = villainModeLabel(mode)
                )
              }
          }

  /** Maps a single villain-style token (e.g. `"tag"`, `"leak:overfold:0.2"`) to a [[VillainMode]]. */
  def parseVillainModeToken(raw: String): Either[String, VillainMode] =
    raw.trim.toLowerCase match
      case "nit"            => Right(VillainMode.Archetype(PlayerArchetype.Nit))
      case "tag"            => Right(VillainMode.Archetype(PlayerArchetype.Tag))
      case "lag"            => Right(VillainMode.Archetype(PlayerArchetype.Lag))
      case "callingstation" => Right(VillainMode.Archetype(PlayerArchetype.CallingStation))
      case "station"        => Right(VillainMode.Archetype(PlayerArchetype.CallingStation))
      case "maniac"         => Right(VillainMode.Archetype(PlayerArchetype.Maniac))
      case "gto"            => Right(VillainMode.Gto)
      case s if s.startsWith("leak:") =>
        parseLeakToken(s)
      case _ =>
        Left("style must be one of: nit, tag, lag, callingstation, station, maniac, gto, leak:<type>:<severity>")

  /** Parses a `leak:<type>:<severity>` token into a [[VillainMode.LeakInjected]]. */
  def parseLeakToken(raw: String): Either[String, VillainMode] =
    val parts = raw.split(":")
    if parts.length != 3 then Left(s"leak token must be leak:<type>:<severity>, got: $raw")
    else
      val leakType = parts(1)
      val severityStr = parts(2)
      for
        severity <- try Right(severityStr.toDouble) catch case _: NumberFormatException =>
          Left(s"invalid severity: $severityStr")
        leakId <- leakType match
          case "overfold"      => Right("overfold-river-aggression")
          case "overcall"      => Right("overcall-big-bets")
          case "turnbluff"     => Right("overbluff-turn-barrel")
          case "passive"       => Right("passive-big-pots")
          case "prefloploose"  => Right("preflop-too-loose")
          case "prefloptight"  => Right("preflop-too-tight")
          case _               => Left(s"unknown leak type: $leakType (use: overfold, overcall, turnbluff, passive, prefloploose, prefloptight)")
      yield VillainMode.LeakInjected(leakId, severity)

  /** Human-readable label suitable for log/UI display. */
  def villainModeLabel(mode: VillainMode): String =
    mode match
      case VillainMode.Archetype(style)             => style.toString
      case VillainMode.Gto                          => "gto"
      case VillainMode.LeakInjected(leakId, sev)    => s"Leak($leakId@$sev)"

  /** Filename-safe slug used in player names like `"Villain01_<slug>"`. */
  def villainModeSlug(mode: VillainMode): String =
    mode match
      case VillainMode.Archetype(style)             => style.toString.toLowerCase(Locale.ROOT)
      case VillainMode.Gto                          => "gto"
      case VillainMode.LeakInjected(leakId, _)      => leakId.replace("-", "").take(12)

  /** `--villainStyle` option parser used by [[TexasHoldemPlayingHall]]'s `parseArgs`. */
  def villainModeOpt(
      options: Map[String, String],
      key: String,
      default: VillainMode
  ): Either[String, VillainMode] =
    options.get(key) match
      case None => Right(default)
      case Some(_) =>
        parseVillainModeToken(options(key)).left.map(_ =>
          "--villainStyle must be one of: nit, tag, lag, callingstation, station, maniac, gto"
        )
