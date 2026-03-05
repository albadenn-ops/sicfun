package sicfun.holdem

import sicfun.core.Card

/** Typed commands for the interactive poker advisor REPL. */
enum AdvisorCommand:
  case NewHand
  case HeroCards(cards: sicfun.holdem.HoleCards)
  case HeroAction(action: PokerAction)
  case VillainAction(action: PokerAction)
  case DealBoard(cards: Vector[Card])
  case Advise
  case Review
  case SessionStats
  case Undo
  case Help
  case Quit
  case Unknown(input: String, reason: String)

/** Parses raw REPL input lines into typed [[AdvisorCommand]] values.
  *
  * Grammar:
  *   new                              → NewHand
  *   hero AcKh / h AcKh              → HeroCards
  *   h raise 6 / h call / h fold     → HeroAction
  *   v bet 8 / v call / v raise 20   → VillainAction (bet = raise)
  *   board Ts9h8d / board Ts 9h 8d   → DealBoard
  *   ? / advise                       → Advise
  *   review                           → Review
  *   session / stats                  → SessionStats
  *   undo                             → Undo
  *   help                             → Help
  *   quit / exit / q                  → Quit
  */
object AdvisorCommandParser:

  def parse(line: String): AdvisorCommand =
    val trimmed = line.trim
    if trimmed.isEmpty then AdvisorCommand.Unknown("", "empty input")
    else
      val parts = trimmed.split("\\s+").toVector
      val cmd = parts.head.toLowerCase
      cmd match
        case "new"                                       => AdvisorCommand.NewHand
        case "hero" | "h" if parts.length >= 2           => parseHeroSub(parts.tail)
        case "villain" | "v" if parts.length >= 2        => parseVillainSub(parts.tail)
        case "board" | "b" if parts.length >= 2          => parseBoardSub(parts.tail)
        case "?" | "advise" | "advice"                   => AdvisorCommand.Advise
        case "review"                                    => AdvisorCommand.Review
        case "session" | "stats"                         => AdvisorCommand.SessionStats
        case "undo"                                      => AdvisorCommand.Undo
        case "help"                                      => AdvisorCommand.Help
        case "quit" | "exit" | "q"                       => AdvisorCommand.Quit
        case _ =>
          // Try interpreting the whole line as a 4-char hole card token
          if trimmed.length == 4 then
            tryParseHoleCards(trimmed) match
              case Some(hc) => AdvisorCommand.HeroCards(hc)
              case None     => AdvisorCommand.Unknown(trimmed, "unrecognized command")
          else AdvisorCommand.Unknown(trimmed, "unrecognized command")

  // ---- Hero sub-commands ----

  private def parseHeroSub(tokens: Vector[String]): AdvisorCommand =
    val first = tokens.head.toLowerCase
    first match
      case "raise" | "bet" => parseRaiseAmount(tokens.tail, isHero = true)
      case "call"          => AdvisorCommand.HeroAction(PokerAction.Call)
      case "fold"          => AdvisorCommand.HeroAction(PokerAction.Fold)
      case "check"         => AdvisorCommand.HeroAction(PokerAction.Check)
      case _ =>
        // Try as hole cards: "h AcKh" or "h Ac Kh"
        if tokens.length == 1 && tokens.head.length == 4 then
          tryParseHoleCards(tokens.head) match
            case Some(hc) => AdvisorCommand.HeroCards(hc)
            case None     => AdvisorCommand.Unknown(s"h ${tokens.mkString(" ")}", s"invalid hole cards: ${tokens.head}")
        else if tokens.length == 2 && tokens(0).length == 2 && tokens(1).length == 2 then
          tryParseHoleCards(tokens(0) + tokens(1)) match
            case Some(hc) => AdvisorCommand.HeroCards(hc)
            case None     => AdvisorCommand.Unknown(s"h ${tokens.mkString(" ")}", s"invalid hole cards: ${tokens.mkString(" ")}")
        else
          AdvisorCommand.Unknown(s"h ${tokens.mkString(" ")}", "expected: raise|bet|call|fold|check or hole cards")

  // ---- Villain sub-commands ----

  private def parseVillainSub(tokens: Vector[String]): AdvisorCommand =
    val first = tokens.head.toLowerCase
    first match
      case "raise" | "bet" => parseRaiseAmount(tokens.tail, isHero = false)
      case "call"          => AdvisorCommand.VillainAction(PokerAction.Call)
      case "fold"          => AdvisorCommand.VillainAction(PokerAction.Fold)
      case "check"         => AdvisorCommand.VillainAction(PokerAction.Check)
      case _ =>
        AdvisorCommand.Unknown(s"v ${tokens.mkString(" ")}", "expected: raise|bet|call|fold|check")

  // ---- Board sub-command ----

  private def parseBoardSub(tokens: Vector[String]): AdvisorCommand =
    // Support: "board Ts9h8d" (one token, split into 2-char chunks)
    //      or: "board Ts 9h 8d" (separate tokens)
    val cardTokens =
      if tokens.length == 1 && tokens.head.length > 2 && tokens.head.length % 2 == 0 then
        tokens.head.grouped(2).toVector
      else
        tokens

    val parsed = cardTokens.map(Card.parse)
    if parsed.exists(_.isEmpty) then
      val bad = cardTokens.zip(parsed).collect { case (t, None) => t }
      AdvisorCommand.Unknown(s"board ${tokens.mkString(" ")}", s"invalid card(s): ${bad.mkString(", ")}")
    else
      AdvisorCommand.DealBoard(parsed.flatten)

  // ---- Helpers ----

  private def parseRaiseAmount(tokens: Vector[String], isHero: Boolean): AdvisorCommand =
    if tokens.isEmpty then
      val who = if isHero then "h" else "v"
      AdvisorCommand.Unknown(s"$who raise", "raise requires an amount (e.g., raise 6)")
    else
      tokens.head.toDoubleOption match
        case Some(amount) if amount > 0.0 =>
          val action = PokerAction.Raise(amount)
          if isHero then AdvisorCommand.HeroAction(action) else AdvisorCommand.VillainAction(action)
        case Some(_) =>
          AdvisorCommand.Unknown(s"raise ${tokens.head}", "raise amount must be positive")
        case None =>
          AdvisorCommand.Unknown(s"raise ${tokens.head}", s"invalid number: ${tokens.head}")

  private def tryParseHoleCards(token: String): Option[HoleCards] =
    try Some(CliHelpers.parseHoleCards(token))
    catch case _: Exception => None
