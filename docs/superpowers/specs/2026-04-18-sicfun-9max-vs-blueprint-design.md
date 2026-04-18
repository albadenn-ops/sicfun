# sicfun 9-max vs. Blueprint — Spec A (Dealer + Agents + Runner)

**Fecha:** 2026-04-18
**Autor:** Claude (sicfun sidecar planner)
**Estado:** Draft — pendiente de revisión del usuario antes de pasar a planning

## 1. Objetivo

Permitir que sicfun juegue 9-max NLHE contra oponentes blueprint-only que actúan como proxy de un solver estilo Pluribus, y medir cuantitativamente el valor añadido por la capa estratégica (`strategic/*`) sobre una baseline blueprint pura. Este spec cubre **solo** la infraestructura de juego: dealer N-seat, API de agentes, runner de mesa y gates de nativo estricto. El entrenamiento del blueprint 9-max vive en un spec separado (Spec B, posterior).

### 1.1 No objetivos

- No entrena un blueprint. Usa blueprints cargados desde archivo (formato definido aquí, contenido generado por Spec B).
- No reclama ser equivalente a Pluribus. Pluribus es código cerrado de Facebook/CMU; un blueprint-only agent es un proxy metodológico honesto, no un sustituto.
- No modifica `AcpcHeadsUpDealer`, `SlumbotMatchRunner`, ni ningún componente de heads-up en producción.
- No introduce fallback silencioso a Scala cuando las DLLs nativas no carguen.

### 1.2 Criterios de éxito

- Correr 100.000 manos 9-max con 1 `StrategicAgent` + 8 `BlueprintOnlyAgent` sin excepciones no capturadas.
- Reportar mbb/100 del `StrategicAgent` con IC95 bootstrap.
- Desglose por los cuatro mundos de `FourWorldDecomposition` disponible.
- Todos los tests munit verdes. Propiedad de conservación de side pots verificada en `AcpcTableDealerTest`.
- Ninguna ejecución del runner arranca si alguna DLL requerida no carga en modo strict.

## 2. Contexto

El repositorio ya contiene:

- **CFR nativo (CUDA + CPU):** [src/main/native/build/sicfun_cfr_cuda.dll](src/main/native/build/sicfun_cfr_cuda.dll), [src/main/native/build/sicfun_cfr_native.dll](src/main/native/build/sicfun_cfr_native.dll). Fuente: [src/main/native/jni/HoldemCfrNativeCpuBindings.cpp](src/main/native/jni/HoldemCfrNativeCpuBindings.cpp), [src/main/native/jni/HoldemCfrNativeGpuBindings.cu](src/main/native/jni/HoldemCfrNativeGpuBindings.cu), [src/main/native/jni/CfrNativeSolverCore.hpp](src/main/native/jni/CfrNativeSolverCore.hpp).
- **Solvers online:** POMCP ([src/main/native/jni/WPomcpSolver.hpp](src/main/native/jni/WPomcpSolver.hpp) + `HoldemPomcpNativeBindings.cpp` + `sicfun_pomcp_native.dll`), PFT-DPW ([src/main/native/jni/PftDpwSolver.hpp](src/main/native/jni/PftDpwSolver.hpp)).
- **Inferencia bayesiana / DDRE:** `sicfun_bayes_{cuda,native}.dll`, `sicfun_ddre_{cuda,native}.dll`.
- **Evaluador + equity:** `sicfun_gpu_kernel.dll`, `sicfun_opencl_kernel.dll`, `sicfun_native_cpu.dll`.
- **Capa estratégica completa** en [src/main/scala/sicfun/holdem/strategic/](src/main/scala/sicfun/holdem/strategic/): `AugmentedState`, `RivalKernel`, `SafetyBellman`, `PosteriorAttributedBaseline`, `FourWorldDecomposition`, **6 bridges**: `PublicStateBridge`, `BaselineBridge`, `OpponentModelBridge`, `ClassificationBridge`, `SignalBridge`, `ValueBridge`. (`BridgeManifest` y `StrategicSnapshot` conviven en el paquete `bridge/` pero no son bridges — son manifiesto y snapshot agregado respectivamente.)
- **Dealer heads-up:** [src/main/scala/sicfun/holdem/runtime/protocol/AcpcHeadsUpDealer.scala](src/main/scala/sicfun/holdem/runtime/protocol/AcpcHeadsUpDealer.scala) — 886 líneas con N=2 cableado (arrays de tamaño 2, lógica BB/SB hardcodeada). No refactorizable barato.

Limitaciones conocidas:

- **Pluribus no es accesible:** código cerrado, sin API, sin binarios, sin reimplementación pública entrenada a nivel competitivo. Este spec no intenta reproducir eso; usa blueprint-only agents como proxy metodológicamente equivalente.
- **Entrenar un blueprint 9-max a escala Pluribus es infactible** con el hardware local (GTX 960M). Spec B entrenará un blueprint **abstraído a menor escala**; este spec es agnóstico a esa decisión — solo define el formato de carga.
- **Fallback nativo→Scala actualmente silencioso en varios runtimes:** `HeadsUpGpuRuntime`, `PftDpwRuntime`, `WPomcpRuntime`, `WassersteinDroRuntime`, `HoldemPostflopNativeRuntime` cargan DLLs de forma independiente; si `System.load`/`System.loadLibrary` falla, algunos paths caen a Scala sin error. El runner 9-max exige modo estricto.

## 3. Arquitectura

### 3.1 Archivos nuevos

```
src/main/scala/sicfun/holdem/runtime/protocol/
  AcpcTableDealer.scala              # N-seat dealer, 2..9, side pots, rotación
  TableDealerTypes.scala             # SeatId, SidePot, BettingRoundEvent, HandOutcome

src/main/scala/sicfun/holdem/runtime/agent/      (paquete nuevo)
  SeatAgent.scala                    # trait base
  BlueprintOnlyAgent.scala           # política blueprint, sin adaptación
  StrategicAgent.scala               # blueprint + overlay estratégico
  BlueprintStore.scala               # carga y valida blueprint desde disco
  BlueprintFormat.scala              # schema del archivo .blueprint (binario)

src/main/scala/sicfun/holdem/runtime/
  NineMaxMatchRunner.scala           # orquesta mesa de 9, métricas, logging
  NativeStrictMode.scala             # gate de strict-native, probe eager de DLLs

src/main/scala/sicfun/holdem/runtime/metrics/    (paquete nuevo)
  MbbMetrics.scala                   # mbb/100 + bootstrap IC95
  FourWorldMetrics.scala             # desglose por decomposition

src/main/resources/blueprints/       (gitignored, contenido de Spec B)
  .gitkeep

data/matches/                        (gitignored)
  .gitkeep
```

### 3.2 Tests nuevos (munit)

```
src/test/scala/sicfun/holdem/runtime/protocol/
  AcpcTableDealerTest.scala          # side pots, cierre de ronda, rotación, property-based
  AcpcTableDealerSidePotPropertyTest.scala

src/test/scala/sicfun/holdem/runtime/agent/
  BlueprintOnlyAgentTest.scala
  StrategicAgentTest.scala           # mocks de bridges
  BlueprintStoreTest.scala           # checksum, versión, round-trip

src/test/scala/sicfun/holdem/runtime/
  NineMaxMatchRunnerSmokeTest.scala  # 1000 manos, dummy blueprint uniforme
  NativeStrictModeTest.scala         # probes sintéticos, stale detection

src/test/scala/sicfun/holdem/runtime/metrics/
  MbbMetricsTest.scala               # bootstrap determinista con seed
  FourWorldMetricsTest.scala
```

### 3.3 No se toca

`AcpcHeadsUpDealer.scala`, `SlumbotMatchRunner.scala`, `AcpcMatchRunner.scala` permanecen sin cambios. El nuevo dealer es independiente.

## 4. Componentes

### 4.1 `AcpcTableDealer`

Dealer genérico 2..9 jugadores. Responsabilidades:

- Reparto de cartas (2 hole cards por seat activo, 5 community cards).
- Rotación del button tras cada mano (BTN → SB → BB → UTG → UTG+1 → MP → MP+1 → HJ → CO al llenar 9 seats; secuencia colapsa correctamente para N<9).
- Postear blinds (SB/BB con chips configurables; ante opcional).
- Gestión de rondas (preflop/flop/turn/river) con cierre por "acción vuelve al último agresor" (no por "todos igualaron").
- Side pots multi-nivel. Cada all-in crea una capa; invariante `sum(pots) == sum(contributions)` verificado en cada evento.
- Showdown parcial (solo seats no-folded revelan al río).
- Emisión de `BettingRoundEvent` por cada acción, `HandOutcome` al final.
- No asume bot específico — recibe `Seq[SeatAgent]` y les consulta según el turno.

Estado interno inmutable por mano (nueva instancia por `playHand`). Configuración por mesa (chips, blinds, ante) inmutable por match.

### 4.2 `SeatAgent` — API

```scala
trait SeatAgent:
  def seatId: SeatId
  def onMatchStart(tableConfig: TableConfig): Unit
  def onHandStart(snapshot: TableSnapshot): Unit
  def decide(snapshot: TableSnapshot, legalActions: Set[PokerAction]): PokerAction
  def onHandEnd(snapshot: TableSnapshot, outcome: HandOutcome): Unit
  def onMatchEnd(summary: MatchSummary): Unit
```

- `TableSnapshot`: estado público visible para este seat (cartas propias, board, stacks, historial de acciones, posiciones).
- `decide` es síncrono; el runner aplica timeout (default 5s, configurable).
- Agent puede mantener estado entre manos (necesario para `StrategicAgent`).

### 4.3 `BlueprintOnlyAgent`

Proxy de un solver estilo Pluribus. Comportamiento:

1. En `decide`, extrae el **infostate abstraído** del snapshot según la misma abstracción que usó el entrenamiento del blueprint (abstracción de cartas por buckets + abstracción de acción fija).
2. Consulta `BlueprintStore` para obtener la distribución de acciones sobre acciones abstraídas.
3. Samplea acción abstraída, la traduce a acción concreta legal (con estrategia de "translation" estándar: snap al tamaño legal más cercano por log-ratio).
4. Sin estado entre manos. Sin adaptación al rival. Idéntico a cómo Pluribus ejecuta en live play.

**Determinismo opcional** vía seed; por defecto usa RNG aleatorio.

### 4.4 `StrategicAgent`

Integra el blueprint con la capa estratégica completa. En `decide`:

1. Extrae infostate abstraído → consulta blueprint → obtiene `baselinePolicy` (distribución sobre acciones).
2. Construye/actualiza `AugmentedState` a través de los **6 bridges** (verificados contra código 2026-04-18):
   - `PublicStateBridge` — estado público actual
   - `BaselineBridge` — política baseline del blueprint
   - `OpponentModelBridge` — `RivalKernel` por cada seat activo
   - `ClassificationBridge` — clase estratégica del seat actual (BTN/CO/etc.)
   - `SignalBridge` — señales observadas esta mano
   - `ValueBridge` — función de valor por mundo

3. Pipeline real de `SafetyBellman` (firmas verificadas contra [SafetyBellman.scala](src/main/scala/sicfun/holdem/strategic/safety/SafetyBellman.scala) al 2026-04-18; no existe método monolítico):

   La API es **a nivel MDP-indexado**, no opera directamente sobre `AugmentedState`. `StrategicAgent` debe construir una inmersión MDP (índices de estado/acción, matriz `robustLosses[s][a]`, `transitions(s,a,p)`, `numProfiles`, vector `qValues[a]`) a partir de los bridges antes de invocarla.

   1. `tSafe(currentBound, robustLosses, gamma, transitions, numProfiles, terminalStates)` — un paso del operador T_safe.
   2. `computeBStar(robustLosses, gamma, transitions, numProfiles, maxIterations=200, tolerance=1e-10, terminalStates)` — iteración de punto fijo hasta convergencia bajo `tolerance=1e-10`. Devuelve `Array[Double]` con B*(s) por estado.
   3. `safeActionSet(stateIndex, bound, robustLosses, gamma, transitions, numProfiles): IndexedSeq[Int]` — devuelve índices de acciones safe en `stateIndex`.
   4. `safeFeasibleAction(qValues, safeActions): Int` — selecciona la acción de mayor Q-value entre las safe; si el conjunto safe es vacío, devuelve la de mayor Q-value global (fallback algorítmico dentro de la API, no native→scala).
   5. El entero devuelto se mapea al `PokerAction` concreto vía la tabla de abstracción-acción usada al construir el MDP.

4. `DetectionPredicate.detectModeling(rivalId: PlayerId, history: Vector[PublicAction], publicState: PublicState): Boolean` evalúa si el rival está modelando activamente a SICFUN. Tres implementaciones en código:
   - `NeverDetect` — siempre `false`; útil como baseline y test stub.
   - `AlwaysDetect` — siempre `true`; fuerza retreat constante para tests.
   - `FrequencyAnomalyDetection(window, threshold, baselineFrequency)` — dispara cuando la fracción de acciones agresivas del rival en los últimos `window` excede `baselineFrequency + threshold`.

5. Lógica real de explotación (verificada contra [ExploitationInterpolation.scala](src/main/scala/sicfun/holdem/strategic/exploitation/ExploitationInterpolation.scala)):

   **La convención es la opuesta a lo que sugiere la intuición:** detección de modelado por el rival **reduce** la explotación, no la aumenta. `ExploitationInterpolation.retreat(state, config)` se invoca **cuando `detectModeling == true`** y baja `beta` hacia 0 (retorno al anchor baseline). Cuando `detectModeling == false`, `beta` se mantiene (sujeto a clamp de seguridad). El flujo completo está implementado en `updateExploitation(state, config, rivalId, history, publicState, detector, exploitabilityFn, epsilonNE)`: aplica retreat si detector fires, luego clampea `beta` vía `clampForSafety(beta, exploitabilityFn, epsilonNE, epsilonAdapt)` (o `clampForCertificate(beta, requiredBudget, availableBudget)` si se usa certificado Bellman-safe en lugar de oracle escalar).

   `interpolatePosterior(beta, refPosterior, attribPosterior)` opera sobre `DiscreteDistribution[StrategicClass]` — mezcla la creencia sobre la **clase estratégica** del rival, no directamente sobre acciones. La política del agente se deriva después de esa creencia vía el `baselinePolicy` ponderada por clase (el puente que conecta beta-interpolado con acción concreta vive en el bridge/value pipeline, no en este módulo).

Mantiene `RivalKernel × (N-1)` entre manos, actualizados en `onHandEnd` vía `HoldemBayesProvider` + `HoldemDdreProvider` (ambos usan DLLs nativas).

### 4.5 `BlueprintStore` + `BlueprintFormat`

Formato binario custom (`.blueprint`):

- **Header:** magic bytes `SICFBP01`, versión, fecha de entrenamiento, hash de la spec de abstracción (cartas + acción), seat-count del juego entrenado.
- **Tabla de abstracción:** mapping infostate-hash → offset en tabla de políticas.
- **Tabla de políticas:** por cada infostate, distribución de probabilidades sobre acciones abstraídas (float32 densamente packed).

`BlueprintStore.load(path)`:

- Valida magic, versión, hash de spec de abstracción contra la spec esperada por el código.
- Mmap del archivo para acceso O(1) en lookup.
- Fail fast si versión incompatible o hash no coincide.

`BlueprintStore.lookup(infostateHash): ActionDistribution` — consulta pura.

### 4.6 `NineMaxMatchRunner`

Orquesta match completo.

```scala
class NineMaxMatchRunner(
  tableConfig: TableConfig,
  agents: Seq[SeatAgent],        // size == tableConfig.numSeats
  numHands: Int,
  rngSeed: Long,
  matchId: String
):
  def run(): MatchResult
```

Flujo por mano:
1. `AcpcTableDealer.startHand(rotatedSeatOrder)`
2. Repartir cartas → `onHandStart` para cada agent
3. Loop de rondas: consultar `decide` del agent en turno → validar acción legal → aplicar al dealer → emitir evento
4. Showdown si corresponde → `HandOutcome`
5. `onHandEnd` para cada agent
6. Rotar button, actualizar stacks

Salida:
- `MatchResult` con mbb/100 por seat, IC95 bootstrap, desglose `FourWorldDecomposition` para el `StrategicAgent`, log completo en `data/matches/<matchId>.jsonl`.
- Parada anticipada opcional si IC95 no cruza 0 tras N mínimo (default 50.000 manos).

### 4.7 `NativeStrictMode`

Gate de arranque, invocado por `NineMaxMatchRunner` antes de la primera mano.

**DLLs requeridas para ejecutar el runner en modo strict, con justificación:**

Núcleo (siempre requerido):
- `sicfun_gpu_kernel.dll` + `sicfun_native_cpu.dll` — evaluador de manos y equity. Usadas por todos los bridges de valor.
- `sicfun_bayes_cuda.dll` + `sicfun_bayes_native.dll` — `HoldemBayesProvider` para updates de `RivalKernel` en `StrategicAgent.onHandEnd`.
- `sicfun_ddre_cuda.dll` + `sicfun_ddre_native.dll` — `HoldemDdreProvider` para inferencia DDRE en `OpponentModelBridge`.

Condicional según config del `StrategicAgent` (si la config lo habilita, **se exigen**; si no, se omiten del probe):
- `sicfun_pomcp_native.dll` — requerido si `StrategicAgent.solver == WPomcp`.
- `sicfun_postflop_cuda.dll` + `sicfun_postflop_native.dll` — requerido si `StrategicAgent.solver == PostflopSubgame` o si `ValueBridge` delega al solver postflop en river.
- `sicfun_cfr_cuda.dll` + `sicfun_cfr_native.dll` — **no requeridas en runtime**. Solo Spec B las usa para entrenamiento offline. Probe no las exige, pero sí verifica que existan en disco si se van a reentrenar.

`BlueprintOnlyAgent` **no carga ninguna DLL**: solo hashing + lookup mmap. Un match de 9 `BlueprintOnlyAgent`s sin `StrategicAgent` exige solo el núcleo (kernel + cpu) para el `AcpcTableDealer`.

**Flujo del probe eager:**
1. Para cada DLL requerida: `System.load(absolutePath)` desde `src/main/native/build/`. Recoger lista de fallos.
2. Si lista de fallos no vacía en modo strict → `throw NativeStrictModeViolation(missing: Seq[String], hint: "rebuild native with build-windows-cuda11.ps1")`.
3. Chequeo de frescura: para cada DLL cargada, obtener `mtime(dll)` y `mtime(cada fuente JNI asociada)`. Si alguna fuente > DLL: `throw NativeStaleBuildError(staleDll, staleSrc)`.
4. Mapping DLL → fuentes JNI está en una tabla estática en `NativeStrictMode` (mantenible a mano; hay 12 DLLs y ~17 fuentes).

**Flag:**
- `sicfun.native.strict` (system property) o `SICFUN_NATIVE_STRICT` (env var). Default `true` para `NineMaxMatchRunner`.
- Solo `false` explícito permite continuar con fallback; en ese caso loguea `WARN [BENCHMARK-CONTAMINATED] native <lib> unavailable, using scala fallback`.

**Política de errores mid-match:**
- `UnsatisfiedLinkError` y `NativeCallFailure` durante `decide`/`onHandEnd` **no se capturan** por el runner. Propagan → la mano aborta → el runner aborta con diagnóstico. No hay "swallow and continue".

## 5. Flujo de datos

```
[.blueprint file]
       |
       v
[BlueprintStore] --mmap--> [ActionDistribution lookup]
       |                            ^
       |                            |
       +---> [BlueprintOnlyAgent] --+
       |                            
       +---> [StrategicAgent] --+
                                |
                                v
                        [Bridges × 6] <--- [RivalKernel × 8]
                                |                    ^
                                v                    |
                        [AugmentedState]             |
                                |                    |
                                v                    |
                        [SafetyBellman] ---> [action]
                                                     |
[AcpcTableDealer] <-----decide()---------------------+
       |
       v
[BettingRoundEvent stream] ---> [NineMaxMatchRunner]
       |                              |
       v                              v
[HandOutcome]                  [MbbMetrics + FourWorldMetrics]
       |                              |
       v                              v
[onHandEnd → kernel updates]   [MatchResult + jsonl log]
```

## 6. Manejo de errores

| Caso | Comportamiento |
|------|----------------|
| DLL faltante en arranque (strict) | `NativeStrictModeViolation`, runner no arranca |
| DLL stale (fuente más nueva) | `NativeStaleBuildError`, runner no arranca |
| Blueprint file ausente | `BlueprintNotFoundError` en `NineMaxMatchRunner.run()` |
| Blueprint versión incompatible | `BlueprintVersionMismatch` con spec-hash esperado vs encontrado |
| Agent lanza excepción en `decide` | Match abortado, estado serializado a jsonl, excepción propagada |
| Timeout de `decide` (>5s) | `AgentDecisionTimeout(seatId, elapsed)`, match abortado |
| Acción devuelta no legal | `IllegalActionError(seatId, action, legalSet)`, match abortado |
| `UnsatisfiedLinkError` mid-call | No capturado, propaga, match abortado |
| Invariante de side pots violado | `SidePotInvariantViolation`, match abortado con dump del estado |
| Rival kernel update falla en `onHandEnd` | Propaga — no se silencia |

**No** se introducen fallbacks silenciosos, políticas random por defecto, ni "auto-fold en caso de error". Consistente con el constraint `no_fake_code`.

## 7. Testing

### 7.1 Unit tests

- `AcpcTableDealerTest`: cierre de ronda por último agresor (no por "todos igualaron"); rotación de button para N=2,3,6,9; posteo correcto de blinds; showdown parcial con foldeos mezclados.
- `BlueprintOnlyAgentTest`: lookup correcto para infostate conocido; translation de acción abstraída a legal; determinismo bajo seed fijo.
- `StrategicAgentTest`: los 6 bridges mockeados; con `NeverDetect` se verifica que la acción elegida es la devuelta por `safeFeasibleAction(qValues, safeActions)` sobre el pipeline completo `computeBStar → safeActionSet → safeFeasibleAction`, con `beta` retenido; con `AlwaysDetect` se verifica que `updateExploitation` invoca `retreat` cada paso y que `beta` decae hacia 0, colapsando la política hacia el baseline (no hacia explotación — la semántica es que la detección desactiva el exploit, no lo amplifica).
- `BlueprintStoreTest`: round-trip escritura/lectura; rechazo de magic incorrecto; rechazo de spec-hash incorrecto.
- `NativeStrictModeTest`: probe sintético con DLL inexistente → viola; con DLL "stale" simulada (timestamp manipulado) → viola; con flag `false` → warn y continúa.

### 7.2 Property-based (munit + scalacheck)

- `AcpcTableDealerSidePotPropertyTest`: para cualquier secuencia válida de acciones que incluya K all-ins con stacks diversos, `sum(pots) == sum(contributions)`, cada pot se distribuye solo entre seats elegibles, total distribuido == total apostado.
- Rotación de button: tras N manos con N seats, cada seat ha sido button exactamente una vez.

### 7.3 Smoke end-to-end

- `NineMaxMatchRunnerSmokeTest`: blueprint dummy uniforme (siempre distribuye prob igual sobre {fold, call, raise-pot}), 1000 manos, verificar que termina sin excepciones y que el log jsonl tiene 1000 entries `HandOutcome`.

### 7.4 Benchmark (no automatizado)

- 100.000 manos con blueprint de Spec B cuando esté disponible. Meta: producir `MatchResult` firmado con mbb/100 e IC95.

## 8. Dependencias con otros specs

- **Depende de Spec B** para contenido útil del blueprint. Spec A entrega infraestructura + agentes consumibles con blueprint dummy.
- **Depende de la capa estratégica existente** (`strategic/*`). Cualquier regresión ahí rompe `StrategicAgent`.
- **Depende de las DLLs nativas existentes.** Si la build del 2026-04-18 se invalida por cambios futuros, `NativeStrictMode` detendrá el runner hasta que se reconstruya.

## 9. Riesgos

| Riesgo | Mitigación |
|--------|------------|
| `AcpcTableDealer` introduce bugs sutiles en side pots | Property-based tests exhaustivos sobre invariante `sum(pots) == sum(contributions)` y elegibilidad por pot; casos canónicos (3-way all-in desigual, 4-way con dos laterales) como unit tests dedicados |
| `StrategicAgent` tiene mismatches de bridges cuando el infostate 9-max no coincide con lo que la capa estratégica espera (fue diseñada multiway pero testeada HU) | Sección de integration test dedicada; fail-fast si algún bridge devuelve `None` inesperado |
| Blueprint de Spec B llega y no calza con formato de este spec | Spec B debe citar `BlueprintFormat` de este spec como contrato congelado |
| Strict-native mode rompe dev loops locales | Flag `sicfun.native.strict=false` disponible con warning explícito |
| 100.000 manos no dan IC95 separado de 0 | Diseño acepta resultado neutro como outcome válido del experimento; no es un bug |

## 10. Resumen

Spec A entrega la infraestructura de juego 9-max: dealer genérico, API de agentes, runner con métricas, y gate de nativo estricto que fuerza uso de DLLs recién compiladas sin fallbacks silenciosos. No reclama equivalencia con Pluribus; construye el marco donde un blueprint (Spec B) jugado por 8 agentes enfrentado a 1 `StrategicAgent` produce la medición honesta del valor añadido por la capa estratégica.
