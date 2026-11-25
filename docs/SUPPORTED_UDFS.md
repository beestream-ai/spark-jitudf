# Supported UDFs (TC/cuDF backend)

This document defines which Spark Java UDFs the TC/cuDF backend compiles for the GPU, what is
preserved when it does, and what happens when it does not. It is derived from the eligibility
implementation (`spark-jitudf-plugin/src/main/java/ai/beestream/jitudf/`), the unit and
provider tests, the 153-case provider coverage corpus, and the GPU suites run on an RTX 4070 and an
A100. Nothing here is a claim about arbitrary Java.

Paths in the *Evidence* columns are relative to this repository. `T:` is a plugin unit test
(`spark-jitudf-plugin/src/test/java/ai/beestream/jitudf/`), `G:` a GPU suite in
`spark-jitudf-bench/src/main/java/ai/beestream/jitudf/bench/`, and `Cnn`/`Snn`/`Unn`/`Rnn`/`Mnn`/`Gnn`
are row ids of the coverage corpus in
[`CoverageCorpus.java`](../spark-jitudf-plugin/src/test/java/ai/beestream/jitudf/CoverageCorpus.java).

## How eligibility works

* **Per UDF, at planning time.** RAPIDS asks the provider about one `ScalaUDF` at a time, only when
  no hand-written `RapidsUDF` exists and `spark.rapids.sql.tornadovm.enabled=true`. The provider
  either returns a compiled `RapidsUDF` or declines with a recorded reason
  (`TornadoUdfProvider.decisions()`). No data is touched to decide.
* **Declining is safe.** A declined UDF keeps exactly the CPU path RAPIDS would have used: the RAPIDS
  CPU bridge (`GpuCpuBridge`, on by default), the row-based UDF operator if
  `spark.rapids.sql.rowBasedUDF.enabled=true`, or Spark's CPU projection.
* **Eligibility is not placement.** A compiled UDF appears as `gpu_<name>` only if RAPIDS places the
  surrounding projection on the GPU. A UDF whose receiver must not be re-serialized on the worker
  (see [Records and captured state](#records-and-captured-state)) is kept out of the CPU bridge; by
  default RAPIDS then runs that whole projection on the CPU, including compiled siblings in it. With
  `spark.rapids.sql.rowBasedUDF.enabled=true` the same projection stays on the GPU, with the
  declined UDF as `gpu_row_based_<name>` beside the compiled `gpu_<sibling>`. Results are correct in
  both configurations; only placement differs.
* **After planning, failures are failures.** If a compiled UDF fails on an executor (NVRTC, a
  missing native symbol, a device error), the task fails. There is no executor-side fallback.

## Status categories

| Status | Meaning |
|---|---|
| **Supported** | Compiled when the UDF stays inside the stated constraints; GPU results tested against CPU Spark. |
| **Conditional** | Compiled only under an additional proof or configuration; otherwise declined (safe CPU path). |
| **Limited** | A narrow, specific shape is compiled; everything else in the category is declined. |
| **Declined** | Never compiled; always the CPU path. |
| **Out of scope** | Not a JVM `ScalaUDF`; never offered to the provider. |

## Core numerical capabilities

| Capability | Status | Constraints and semantics | Evidence |
|---|---|---|---|
| Fixed-width input columns | Supported | Spark `IntegerType`, `LongType`, `FloatType`, `DoubleType`, `BooleanType`. Parameters boxed or primitive. Any other Spark type (string, decimal, date, timestamp, binary, struct, map) is declined. | C01–C03; G: `CudfTransformIntegrationTest` (`BothBooleanUdf`) |
| Multiple inputs | Supported | Java `UDF1`…`UDF5` and above; arity 1–5 tested on the GPU, mixed widths. The binding imposes no arity limit. | C02, C03; G: `CudfTransformIntegrationTest` (`TwoIntUdf`, `ThreeInputUdf`, `FiveInputUdf`) |
| Typed outputs | Supported | INT32, INT64, FLOAT32, FLOAT64, BOOL8, matching the registered Spark return type. The Java return type must agree with it. | C04–C07 |
| Integer arithmetic | Supported | Java two's-complement wraparound on `int`/`long`, shifts, bitwise ops, comparisons. | C01–C04, C40; G: `CodexDifferentialTest` |
| Branches | Supported | `if`/ternary/`switch` over primitives. Reference comparison (`==` on boxes) is declined. | C08 |
| Loops | Supported | Counted and data-dependent loops (`for`/`while`), e.g. Collatz. Recursion and call cycles are declined. | C09, C10; G: `CudfTransformIntegrationTest` (`CollatzUdf`) |
| Static helper methods | Supported | `static` methods of the UDF's own class, primitive signatures, up to 32 reachable methods. Instance helper calls are declined. | C11, C48 |
| Numerical library calls | Conditional | `java.lang.Math`/`StrictMath`: `sqrt exp log log10 sin cos tan tanh sinh cosh asin acos atan atan2 cbrt abs floor ceil rint pow min max fma signum hypot toRadians toDegrees floorMod floorDiv`. `Integer`/`Long`: `bitCount rotateLeft rotateRight reverse reverseBytes signum numberOfLeadingZeros numberOfTrailingZeros highestOneBit lowestOneBit`. Anything else outside the UDF's class is declined. Floating-point functions need the FP opt-in below. | C14, C15, C17, C33; T: `ArithmeticEffectsTest` |
| Integer division and remainder | Supported | Java semantics: `Integer.MIN_VALUE / -1` wraps, `MIN % -1 == 0`. A divisor that cannot be proven non-zero on every control-flow path is compiled with a zero check; a zero divisor fails the query with a cuDF `DIVISION_BY_ZERO` evaluation error (Java: `ArithmeticException`). A provably non-zero divisor gets no check. | C12, C13, S18–S29; T: `ConditionalDivisorTest`; G: `CodexDifferentialTest` |
| Floating point | Conditional | Any reachable FP operation requires `spark.rapids.sql.incompatibleOps.enabled=true`, set explicitly (the provider treats an unset key as `false`, although RAPIDS' own default is `true`); otherwise declined. Results are not guaranteed bit-identical to the JVM: CUDA math functions and FMA contraction can differ by a few ULP. FP LIST sums are also declined if `spark.rapids.sql.variableFloatAgg.enabled` is `false` (its default is `true`). | C05, C06, C14, C35, U07, U23–U25 |
| Narrowing and widening conversions | Supported | Primitive conversions (`i2l`, `d2i`, …) with Java semantics; FP conversions need the FP opt-in. | C31, C49, U24 |
| Casts | Conditional | A `CHECKCAST`/`instanceof` is compiled only when the analysis proves it cannot fail; casts that could fail (e.g. `Integer`→`Long`, row-dependent types) are declined. | C30, C45, C46, U04, U20, U21; T: `CastAndStateSemanticsTest` |
| Determinism | Required | Non-deterministic UDFs (`asNondeterministic`) are declined. | — |

## Nulls

| Capability | Status | Constraints and semantics | Evidence |
|---|---|---|---|
| Nullable input columns | Supported | The UDF is compiled in cuDF's null-aware mode: each argument arrives as a `cuda::std::optional` and the compiled body sees `null` exactly where the Java body would. | C20–C22; G: `CudfTransformIntegrationTest` |
| UDF tests for null / returns null | Supported | `x == null` checks and `return null` compile; the output column carries nulls. | C21, C22 |
| Unboxing a null | Supported, diagnostic differs | Where Java would throw `NullPointerException` (unboxing a null `Integer`), the GPU fails the query with a cuDF evaluation error, reported as `UNKNOWN_ERROR` (code 3). **Success versus failure is preserved; the Java exception type is not.** | C20; G: `CodexDifferentialTest` (null-unboxing checks) |
| Spark primitive-parameter Scala UDFs | Supported | Spark itself skips such a UDF for null input (`KnownNotNull`), and the compiled code then sees no null. | M13 |

## Validated Java-language extensions

| Capability | Status | Constraints and semantics | Evidence |
|---|---|---|---|
| Java UDF classes (`UDF1`…), lambdas, Spark's `ToScalaUDF` adapter | Supported | The method Spark actually calls is resolved by JVM virtual-dispatch rules from the runtime class (including compiler bridges), not by name. | C27–C29, S01–S06; T: `DispatchResolutionTest`, `SparkRegistrationTest` |
| Inherited implementations, overloads, generic bridges | Supported | The dispatched method is compiled even when a subclass adds overloads with other signatures. Interface default methods are declined (gap G01). | S01–S06, G01 |
| Forwarding adapters | Conditional | A lambda that only forwards its arguments unchanged to one captured object (`x -> inner.call(x)`, typed or erased) compiles the target. A wrapper that computes anything itself, reorders arguments, casts unsafely, or catches and changes exceptions is declined. | S07–S11, S34, S36, U01, U26–U29, U36; G02 |
| Static method references | Conditional | `(UDF1<Integer,Integer>) Type::method` with a primitive or boxed target. Java's argument adaptation is preserved: a primitive parameter is unboxed **before** the target runs, so a null fails the query (`UNKNOWN_ERROR`) even if the argument is unused; a boxed parameter receives the null. References that need unboxing plus widening (`Integer` → `long` parameter) are declined (gap G05). | M01–M12; G: `CodexDifferentialTest` (primitive/mixed/unused method references) |
| Bound method references | Conditional | `inner::call` on a captured receiver is resolved by dispatch; null-sensitive over nullable input. | S34 |

## Records and captured state

The compiled code has no access to the UDF object. Configuration is supported by **folding** a
field's value into the generated code, which is correct only if every executor sees the same value
after deserialization.

| Capability | Status | Constraints and semantics | Evidence |
|---|---|---|---|
| Constructor-configured `final` primitive fields | Conditional | `final int/long/boolean/float/double` instance fields are folded when Java serialization fixes their executor value on the driver: default serialization, or `writeObject`/`readObject`/`readResolve` whose bytecode is a pure function of the object's primitive state. `transient` fields fold to their default. Different instances get different compiled code and cache keys. | C18, C19, C34, C43, C44, S12–S17, S32, S33, S35; T: `ExecutorInvariantFieldsTest` (two-JVM) |
| Environment-dependent or non-deterministic state | Declined | A hook or constructor that reads a system property, the environment, a static, or performs any call outside the stream field API; `writeReplace`; `readObjectNoData`; `Externalizable`; a non-serializable superclass with a non-trivial constructor; a mutable referenced object; a field of a reachable object whose hook may fail on the worker. | U22, U30, U31, U33–U35, U37–U39 |
| Records | Conditional | Java deserializes a record by calling its canonical constructor on the executor. Component-copy constructors are folded directly. A computed constructor is accepted only if it is pure numerical bytecode and the record's components are a fixed point of a serialization round trip. Reachable records are checked even when not read. | R01–R03, R10, R14; declined R04–R09, R13 |
| Static fields | Declined | Any `getstatic` at execution time is declined, including environment-initialised `static final` fields; `javac`-inlined compile-time constants are unaffected. | T: `RelatedRiskTest` |
| Receiver re-serialization in fallback | Handled | If a UDF's receiver must not be serialized a second time on the worker (non-idempotent record constructors or hooks), the provider tells RAPIDS, which then keeps it out of the CPU bridge. See [How eligibility works](#how-eligibility-works) for placement. | G: `SeparateExecutorTest` |

## LIST (array) columns

| Capability | Status | Constraints and semantics | Evidence |
|---|---|---|---|
| Full-row numerical folds over `ArrayType` | Limited | One `ArrayType` argument of a fixed-width element type with `containsNull=false` (element nulls declined), read as a Java primitive array or a Scala `Seq`, in a loop that the splitter proves is a sum of a per-element expression over every element from index 0, with a zero initial accumulator. Lowered to a per-element cuDF transform plus a segmented sum. Empty rows give 0; integer sums wrap; a null list row fails the query (Java would dereference it). | C23–C26, C32, C36–C42, S30, S31; T: `ListReductionTest`, `ListFoldSemanticsTest` |
| Other list loops | Declined | Stride ≠ 1, `break`/`continue`/early return, conditional accumulator updates, reversed traversal, non-zero initial value, work after the loop, non-additive accumulation. | U02, U10–U19 |
| General per-row LIST transforms, array outputs | Declined | Not implemented. | — |

## Declined patterns (always the CPU path)

| Pattern | Notes |
|---|---|
| String, decimal, date/timestamp, binary, struct and map inputs or outputs | Not fixed-width numerical types. |
| Object allocation (`new`), arrays other than LIST inputs | — |
| Exceptions: `throw`, `try`/`catch` in the UDF body | Forwarding adapters that rethrow runtime exceptions unchanged are the one accepted shape (S36). |
| Instance method calls, interface calls, calls into classes other than the whitelisted JDK methods | Includes most third-party libraries. |
| Mutable instance or static state, `synchronized`, `invokedynamic` (string concatenation, nested lambdas) | — |
| Non-deterministic UDFs | — |
| Any bytecode outside the analysis' positive list | The list is in `UdfEligibilityAnalyzer.ALLOWED_OPCODES`. |

## Out of scope

Python UDFs, pandas UDFs, UDAFs and table functions are not `ScalaUDF` expressions and are never
offered to the provider. Hive UDFs (`HiveSimpleUDF`) are not handled.

## Known safe coverage gaps

These nine corpus rows are semantically safe UDFs that are declined today. They are documented
limitations, not correctness problems: each runs correctly on the CPU path.

| Row | Pattern | Reason it is declined |
|---|---|---|
| G01 | Implementation is an interface default method | Dispatch to default methods is not modelled |
| G02 | Wrapper computing on the forwarded result (`x -> inner(x) * 2`) | Forwarding is accepted only when the wrapper computes nothing itself |
| G03 | Java lambda capturing an effectively-final `int` | Captured scalars in lambdas are not folded |
| G04 | LIST fold holding the element in a temporary read inside a conditional | Recognizer limitation of the LIST splitter |
| G05 | `Integer` method reference to a method taking `long` | Unboxing plus widening adaptation is not modelled |
| R11 | Record with an unread `enum` component | Enum deserialization is not on the trusted-leaf list |
| R12 | Record with an unread `ArrayList` component | JDK collection serialization hooks are not proven |
| R15 | Computed record constructor that tests an array-valued component | The fixed-point proof certifies only primitive and immutable scalar components |
| M11 | Wide primitive result plus a reference comparison in the target | TC declines reference comparison in null-aware mode (`TC-E1002`) |

## Error semantics, summarized

| Java behavior | GPU behavior | Preserved |
|---|---|---|
| Normal result | Same value (integer: bit-exact; floating point: see above) | yes |
| `ArithmeticException` from integer `/`, `%`, `floorDiv`, `floorMod` by zero | Query fails with cuDF `DIVISION_BY_ZERO` | failure yes, exception type no |
| `NullPointerException` from unboxing | Query fails with cuDF `UNKNOWN_ERROR` (code 3) | failure yes, exception type no |
| `NullPointerException` from a null LIST row | Query fails (`NullListException`) | failure yes, exception type no |
| Any other exception | The UDF is declined at planning (no `throw` paths are compiled) | n/a |

## The coverage corpus

[`CoverageCorpus.java`](../spark-jitudf-plugin/src/test/java/ai/beestream/jitudf/CoverageCorpus.java)
holds 153 curated cases, each labeled SAFE (should compile), UNSAFE (must stay on the CPU) or GAP
(safe but declined). [`CoverageMatrixTest`](../spark-jitudf-plugin/src/test/java/ai/beestream/jitudf/CoverageMatrixTest.java)
runs the real provider on each and writes `spark-jitudf-plugin/target/coverage-matrix.tsv`:

| | Cases |
|---|---:|
| Correctly GPU-eligible | 102 |
| Correctly kept on the CPU (required fallback) | 42 |
| Safe but declined (the gaps above) | 9 |
| **Total** | **153** |

The corpus is a set of compiler test cases chosen to probe semantic edges. **It is not a sample of
real-world UDFs, and 102/153 is not an acceptance rate for customer workloads.**
