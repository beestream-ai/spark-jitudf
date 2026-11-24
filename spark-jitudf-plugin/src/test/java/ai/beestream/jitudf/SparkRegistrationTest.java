/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitudf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.catalyst.expressions.Expression;
import org.apache.spark.sql.catalyst.expressions.ScalaUDF;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * The three counterexamples of Codex's second review through ordinary {@code spark.udf().register}:
 * the function object the provider is given is taken from Spark's own analysed plan, exactly what
 * the RAPIDS bridge hands it, and Spark's CPU execution is the reference.
 *
 * <p>
 * This runs on the CPU only. The GPU side of the same registrations -- the physical plan and the
 * values computed by cuDF -- is {@code CodexDifferentialTest}'s second-review section in the bench
 * module, which needs a GPU.
 * </p>
 */
class SparkRegistrationTest {

    static final DataType I = DataTypes.IntegerType;
    static final DataType L = DataTypes.LongType;

    private static SparkSession spark;

    @BeforeAll
    static void start() {
        spark = SparkSession.builder().master("local[2]").appName("registration").config("spark.ui.enabled", "false")
                .config("spark.sql.shuffle.partitions", "2").getOrCreate();
    }

    @AfterAll
    static void stop() {
        spark.stop();
    }

    @BeforeEach
    void reset() {
        TornadoUdfProvider.resetDecisions();
    }

    /** The ScalaUDF Spark planned for {@code df}, and its CPU result rows. */
    private static ScalaUDF scalaUdfOf(Dataset<Row> df) {
        List<Expression> found = new ArrayList<>();
        collectPlan(df.queryExecution().analyzed(), found);
        assertEquals(1, found.size(), "one ScalaUDF in the plan");
        return (ScalaUDF) found.get(0);
    }

    private static void collectPlan(org.apache.spark.sql.catalyst.plans.logical.LogicalPlan plan, List<Expression> out) {
        for (Expression e : scala.jdk.javaapi.CollectionConverters.asJava(plan.expressions())) {
            collect(e, out);
        }
        for (org.apache.spark.sql.catalyst.plans.logical.LogicalPlan child : scala.jdk.javaapi.CollectionConverters.asJava(plan.children())) {
            collectPlan(child, out);
        }
    }

    private static void collect(Expression e, List<Expression> out) {
        if (e instanceof ScalaUDF) {
            out.add(e);
        }
        for (Expression child : scala.jdk.javaapi.CollectionConverters.asJava(e.children())) {
            collect(child, out);
        }
    }

    /** What the provider decides for the registered UDF, given exactly what Spark planned. */
    private static ProviderHarness.Outcome provider(ScalaUDF udf, String name) {
        List<DataType> inputs = new ArrayList<>();
        boolean nullable = false;
        for (Expression child : scala.jdk.javaapi.CollectionConverters.asJava(udf.children())) {
            inputs.add(child.dataType());
            nullable |= child.nullable();
        }
        return ProviderHarness.cudf(udf.function(), inputs.toArray(new DataType[0]), udf.dataType(), nullable, name);
    }

    @Test
    void inheritedOverloadThroughRegistration() throws Exception {
        spark.udf().register("inherited_overload", new DispatchResolutionTest.PrimitiveOverloadChild(), DataTypes.IntegerType);
        Dataset<Row> df = spark.range(-5, 20).selectExpr("id", "inherited_overload(CAST(id AS INT), 0) AS r").orderBy("id");
        List<Row> rows = df.collectAsList();
        for (Row row : rows) {
            assertEquals((int) row.getLong(0) + 1, row.getInt(1), "Spark runs the inherited call(Integer,Integer)");
        }
        ScalaUDF udf = scalaUdfOf(df);
        ProviderHarness.Outcome o = provider(udf, "inherited_overload");
        assertTrue(o.accepted(), o::decision);
        Method body = LoweredOracle.scalarBody(udf.function(), List.of(I, I), I);
        for (Row row : rows) {
            assertEquals(row.getInt(1), LoweredOracle.run(body, (int) row.getLong(0), 0));
        }
    }

    @Test
    void typedForwarderThroughRegistration() throws Exception {
        DispatchResolutionTest.Popcount inner = new DispatchResolutionTest.Popcount();
        UDF1<Long, Integer> forwarder = x -> inner.call(x);
        spark.udf().register("typed_forwarder", forwarder, DataTypes.IntegerType);
        Dataset<Row> df = spark.range(0, 50).selectExpr("id", "typed_forwarder(id * 2654435761) AS r").orderBy("id");
        List<Row> rows = df.collectAsList();
        ProviderHarness.Outcome o = provider(scalaUdfOf(df), "typed_forwarder");
        assertTrue(o.accepted(), o::decision);
        Method body = LoweredOracle.scalarBody(scalaUdfOf(df).function(), List.of(L), I);
        for (Row row : rows) {
            assertEquals(row.getInt(1), LoweredOracle.run(body, row.getLong(0) * 2654435761L));
        }
    }

    @Test
    void environmentDependentReadResolveThroughRegistration() {
        spark.udf().register("resolved_factor", new ExecutorInvariantFieldsTest.ResolvedFactor(9), DataTypes.IntegerType);
        Dataset<Row> df = spark.range(0, 10).selectExpr("id", "resolved_factor(CAST(id AS INT)) AS r").orderBy("id");
        for (Row row : df.collectAsList()) {
            assertEquals((int) row.getLong(0) + 9, row.getInt(1), "this JVM has no review.factor, so its executors compute x + 9");
        }
        ProviderHarness.Outcome o = provider(scalaUdfOf(df), "resolved_factor");
        assertFalse(o.accepted(), () -> "must not be compiled with a driver constant: " + o.cuda());

        spark.udf().register("offset", new ExecutorInvariantFieldsTest.Offset(41), DataTypes.IntegerType);
        Dataset<Row> ok = spark.range(0, 10).selectExpr("id", "offset(CAST(id AS INT)) AS r");
        ProviderHarness.Outcome folded = provider(scalaUdfOf(ok), "offset");
        assertTrue(folded.accepted(), folded::decision);
        assertTrue(folded.cuda().contains("41"), folded::cuda);
    }

    @Test
    void conditionalDivisorThroughRegistration() throws Exception {
        spark.udf().register("cond_floor_mod", new ConditionalDivisorTest.CondFloorMod(), DataTypes.IntegerType);
        Dataset<Row> safe = spark.range(-20, 20).selectExpr("id", "cond_floor_mod(CAST(id AS INT), CAST(id % 5 AS INT) + 7) AS r").orderBy("id");
        List<Row> rows = safe.collectAsList();
        ScalaUDF udf = scalaUdfOf(safe);
        ProviderHarness.Outcome o = provider(udf, "cond_floor_mod");
        assertTrue(o.accepted(), o::decision);
        assertTrue(CoverageCorpus.zeroTest(o.cuda()), o::cuda);
        Method body = LoweredOracle.scalarBody(udf.function(), List.of(I, I), I);
        for (Row row : rows) {
            int x = (int) row.getLong(0);
            assertEquals(row.getInt(1), LoweredOracle.run(body, x, (int) (row.getLong(0) % 5) + 7));
        }
        // (7, 0): Spark's CPU plan raises ArithmeticException, and so does the compiled body.
        Dataset<Row> zero = spark.range(7, 8).selectExpr("cond_floor_mod(CAST(id AS INT), 0) AS r");
        Throwable failure = null;
        try {
            zero.collectAsList();
        } catch (Exception e) {
            failure = e;
        }
        assertNotNull(failure, "Spark must fail on a zero divisor");
        boolean arithmetic = false;
        for (Throwable t = failure; t != null; t = t.getCause()) {
            arithmetic |= t instanceof ArithmeticException;
        }
        assertTrue(arithmetic, String.valueOf(failure));
        assertEquals("threw:ArithmeticException", LoweredOracle.run(body, 7, 0));
    }
}
