// Quick start for the TC/cuDF backend. Feed it to spark-shell on standard input, exactly as shown in
// README.md and docs/INSTALL.md: the RAPIDS distribution with the generalized JNI, the plugin and the
// bench jar on --jars, -Dtornado.spark.backend=cudf-transform on the driver, and
// spark.rapids.sql.tornadovm.enabled=true.
//
// It uses UDFs that already exist and are tested in this repository; nothing here is specific to
// this example.
import org.apache.spark.sql.types.DataTypes
import ai.beestream.jitudf.TornadoUdfProvider
import ai.beestream.jitudf.bench.CudfTransformIntegrationTest
import ai.beestream.jitudf.bench.udfs.StringUdf

// 1. An integer UDF with a data-dependent loop (steps of the Collatz sequence), registered the
//    ordinary way. RAPIDS' own udf-compiler declines it: an expression tree cannot hold a loop.
spark.udf.register("collatz", new CudfTransformIntegrationTest.CollatzUdf(), DataTypes.IntegerType)
val gpuQuery = spark.range(1, 1000001).selectExpr("sum(collatz(id)) AS total_steps")
println("=== GPU (TC/cuDF) result: total_steps = " + gpuQuery.collect()(0).getLong(0))
// The plan the query actually ran with. explain() before running shows only AQE's initial plan,
// which does not yet contain the GPU operators.
println("=== executed plan: expect gpu_collatz in the GPU operators, and no GpuCpuBridge ===")
println(gpuQuery.queryExecution.executedPlan)
println("provider decision for collatz: " + TornadoUdfProvider.decisions().get("collatz"))

// 2. The same query on the CPU, as the oracle.
spark.conf.set("spark.rapids.sql.enabled", "false")
val cpuQuery = spark.range(1, 1000001).selectExpr("sum(collatz(id)) AS total_steps")
println("=== CPU result (spark.rapids.sql.enabled=false): total_steps = " + cpuQuery.collect()(0).getLong(0))
spark.conf.set("spark.rapids.sql.enabled", "true")

// 3. A UDF outside the supported domain (it returns a String). The provider declines it while
//    planning, and the query keeps the CPU path RAPIDS would have used anyway.
spark.udf.register("sign_label", new StringUdf(), DataTypes.StringType)
val fallback = spark.range(-2, 3).selectExpr("id", "sign_label(CAST(id AS DOUBLE)) AS label")
println("=== declined UDF: " + fallback.collect().mkString(" "))
println(fallback.queryExecution.executedPlan)
println("provider decision for sign_label: " + TornadoUdfProvider.decisions().get("sign_label"))

System.exit(0)
