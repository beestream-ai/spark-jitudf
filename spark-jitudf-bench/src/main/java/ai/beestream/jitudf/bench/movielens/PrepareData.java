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
package ai.beestream.jitudf.bench.movielens;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

/**
 * Offline preparation for the MovieLens Tag Genome similarity application.
 *
 * <p>
 * This is deliberately not part of any timed measurement. It reads the raw MovieLens 25M CSVs once
 * and writes Parquet that every baseline then reads identically, so that the thing being compared
 * is the scoring query and not CSV parsing.
 * </p>
 *
 * <p>
 * What it produces:
 * </p>
 * <ul>
 * <li><b>movie_vectors</b> &mdash; one row per movie that has a Tag Genome entry, carrying a
 * fixed-order {@code array&lt;double&gt;} of tag relevance values. The order is by {@code tagId}
 * ascending, which is what makes the vectors comparable across rows: element <i>k</i> of every
 * vector is the same tag. Values are the dataset's own relevance scores; nothing is synthesised.</li>
 * <li><b>movie_popularity</b> &mdash; rating counts per movie, used only to choose which movies are
 * worth comparing. Popularity is a defensible, reproducible candidate policy: it needs no model and
 * no randomness, and it mirrors what a catalogue-facing "more like this" query would do.</li>
 * </ul>
 */
public final class PrepareData {

    /** The Tag Genome has exactly this many tags; a vector of any other length is a data error. */
    static final int GENOME_TAGS = 1128;

    public static void main(String[] args) {
        String input = args.length > 0 ? args[0] : "third_party/ml-25m";
        String output = args.length > 1 ? args[1] : "third_party/ml-prepared";

        SparkSession spark = SparkSession.builder() //
                .appName("movielens-prepare") //
                .master("local[*]") //
                .config("spark.sql.shuffle.partitions", "32") //
                .config("spark.driver.memory", "8g") //
                .getOrCreate();
        spark.sparkContext().setLogLevel("WARN");

        Dataset<Row> genome = spark.read().option("header", "true").option("inferSchema", "true") //
                .csv(input + "/genome-scores.csv");
        System.out.println("genome-scores rows: " + genome.count());

        // sort_array over a struct orders by its first field, so this is relevance ordered by tagId.
        // Doing it in SQL keeps the whole pivot on Spark rather than collecting anything to the driver.
        genome.createOrReplaceTempView("genome");
        Dataset<Row> vectors = spark.sql(
                "SELECT movieId, " //
                        + "  transform(sort_array(collect_list(struct(tagId, relevance))), x -> cast(x.relevance as double)) AS features " //
                        + "FROM genome GROUP BY movieId");

        Dataset<Row> movies = spark.read().option("header", "true").option("inferSchema", "true") //
                .csv(input + "/movies.csv");

        Dataset<Row> ratings = spark.read().option("header", "true").option("inferSchema", "true") //
                .csv(input + "/ratings.csv");
        Dataset<Row> popularity = ratings.groupBy("movieId").count().withColumnRenamed("count", "nRatings");
        System.out.println("ratings rows: " + ratings.count());

        Dataset<Row> joined = vectors.join(movies, "movieId").join(popularity, "movieId") //
                .selectExpr("cast(movieId as int) as movieId", "title", "genres", //
                        "cast(nRatings as int) as nRatings", "features");

        // Every vector must be the full genome, and no element may be null: both are conditions the
        // scoring UDF and its GPU compilation rely on, and it is cheaper to find out here.
        long wrongLength = joined.filter("size(features) <> " + GENOME_TAGS).count();
        long anyNull = joined.filter("exists(features, x -> x is null)").count();
        System.out.println("vectors with wrong length: " + wrongLength);
        System.out.println("vectors containing a null: " + anyNull);
        if (wrongLength != 0 || anyNull != 0) {
            throw new IllegalStateException("the prepared vectors are not uniformly " + GENOME_TAGS + "-dimensional and non-null");
        }

        // Tighten the schema before writing. A Parquet column is nullable by default, and a
        // nullable column is refused by the kernel compiler because a null has no defined meaning
        // inside a kernel -- so the vectors are declared non-null here, where it can actually be
        // checked, rather than asserted later where it cannot. Rows carrying a null in any of these
        // fields are dropped first, so the declaration is true rather than merely asserted.
        Dataset<Row> complete = joined.na().drop();
        System.out.println("rows dropped for nulls: " + (joined.count() - complete.count()));
        StructType tight = new StructType() //
                .add("movieId", DataTypes.IntegerType, false) //
                .add("title", DataTypes.StringType, false) //
                .add("genres", DataTypes.StringType, false) //
                .add("nRatings", DataTypes.IntegerType, false) //
                .add("features", DataTypes.createArrayType(DataTypes.DoubleType, false), false);
        Dataset<Row> tightened = spark.createDataFrame(complete.toJavaRDD(), tight);

        // NVIDIA's published UDF is typed on float, not double, so the native baseline needs a
        // float copy of the same vectors. Written here, offline, so that no baseline pays a cast
        // inside its timed query. The precision difference is real and is reported with the result:
        // the native baseline moves half the bytes and does single-precision arithmetic.
        tightened.selectExpr("movieId", "title", "genres", "nRatings", "features", //
                "transform(features, x -> cast(x as float)) AS features_float") //
                .repartition(16).write().mode(SaveMode.Overwrite).parquet(output + "/movie_vectors");
        Dataset<Row> readBack = spark.read().parquet(output + "/movie_vectors");
        System.out.println("movie_vectors rows written: " + readBack.count());
        System.out.println("schema as read back (nullability is what the compiler checks):");
        readBack.schema().printTreeString();
        System.out.println("wrote " + output + "/movie_vectors");
        spark.stop();
    }

    private PrepareData() {
    }
}
