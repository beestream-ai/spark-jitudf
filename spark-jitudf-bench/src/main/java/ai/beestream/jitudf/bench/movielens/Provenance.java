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

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

/**
 * Immutable per-execution evidence for a timed run.
 *
 * <p>
 * The problem this solves: the benchmark used to time five queries, keep only each result's row
 * count, then run a <em>sixth, untimed</em> query whose plan and output were saved — and those
 * artifacts were then cited as evidence of what the timed queries did. The filenames also encoded
 * only mode, size, batch and partition count, so a later experiment silently overwrote an earlier
 * one's plan and output. Neither habit survives contact with the question "which execution
 * produced this number".
 * </p>
 *
 * <p>
 * Every timed execution now gets its own digest of its own collected rows, its own executed plan,
 * and a line in a machine-readable manifest keyed by a run identifier that includes the Spark
 * application id. Nothing is overwritten: a file that already exists is an error, not a target.
 * Digests and plan capture happen strictly after the timer stops, so evidence collection cannot
 * inflate the measurement it describes.
 * </p>
 */
public final class Provenance {

    private Provenance() {
    }

    /** SHA-256 of the collected result, order-sensitive, as the lowercase hex of the digest. */
    public static String digest(List<Row> rows) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            StringBuilder sb = new StringBuilder(64);
            for (Row r : rows) {
                sb.setLength(0);
                for (int i = 0; i < r.size(); i++) {
                    // Row.toString() formats doubles in a locale- and version-sensitive way; the
                    // raw value's canonical string is what must be stable across runs.
                    Object v = r.get(i);
                    sb.append(v instanceof Double d ? Double.toString(d) : String.valueOf(v)).append('\u001f');
                }
                md.update(sb.toString().getBytes(StandardCharsets.UTF_8));
                md.update((byte) '\n');
            }
            return hex(md.digest());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /** SHA-256 of a string, as lowercase hex. */
    public static String textHash(String s) {
        try {
            return hex(MessageDigest.getInstance("SHA-256").digest(s.getBytes(StandardCharsets.UTF_8)));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /** SHA-256 of a file, or {@code "absent"} when it is not there. */
    public static String fileHash(Path p) {
        try {
            if (!Files.exists(p)) {
                return "absent";
            }
            return hex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(p)));
        } catch (Exception e) {
            return "unreadable";
        }
    }

    /**
     * SHA-256 over the prepared dataset's file names, sizes and modification times.
     *
     * <p>
     * Hashing the Parquet payload itself would read hundreds of megabytes on every run. The
     * identity that matters here is "is this the same prepared dataset", which the directory
     * listing answers, and a changed payload changes its size or timestamp.
     * </p>
     */
    public static String datasetIdentity(String dataDir) {
        try {
            Path root = Path.of(dataDir);
            if (!Files.isDirectory(root)) {
                return "absent";
            }
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            List<Path> files = new ArrayList<>();
            try (var walk = Files.walk(root)) {
                walk.filter(Files::isRegularFile).sorted().forEach(files::add);
            }
            for (Path f : files) {
                md.update((root.relativize(f) + ":" + Files.size(f) + ":"
                        + Files.getLastModifiedTime(f).toMillis()).getBytes(StandardCharsets.UTF_8));
            }
            return hex(md.digest()) + "/" + files.size() + "files";
        } catch (Exception e) {
            return "unreadable";
        }
    }

    /** The executed physical plan, which is what ran rather than what was asked for. */
    public static String executedPlan(Dataset<Row> q) {
        return q.queryExecution().executedPlan().toString();
    }

    /** Source revision plus a digest of the working tree's modifications, which HEAD alone omits. */
    public static Map<String, String> sourceIdentity(String repoRoot) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("head", git(repoRoot, "rev-parse", "HEAD"));
        m.put("branch", git(repoRoot, "rev-parse", "--abbrev-ref", "HEAD"));
        String status = git(repoRoot, "status", "--porcelain");
        m.put("dirtyPaths", status.isEmpty() ? "0" : String.valueOf(status.split("\n").length));
        try {
            m.put("dirtyDigest", status.isEmpty() ? "clean"
                    : hex(MessageDigest.getInstance("SHA-256").digest(status.getBytes(StandardCharsets.UTF_8))).substring(0, 16));
        } catch (Exception e) {
            m.put("dirtyDigest", "unreadable");
        }
        return m;
    }

    private static String git(String repoRoot, String... args) {
        try {
            List<String> cmd = new ArrayList<>(List.of("git", "-C", repoRoot));
            cmd.addAll(List.of(args));
            Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
            String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
            return p.waitFor() == 0 ? out : "unavailable";
        } catch (Exception e) {
            return "unavailable";
        }
    }

    private static String hex(byte[] b) {
        StringBuilder sb = new StringBuilder(b.length * 2);
        for (byte x : b) {
            sb.append(String.format(Locale.ROOT, "%02x", x));
        }
        return sb.toString();
    }

    /** Writes {@code content} to {@code file}, refusing to overwrite an existing artifact. */
    public static void writeNew(Path file, String content) {
        try {
            Files.createDirectories(file.getParent());
            Files.writeString(file, content, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
        } catch (java.nio.file.FileAlreadyExistsException e) {
            throw new IllegalStateException("refusing to overwrite existing evidence: " + file, e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /** Appends one JSON object per line; a manifest that is append-only cannot lose a run. */
    public static void appendManifest(Path file, Map<String, String> record) {
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, String> e : record.entrySet()) {
            if (!first) {
                sb.append(',');
            }
            first = false;
            sb.append('"').append(escape(e.getKey())).append("\":\"").append(escape(e.getValue())).append('"');
        }
        sb.append("}\n");
        try {
            Files.createDirectories(file.getParent());
            Files.writeString(file, sb.toString(), StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "");
    }
}
