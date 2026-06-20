# Benchmark Generator

We leverage LLMs to generate benchmarks for table discovery systems. A benchmark
is a list of $(Q,[T])$, where $Q$ is a content or context question and $[T]$ is a list of tables that are relevant to answer $Q$. There are two kinds of benchmarks: content and context benchmarks. The former focuses on content questions ($Q_c$), while the latter focuses on context questions ($Q_x$). You can generate each benchmark in the corresponding folder (`content/context`). Each folder contains instructions to produce or download generated benchmarks.
