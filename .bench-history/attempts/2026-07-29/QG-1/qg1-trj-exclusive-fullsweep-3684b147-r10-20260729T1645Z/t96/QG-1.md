host=threadripperje | producer_os=linux | physical_cores=64 | logical_threads=128 | ram_bytes=536069869568 | numa_nodes=1 | process_available_threads=128 | thread_counts_requested=[96] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture=bulk/xlarge/96/positions_on | requested_threads=96 | affinity=0-127 | quill_threads_actually_used=1 | quill_peak_new_workers=96 | tantivy_threads_actually_used=2759 | tantivy_peak_new_workers=193
engine=quill | role=subject | version=0.2.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
engine=tantivy | role=oracle | version=0.26.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/96/positions_on | quill | docs_per_second (docs/s) | 29268.189123 | [29139.313907, 29368.691808] | 29492.650171 | 29492.650171 | 0.560 | 10 | sampled
bulk/xlarge/96/positions_on | tantivy | docs_per_second (docs/s) | 113641.286156 | [112137.001421, 115957.408528] | 117530.614800 | 117530.614800 | 2.111 | 10 | sampled
bulk/xlarge/96/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.256653 | [0.253521, 0.259777] | 0.265396 | 0.265396 | 1.647 | 10 | sampled
bulk/xlarge/96/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.032392 | [0.964372, 1.037515] | 1.065079 | 1.065079 | 3.566 | 10 | sampled
bulk/xlarge/96/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.989442 | [0.980477, 0.998565] | 1.013013 | 1.013013 | 1.126 | 10 | sampled
