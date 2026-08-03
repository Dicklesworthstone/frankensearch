host=threadripperje | producer_os=linux | physical_cores=64 | logical_threads=128 | ram_bytes=536069869568 | numa_nodes=1 | process_available_threads=128 | thread_counts_requested=[1] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture=bulk/xlarge/1/positions_on | requested_threads=1 | affinity=0-127 | quill_threads_actually_used=1 | quill_peak_new_workers=1 | tantivy_threads_actually_used=39 | tantivy_peak_new_workers=8
engine=quill | role=subject | version=0.2.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
engine=tantivy | role=oracle | version=0.26.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/1/positions_on | quill | docs_per_second (docs/s) | 35371.869614 | [35022.491736, 36044.670839] | 36364.112665 | 36364.112665 | 1.553 | 10 | sampled
bulk/xlarge/1/positions_on | tantivy | docs_per_second (docs/s) | 139186.054699 | [134345.673572, 141030.891826] | 144262.932686 | 144262.932686 | 2.563 | 10 | sampled
bulk/xlarge/1/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.257442 | [0.250271, 0.262109] | 0.264055 | 0.264055 | 2.311 | 10 | sampled
bulk/xlarge/1/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.998882 | [0.975242, 1.036296] | 1.071200 | 1.071200 | 3.581 | 10 | sampled
bulk/xlarge/1/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.997971 | [0.980798, 1.010896] | 1.026845 | 1.026845 | 1.894 | 10 | sampled
