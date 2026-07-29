host=threadripperje | producer_os=linux | physical_cores=64 | logical_threads=128 | ram_bytes=536069869568 | numa_nodes=1 | process_available_threads=128 | thread_counts_requested=[64] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture=bulk/xlarge/64/positions_on | requested_threads=64 | affinity=0-127 | quill_threads_actually_used=1 | quill_peak_new_workers=64 | tantivy_threads_actually_used=1827 | tantivy_peak_new_workers=141
engine=quill | role=subject | version=0.2.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
engine=tantivy | role=oracle | version=0.26.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/64/positions_on | quill | docs_per_second (docs/s) | 27422.718809 | [26760.997408, 27647.115548] | 27957.525655 | 27957.525655 | 1.666 | 10 | sampled
bulk/xlarge/64/positions_on | tantivy | docs_per_second (docs/s) | 122979.243705 | [120801.570681, 126632.252778] | 128339.843001 | 128339.843001 | 3.106 | 10 | sampled
bulk/xlarge/64/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.219276 | [0.217299, 0.226588] | 0.235649 | 0.235649 | 3.315 | 10 | sampled
bulk/xlarge/64/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.997052 | [0.960788, 1.035182] | 1.071367 | 1.071367 | 4.094 | 10 | sampled
bulk/xlarge/64/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.999136 | [0.987977, 1.005863] | 1.008747 | 1.008747 | 0.943 | 10 | sampled
