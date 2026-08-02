host=threadripperje | producer_os=linux | physical_cores=64 | logical_threads=128 | ram_bytes=536069869568 | numa_nodes=1 | process_available_threads=128 | thread_counts_requested=[2] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture=bulk/xlarge/2/positions_on | requested_threads=2 | affinity=0-127 | quill_threads_actually_used=1 | quill_peak_new_workers=2 | tantivy_threads_actually_used=85 | tantivy_peak_new_workers=10
engine=quill | role=subject | version=0.2.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
engine=tantivy | role=oracle | version=0.26.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/2/positions_on | quill | docs_per_second (docs/s) | 37252.012182 | [37131.479014, 37631.512257] | 37820.932734 | 37820.932734 | 0.776 | 10 | sampled
bulk/xlarge/2/positions_on | tantivy | docs_per_second (docs/s) | 196637.499775 | [193115.496062, 201252.518029] | 207001.769567 | 207001.769567 | 2.620 | 10 | sampled
bulk/xlarge/2/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.189424 | [0.183447, 0.193405] | 0.197529 | 0.197529 | 2.911 | 10 | sampled
bulk/xlarge/2/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.974636 | [0.929520, 1.005964] | 1.046304 | 1.046304 | 5.161 | 10 | sampled
bulk/xlarge/2/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 1.003950 | [0.995970, 1.012961] | 1.018662 | 1.018662 | 0.958 | 10 | sampled
