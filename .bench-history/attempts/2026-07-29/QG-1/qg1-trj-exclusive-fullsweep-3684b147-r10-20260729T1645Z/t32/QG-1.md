host=threadripperje | producer_os=linux | physical_cores=64 | logical_threads=128 | ram_bytes=536069869568 | numa_nodes=1 | process_available_threads=128 | thread_counts_requested=[32] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture=bulk/xlarge/32/positions_on | requested_threads=32 | affinity=0-127 | quill_threads_actually_used=1 | quill_peak_new_workers=32 | tantivy_threads_actually_used=984 | tantivy_peak_new_workers=75
engine=quill | role=subject | version=0.2.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
engine=tantivy | role=oracle | version=0.26.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/32/positions_on | quill | docs_per_second (docs/s) | 27429.558781 | [27404.440426, 27552.315106] | 27649.146099 | 27649.146099 | 0.349 | 10 | sampled
bulk/xlarge/32/positions_on | tantivy | docs_per_second (docs/s) | 125514.969759 | [121397.273951, 131527.561800] | 138390.074877 | 138390.074877 | 5.112 | 10 | sampled
bulk/xlarge/32/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.218925 | [0.209413, 0.227571] | 0.234344 | 0.234344 | 4.938 | 10 | sampled
bulk/xlarge/32/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.989875 | [0.959044, 1.030804] | 1.145702 | 1.145702 | 6.036 | 10 | sampled
bulk/xlarge/32/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.991570 | [0.974924, 1.005959] | 1.022171 | 1.022171 | 1.722 | 10 | sampled
