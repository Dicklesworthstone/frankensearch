host=threadripperje | producer_os=linux | physical_cores=64 | logical_threads=128 | ram_bytes=536069869568 | numa_nodes=1 | process_available_threads=128 | thread_counts_requested=[4] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture=bulk/xlarge/4/positions_on | requested_threads=4 | affinity=0-127 | quill_threads_actually_used=1 | quill_peak_new_workers=4 | tantivy_threads_actually_used=504 | tantivy_peak_new_workers=16
engine=quill | role=subject | version=0.2.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
engine=tantivy | role=oracle | version=0.26.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/4/positions_on | quill | docs_per_second (docs/s) | 35577.625362 | [35352.388793, 35748.093718] | 35814.711510 | 35814.711510 | 0.798 | 10 | sampled
bulk/xlarge/4/positions_on | tantivy | docs_per_second (docs/s) | 162203.108330 | [158986.996039, 165975.632091] | 176034.151248 | 176034.151248 | 3.891 | 10 | sampled
bulk/xlarge/4/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.219794 | [0.214915, 0.221372] | 0.236992 | 0.236992 | 3.864 | 10 | sampled
bulk/xlarge/4/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.015860 | [0.878635, 1.125528] | 1.159932 | 1.159932 | 10.836 | 10 | sampled
bulk/xlarge/4/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.995608 | [0.993103, 1.000167] | 1.003960 | 1.003960 | 0.843 | 10 | sampled
