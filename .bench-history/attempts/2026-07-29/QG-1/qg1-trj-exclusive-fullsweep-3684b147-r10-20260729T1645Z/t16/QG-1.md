host=threadripperje | producer_os=linux | physical_cores=64 | logical_threads=128 | ram_bytes=536069869568 | numa_nodes=1 | process_available_threads=128 | thread_counts_requested=[16] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture=bulk/xlarge/16/positions_on | requested_threads=16 | affinity=0-127 | quill_threads_actually_used=1 | quill_peak_new_workers=16 | tantivy_threads_actually_used=720 | tantivy_peak_new_workers=39
engine=quill | role=subject | version=0.2.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
engine=tantivy | role=oracle | version=0.26.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/16/positions_on | quill | docs_per_second (docs/s) | 32733.036071 | [32348.952781, 32882.648591] | 33902.891669 | 33902.891669 | 2.579 | 10 | sampled
bulk/xlarge/16/positions_on | tantivy | docs_per_second (docs/s) | 132850.796768 | [123990.166675, 140085.883285] | 140296.468450 | 140296.468450 | 6.189 | 10 | sampled
bulk/xlarge/16/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.242967 | [0.234367, 0.263901] | 0.273976 | 0.273976 | 6.359 | 10 | sampled
bulk/xlarge/16/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.989684 | [0.906424, 1.019070] | 1.060819 | 1.060819 | 5.942 | 10 | sampled
bulk/xlarge/16/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.998416 | [0.994763, 1.000946] | 1.007387 | 1.007387 | 0.464 | 10 | sampled
