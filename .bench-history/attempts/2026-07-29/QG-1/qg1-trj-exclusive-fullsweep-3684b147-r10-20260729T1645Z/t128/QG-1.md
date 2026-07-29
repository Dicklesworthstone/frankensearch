host=threadripperje | producer_os=linux | physical_cores=64 | logical_threads=128 | ram_bytes=536069869568 | numa_nodes=1 | process_available_threads=128 | thread_counts_requested=[128] | runtime_detected_isa=["avx2", "fma", "bmi2", "aes", "vaes"] | cpu_affinity_allowed_list=0-127 | affinity_or_cpuset_cap=none
fixture=bulk/xlarge/128/positions_on | requested_threads=128 | affinity=0-127 | quill_threads_actually_used=1 | quill_peak_new_workers=128 | tantivy_threads_actually_used=3556 | tantivy_peak_new_workers=264
engine=quill | role=subject | version=0.2.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
engine=tantivy | role=oracle | version=0.26.1 | executable_sha256=90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme=same-static-elf-v1
fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
bulk/xlarge/128/positions_on | quill | docs_per_second (docs/s) | 27130.469368 | [26710.769746, 27426.274965] | 27785.328769 | 27785.328769 | 1.888 | 10 | sampled
bulk/xlarge/128/positions_on | tantivy | docs_per_second (docs/s) | 109413.724007 | [106913.214232, 111899.222132] | 114879.429704 | 114879.429704 | 3.856 | 10 | sampled
bulk/xlarge/128/positions_on | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.249378 | [0.236671, 0.258318] | 0.271371 | 0.271371 | 4.556 | 10 | sampled
bulk/xlarge/128/positions_on | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.982049 | [0.951422, 1.013386] | 1.050853 | 1.050853 | 4.393 | 10 | sampled
bulk/xlarge/128/positions_on | paired_null_quill | docs_per_second_quill_over_quill (ratio) | 0.990940 | [0.982398, 1.017325] | 1.050481 | 1.050481 | 2.309 | 10 | sampled
