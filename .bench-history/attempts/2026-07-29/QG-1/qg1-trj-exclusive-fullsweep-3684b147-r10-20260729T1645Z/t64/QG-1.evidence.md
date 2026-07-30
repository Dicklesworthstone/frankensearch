gate QG-1 | status no_decision | decision none | run qg1-trj-exclusive-xlarge-t64-3684b147-r10-20260729T1645Z | window qg1-trj-exclusive-fullsweep-3684b147-20260729T1645Z
host threadripperje | producer_os linux | physical_cores 64 | logical_threads 128 | ram_bytes 536069869568 | numa_nodes 1 | requested_threads [64] | affinity 0-127
thread_row bulk/xlarge/64/positions_on | requested 64 | affinity 0-127 | quill_actual 1 | quill_cpu_active_new 1 | quill_peak_new 64 | tantivy_actual 1827 | tantivy_cpu_active_new 1827 | tantivy_peak_new 141
engine quill | role subject | version 0.2.1 | executable_sha256 90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme same-static-elf-v1
engine tantivy | role oracle | version 0.26.1 | executable_sha256 90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme same-static-elf-v1
cell | role | estimand | status | control_p50 | treatment_p50 | ratio | ci95_ratio | pairs | reasons
--- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---:
QG-1/bulk/xlarge/64/positions_on/docs_per_second | required | paired_log_ratio | measured_provisional | 122979.243705 | 27422.718809 | 0.219276 | [0.217200, 0.226471] | 10 | 0
