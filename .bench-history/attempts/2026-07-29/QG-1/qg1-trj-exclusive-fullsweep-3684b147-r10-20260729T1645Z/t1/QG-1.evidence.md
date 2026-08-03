gate QG-1 | status no_decision | decision none | run qg1-trj-exclusive-xlarge-t1-3684b147-r10-20260729T1645Z | window qg1-trj-exclusive-fullsweep-3684b147-20260729T1645Z
host threadripperje | producer_os linux | physical_cores 64 | logical_threads 128 | ram_bytes 536069869568 | numa_nodes 1 | requested_threads [1] | affinity 0-127
thread_row bulk/xlarge/1/positions_on | requested 1 | affinity 0-127 | quill_actual 1 | quill_cpu_active_new 1 | quill_peak_new 1 | tantivy_actual 39 | tantivy_cpu_active_new 39 | tantivy_peak_new 8
engine quill | role subject | version 0.2.1 | executable_sha256 90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme same-static-elf-v1
engine tantivy | role oracle | version 0.26.1 | executable_sha256 90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a | scheme same-static-elf-v1
cell | role | estimand | status | control_p50 | treatment_p50 | ratio | ci95_ratio | pairs | reasons
--- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---:
QG-1/bulk/xlarge/1/positions_on/docs_per_second | required | paired_log_ratio | measured_provisional | 139186.054699 | 35371.869614 | 0.257438 | [0.250622, 0.262109] | 10 | 0
