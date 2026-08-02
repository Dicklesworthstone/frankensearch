fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
watch/medium/initial | quill | docs_per_second (docs/s) | 54320.045560 | [53040.835511, 55818.039327] | 56677.641555 | 56677.641555 | 3.129 | 10 | sampled
watch/medium/initial | tantivy | docs_per_second (docs/s) | 160871.099263 | [154231.455818, 162229.718205] | 167288.453872 | 167288.453872 | 3.513 | 10 | sampled
watch/medium/initial | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.342475 | [0.333899, 0.349840] | 0.379563 | 0.379563 | 5.306 | 10 | sampled
watch/medium/initial | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.984855 | [0.949677, 1.065982] | 1.173069 | 1.173069 | 11.026 | 10 | sampled
watch/medium/5000/inprocess | quill | updates_per_second (docs/s) | 28763.268171 | [28484.795204, 28978.713720] | 29355.515049 | 29355.515049 | 1.130 | 10 | sampled
watch/medium/5000/inprocess | tantivy | updates_per_second (docs/s) | 54425.323200 | [51693.805966, 57651.581027] | 61749.685743 | 61749.685743 | 6.790 | 10 | sampled
watch/medium/5000/inprocess | paired_ab | updates_per_second_quill_over_tantivy (ratio) | 0.525371 | [0.491037, 0.559821] | 0.581539 | 0.581539 | 6.714 | 10 | sampled
watch/medium/5000/inprocess | paired_null | updates_per_second_tantivy_over_tantivy (ratio) | 0.932129 | [0.890761, 0.994125] | 1.069049 | 1.069049 | 6.753 | 10 | sampled
watch/medium/5000/inprocess | quill | update_to_searchable_ms (ms) | 174.644545 | [172.927333, 176.012212] | 177.921935 | 177.921935 | 1.133 | 10 | sampled
watch/medium/5000/inprocess | tantivy | update_to_searchable_ms (ms) | 87.249209 | [82.496479, 95.843321] | 103.355518 | 103.355518 | 8.660 | 10 | sampled
watch/medium/5000/inprocess | paired_ab | update_to_searchable_ms_quill_over_tantivy (ratio) | 1.990237 | [1.827415, 2.111947] | 2.217593 | 2.217593 | 8.626 | 10 | sampled
watch/medium/5000/inprocess | paired_null | update_to_searchable_ms_tantivy_over_tantivy (ratio) | 0.977788 | [0.936702, 1.022433] | 1.038695 | 1.038695 | 5.335 | 10 | sampled
watch/medium/5000/freshprocess | quill | updates_per_second (docs/s) | 16873.324557 | [16436.958926, 17170.775310] | 17497.092010 | 17497.092010 | 2.822 | 10 | sampled
watch/medium/5000/freshprocess | tantivy | updates_per_second (docs/s) | 21258.313320 | [20912.551477, 21466.289232] | 21569.082043 | 21569.082043 | 1.900 | 10 | sampled
watch/medium/5000/freshprocess | paired_ab | updates_per_second_quill_over_tantivy (ratio) | 0.793734 | [0.777294, 0.810490] | 0.859272 | 0.859272 | 3.874 | 10 | sampled
watch/medium/5000/freshprocess | paired_null | updates_per_second_tantivy_over_tantivy (ratio) | 1.019121 | [0.999459, 1.045221] | 1.065975 | 1.065975 | 2.650 | 10 | sampled
watch/medium/5000/freshprocess | quill | update_to_searchable_ms (ms) | 301.342617 | [296.258751, 307.818661] | 332.835148 | 332.835148 | 3.555 | 10 | sampled
watch/medium/5000/freshprocess | tantivy | update_to_searchable_ms (ms) | 239.820156 | [236.280982, 247.958463] | 254.812147 | 254.812147 | 2.863 | 10 | sampled
watch/medium/5000/freshprocess | paired_ab | update_to_searchable_ms_quill_over_tantivy (ratio) | 1.245755 | [1.210251, 1.295966] | 1.400849 | 1.400849 | 4.894 | 10 | sampled
watch/medium/5000/freshprocess | paired_null | update_to_searchable_ms_tantivy_over_tantivy (ratio) | 1.003092 | [0.993262, 1.039123] | 1.052645 | 1.052645 | 2.621 | 10 | sampled
