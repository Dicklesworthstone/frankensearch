fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
watch/medium/initial | quill | docs_per_second (docs/s) | 39872.978612 | [32746.835651, 59072.419090] | 63422.278208 | 63422.278208 | 28.455 | 10 | sampled
watch/medium/initial | tantivy | docs_per_second (docs/s) | 131391.797831 | [91623.837413, 192132.248419] | 205269.365608 | 205269.365608 | 34.160 | 10 | sampled
watch/medium/initial | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.341013 | [0.317462, 0.366179] | 0.373940 | 0.373940 | 12.445 | 10 | sampled
watch/medium/initial | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.994765 | [0.987399, 1.050986] | 1.502303 | 1.502303 | 14.347 | 10 | sampled
watch/medium/5000/inprocess | quill | updates_per_second (docs/s) | 31438.344476 | [31104.783237, 31715.963804] | 31983.210094 | 31983.210094 | 1.385 | 10 | sampled
watch/medium/5000/inprocess | tantivy | updates_per_second (docs/s) | 70851.665233 | [68244.122274, 73567.880193] | 75169.892979 | 75169.892979 | 3.675 | 10 | sampled
watch/medium/5000/inprocess | paired_ab | updates_per_second_quill_over_tantivy (ratio) | 0.443997 | [0.426136, 0.459570] | 0.471518 | 0.471518 | 4.265 | 10 | sampled
watch/medium/5000/inprocess | paired_null | updates_per_second_tantivy_over_tantivy (ratio) | 1.008967 | [0.962565, 1.782444] | 2.691287 | 2.691287 | 56.001 | 10 | sampled
watch/medium/5000/inprocess | quill | update_to_searchable_ms (ms) | 306.022131 | [281.047363, 310.817997] | 333.055833 | 333.055833 | 15.826 | 10 | sampled
watch/medium/5000/inprocess | tantivy | update_to_searchable_ms (ms) | 144.993565 | [135.609543, 163.471662] | 258.586496 | 258.586496 | 25.767 | 10 | sampled
watch/medium/5000/inprocess | paired_ab | update_to_searchable_ms_quill_over_tantivy (ratio) | 2.047944 | [1.709623, 2.242787] | 2.309536 | 2.309536 | 17.641 | 10 | sampled
watch/medium/5000/inprocess | paired_null | update_to_searchable_ms_tantivy_over_tantivy (ratio) | 0.955202 | [0.673377, 1.011446] | 1.046373 | 1.046373 | 27.281 | 10 | sampled
watch/medium/5000/freshprocess | quill | updates_per_second (docs/s) | 21969.303240 | [21429.786253, 22628.845376] | 25053.221561 | 25053.221561 | 4.995 | 10 | sampled
watch/medium/5000/freshprocess | tantivy | updates_per_second (docs/s) | 55627.692072 | [53757.810741, 56594.073069] | 57975.590699 | 57975.590699 | 2.786 | 10 | sampled
watch/medium/5000/freshprocess | paired_ab | updates_per_second_quill_over_tantivy (ratio) | 0.397593 | [0.389614, 0.405902] | 0.445116 | 0.445116 | 4.146 | 10 | sampled
watch/medium/5000/freshprocess | paired_null | updates_per_second_tantivy_over_tantivy (ratio) | 0.987540 | [0.592760, 1.041254] | 1.081851 | 1.081851 | 38.642 | 10 | sampled
watch/medium/5000/freshprocess | quill | update_to_searchable_ms (ms) | 224.340131 | [203.515805, 247.922746] | 346.133570 | 346.133570 | 17.733 | 10 | sampled
watch/medium/5000/freshprocess | tantivy | update_to_searchable_ms (ms) | 105.775004 | [99.973619, 139.281143] | 184.166136 | 184.166136 | 22.948 | 10 | sampled
watch/medium/5000/freshprocess | paired_ab | update_to_searchable_ms_quill_over_tantivy (ratio) | 1.927117 | [1.813852, 2.127288] | 2.416411 | 2.416411 | 10.336 | 10 | sampled
watch/medium/5000/freshprocess | paired_null | update_to_searchable_ms_tantivy_over_tantivy (ratio) | 1.055579 | [0.996594, 1.179674] | 1.475061 | 1.475061 | 17.985 | 10 | sampled
