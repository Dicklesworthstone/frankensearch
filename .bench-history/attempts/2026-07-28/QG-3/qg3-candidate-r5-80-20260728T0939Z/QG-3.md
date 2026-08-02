fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
watch/medium/initial | quill | docs_per_second (docs/s) | 8953.808234 | [8854.954987, 9018.820019] | 9330.796412 | 9371.068378 | 3.743 | 80 | sampled
watch/medium/initial | tantivy | docs_per_second (docs/s) | 49295.282585 | [46295.560780, 51052.916487] | 61695.920843 | 62335.832507 | 15.163 | 80 | sampled
watch/medium/initial | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.181260 | [0.171132, 0.187836] | 0.233673 | 0.267230 | 15.789 | 80 | sampled
watch/medium/initial | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.985670 | [0.947772, 1.064356] | 1.347792 | 1.812537 | 21.941 | 80 | sampled
watch/medium/5000/inprocess | quill | updates_per_second (docs/s) | 6172.878885 | [6103.634024, 6367.462677] | 6757.053565 | 7087.938339 | 8.367 | 80 | sampled
watch/medium/5000/inprocess | tantivy | updates_per_second (docs/s) | 19990.576571 | [17827.577907, 20811.729891] | 24211.875371 | 25491.154516 | 19.562 | 80 | sampled
watch/medium/5000/inprocess | paired_ab | updates_per_second_quill_over_tantivy (ratio) | 0.317258 | [0.300283, 0.343210] | 0.472589 | 0.533310 | 23.187 | 80 | sampled
watch/medium/5000/inprocess | paired_null | updates_per_second_tantivy_over_tantivy (ratio) | 0.965468 | [0.916092, 1.038566] | 1.439584 | 1.938251 | 26.741 | 80 | sampled
watch/medium/5000/inprocess | quill | update_to_searchable_ms (ms) | 804.908986 | [795.275606, 813.870940] | 922.191524 | 982.941732 | 6.840 | 80 | sampled
watch/medium/5000/inprocess | tantivy | update_to_searchable_ms (ms) | 262.225297 | [253.275818, 273.024096] | 382.998924 | 422.812499 | 19.688 | 80 | sampled
watch/medium/5000/inprocess | paired_ab | update_to_searchable_ms_quill_over_tantivy (ratio) | 3.117048 | [2.982638, 3.256942] | 3.809621 | 4.048426 | 17.221 | 80 | sampled
watch/medium/5000/inprocess | paired_null | update_to_searchable_ms_tantivy_over_tantivy (ratio) | 0.972325 | [0.914048, 1.101160] | 1.648171 | 2.197071 | 34.908 | 80 | sampled
watch/medium/5000/freshprocess | quill | updates_per_second (docs/s) | 5377.310482 | [5157.907523, 5493.240606] | 5936.769602 | 6047.928225 | 10.653 | 80 | sampled
watch/medium/5000/freshprocess | tantivy | updates_per_second (docs/s) | 14309.687258 | [13413.579338, 14794.439949] | 16903.980922 | 17586.065466 | 16.975 | 80 | sampled
watch/medium/5000/freshprocess | paired_ab | updates_per_second_quill_over_tantivy (ratio) | 0.374237 | [0.366403, 0.385988] | 0.577065 | 0.659093 | 22.645 | 80 | sampled
watch/medium/5000/freshprocess | paired_null | updates_per_second_tantivy_over_tantivy (ratio) | 0.992438 | [0.948522, 1.076851] | 1.342203 | 1.464142 | 20.567 | 80 | sampled
watch/medium/5000/freshprocess | quill | update_to_searchable_ms (ms) | 959.003791 | [945.593253, 1008.928668] | 1193.980026 | 1354.569103 | 14.944 | 80 | sampled
watch/medium/5000/freshprocess | tantivy | update_to_searchable_ms (ms) | 385.182358 | [363.899498, 434.005799] | 580.290520 | 951.434431 | 48.144 | 80 | sampled
watch/medium/5000/freshprocess | paired_ab | update_to_searchable_ms_quill_over_tantivy (ratio) | 2.460350 | [2.371425, 2.592389] | 3.443386 | 4.394269 | 25.566 | 80 | sampled
watch/medium/5000/freshprocess | paired_null | update_to_searchable_ms_tantivy_over_tantivy (ratio) | 1.008166 | [0.959198, 1.095118] | 1.405522 | 1.539890 | 23.475 | 80 | sampled
