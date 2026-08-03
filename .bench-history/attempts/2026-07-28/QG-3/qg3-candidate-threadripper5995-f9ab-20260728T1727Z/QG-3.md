fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
watch/medium/initial | quill | docs_per_second (docs/s) | 59323.915347 | [58447.394095, 59839.281779] | 60870.844426 | 60870.844426 | 1.607 | 10 | sampled
watch/medium/initial | tantivy | docs_per_second (docs/s) | 183101.462314 | [174808.544787, 193260.130102] | 201331.572391 | 201331.572391 | 6.425 | 10 | sampled
watch/medium/initial | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.321658 | [0.301506, 0.343022] | 0.359052 | 0.359052 | 6.419 | 10 | sampled
watch/medium/initial | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.999277 | [0.945651, 1.038543] | 1.062789 | 1.062789 | 8.918 | 10 | sampled
watch/medium/5000/inprocess | quill | updates_per_second (docs/s) | 31390.789180 | [30618.168264, 31596.140051] | 32027.945997 | 32027.945997 | 2.692 | 10 | sampled
watch/medium/5000/inprocess | tantivy | updates_per_second (docs/s) | 59010.466232 | [57686.572496, 63041.705269] | 65594.937225 | 65594.937225 | 5.034 | 10 | sampled
watch/medium/5000/inprocess | paired_ab | updates_per_second_quill_over_tantivy (ratio) | 0.521883 | [0.489708, 0.542868] | 0.556593 | 0.556593 | 5.276 | 10 | sampled
watch/medium/5000/inprocess | paired_null | updates_per_second_tantivy_over_tantivy (ratio) | 1.037966 | [0.954068, 1.637242] | 2.340983 | 2.340983 | 41.639 | 10 | sampled
watch/medium/5000/inprocess | quill | update_to_searchable_ms (ms) | 159.752749 | [158.236105, 167.272595] | 176.920761 | 176.920761 | 3.686 | 10 | sampled
watch/medium/5000/inprocess | tantivy | update_to_searchable_ms (ms) | 78.544058 | [76.021523, 80.629826] | 84.724489 | 84.724489 | 3.894 | 10 | sampled
watch/medium/5000/inprocess | paired_ab | update_to_searchable_ms_quill_over_tantivy (ratio) | 2.063031 | [2.016110, 2.151159] | 2.327246 | 2.327246 | 5.672 | 10 | sampled
watch/medium/5000/inprocess | paired_null | update_to_searchable_ms_tantivy_over_tantivy (ratio) | 1.001309 | [0.875441, 1.038994] | 1.071551 | 1.071551 | 18.792 | 10 | sampled
watch/medium/5000/freshprocess | quill | updates_per_second (docs/s) | 20345.005509 | [20073.426786, 20422.180228] | 20605.826719 | 20605.826719 | 2.658 | 10 | sampled
watch/medium/5000/freshprocess | tantivy | updates_per_second (docs/s) | 34561.966344 | [34124.299221, 35243.779293] | 35730.284715 | 35730.284715 | 1.757 | 10 | sampled
watch/medium/5000/freshprocess | paired_ab | updates_per_second_quill_over_tantivy (ratio) | 0.587493 | [0.573602, 0.593042] | 0.595891 | 0.595891 | 2.738 | 10 | sampled
watch/medium/5000/freshprocess | paired_null | updates_per_second_tantivy_over_tantivy (ratio) | 1.019785 | [1.002510, 1.067532] | 1.073468 | 1.073468 | 4.035 | 10 | sampled
watch/medium/5000/freshprocess | quill | update_to_searchable_ms (ms) | 246.390102 | [245.223845, 250.740966] | 267.791663 | 267.791663 | 2.697 | 10 | sampled
watch/medium/5000/freshprocess | tantivy | update_to_searchable_ms (ms) | 144.985173 | [143.389583, 146.286754] | 147.092532 | 147.092532 | 1.749 | 10 | sampled
watch/medium/5000/freshprocess | paired_ab | update_to_searchable_ms_quill_over_tantivy (ratio) | 1.712915 | [1.684956, 1.762627] | 1.845607 | 1.845607 | 3.009 | 10 | sampled
watch/medium/5000/freshprocess | paired_null | update_to_searchable_ms_tantivy_over_tantivy (ratio) | 0.991915 | [0.970146, 1.032737] | 1.068784 | 1.068784 | 3.943 | 10 | sampled
