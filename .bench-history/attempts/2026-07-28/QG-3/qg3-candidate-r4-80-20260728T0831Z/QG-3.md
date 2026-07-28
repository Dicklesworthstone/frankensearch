fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
watch/medium/initial | quill | docs_per_second (docs/s) | 16171.453038 | [16002.143300, 16334.083714] | 17180.590884 | 17639.552407 | 4.092 | 80 | sampled
watch/medium/initial | tantivy | docs_per_second (docs/s) | 75882.440351 | [73484.675915, 82461.409615] | 95301.779881 | 106839.500043 | 19.524 | 80 | sampled
watch/medium/initial | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.205855 | [0.195520, 0.221912] | 0.341612 | 0.386492 | 22.484 | 80 | sampled
watch/medium/initial | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 1.012897 | [0.980923, 1.049585] | 1.410178 | 1.569828 | 19.492 | 80 | sampled
watch/medium/5000/inprocess | quill | updates_per_second (docs/s) | 12609.528060 | [12416.342626, 12807.396359] | 13915.874514 | 14525.280906 | 8.221 | 80 | sampled
watch/medium/5000/inprocess | tantivy | updates_per_second (docs/s) | 31672.036247 | [30464.683484, 33233.180586] | 39820.916102 | 42384.534948 | 20.397 | 80 | sampled
watch/medium/5000/inprocess | paired_ab | updates_per_second_quill_over_tantivy (ratio) | 0.399770 | [0.373838, 0.417532] | 0.619446 | 0.685124 | 23.172 | 80 | sampled
watch/medium/5000/inprocess | paired_null | updates_per_second_tantivy_over_tantivy (ratio) | 1.011143 | [0.960098, 1.042936] | 1.794313 | 2.131415 | 31.320 | 80 | sampled
watch/medium/5000/inprocess | quill | update_to_searchable_ms (ms) | 415.533602 | [407.574889, 418.018842] | 470.159985 | 492.386920 | 8.262 | 80 | sampled
watch/medium/5000/inprocess | tantivy | update_to_searchable_ms (ms) | 163.090881 | [154.825748, 178.181229] | 299.891700 | 403.757959 | 34.683 | 80 | sampled
watch/medium/5000/inprocess | paired_ab | update_to_searchable_ms_quill_over_tantivy (ratio) | 2.524324 | [2.305033, 2.587909] | 3.228400 | 3.372012 | 27.261 | 80 | sampled
watch/medium/5000/inprocess | paired_null | update_to_searchable_ms_tantivy_over_tantivy (ratio) | 0.969368 | [0.917111, 1.016174] | 1.672818 | 1.932346 | 33.909 | 80 | sampled
watch/medium/5000/freshprocess | quill | updates_per_second (docs/s) | 10386.016189 | [10068.782897, 10808.695064] | 12089.086656 | 12397.849623 | 13.155 | 80 | sampled
watch/medium/5000/freshprocess | tantivy | updates_per_second (docs/s) | 22786.306141 | [21802.402429, 24127.567360] | 27014.904344 | 27717.500529 | 15.400 | 80 | sampled
watch/medium/5000/freshprocess | paired_ab | updates_per_second_quill_over_tantivy (ratio) | 0.448327 | [0.432051, 0.474433] | 0.708157 | 0.763225 | 21.931 | 80 | sampled
watch/medium/5000/freshprocess | paired_null | updates_per_second_tantivy_over_tantivy (ratio) | 0.982197 | [0.958432, 1.050541] | 1.453269 | 1.748950 | 24.320 | 80 | sampled
watch/medium/5000/freshprocess | quill | update_to_searchable_ms (ms) | 485.114549 | [472.353674, 504.424569] | 609.741690 | 655.193779 | 12.307 | 80 | sampled
watch/medium/5000/freshprocess | tantivy | update_to_searchable_ms (ms) | 232.996455 | [221.901108, 255.832047] | 379.696293 | 428.998537 | 24.712 | 80 | sampled
watch/medium/5000/freshprocess | paired_ab | update_to_searchable_ms_quill_over_tantivy (ratio) | 2.076077 | [1.971864, 2.185709] | 2.926778 | 3.197553 | 22.601 | 80 | sampled
watch/medium/5000/freshprocess | paired_null | update_to_searchable_ms_tantivy_over_tantivy (ratio) | 1.007165 | [0.970949, 1.039810] | 1.565579 | 1.753461 | 26.444 | 80 | sampled
