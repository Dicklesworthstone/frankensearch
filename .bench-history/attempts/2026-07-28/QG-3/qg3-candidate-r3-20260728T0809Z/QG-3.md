fixture | engine | metric | p50 | median_ci95 | p95 | p99 | cv_pct (provenance) | runs | admission
--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---
watch/medium/initial | quill | docs_per_second (docs/s) | 18755.484567 | [17691.199327, 19392.977013] | 19590.274134 | 19590.274134 | 5.321 | 10 | sampled
watch/medium/initial | tantivy | docs_per_second (docs/s) | 113998.739531 | [103328.065303, 122840.770351] | 126878.840908 | 126878.840908 | 10.280 | 10 | sampled
watch/medium/initial | paired_ab | docs_per_second_quill_over_tantivy (ratio) | 0.162790 | [0.153180, 0.175944] | 0.204110 | 0.204110 | 10.156 | 10 | sampled
watch/medium/initial | paired_null | docs_per_second_tantivy_over_tantivy (ratio) | 0.997960 | [0.913815, 1.131828] | 1.259602 | 1.259602 | 12.164 | 10 | sampled
watch/medium/5000/inprocess | quill | updates_per_second (docs/s) | 14613.811185 | [14045.719452, 16144.437854] | 16883.085139 | 16883.085139 | 6.941 | 10 | sampled
watch/medium/5000/inprocess | tantivy | updates_per_second (docs/s) | 40864.586286 | [34998.918323, 43020.097949] | 47310.109450 | 47310.109450 | 21.847 | 10 | sampled
watch/medium/5000/inprocess | paired_ab | updates_per_second_quill_over_tantivy (ratio) | 0.355016 | [0.341247, 0.494362] | 0.762110 | 0.762110 | 32.474 | 10 | sampled
watch/medium/5000/inprocess | paired_null | updates_per_second_tantivy_over_tantivy (ratio) | 1.147019 | [0.634518, 1.518754] | 1.957991 | 1.957991 | 41.029 | 10 | sampled
watch/medium/5000/inprocess | quill | update_to_searchable_ms (ms) | 337.711863 | [327.925191, 362.232278] | 414.892417 | 414.892417 | 8.669 | 10 | sampled
watch/medium/5000/inprocess | tantivy | update_to_searchable_ms (ms) | 123.437562 | [117.044682, 142.591517] | 229.121793 | 229.121793 | 25.976 | 10 | sampled
watch/medium/5000/inprocess | paired_ab | update_to_searchable_ms_quill_over_tantivy (ratio) | 2.828458 | [2.327995, 2.946656] | 3.056018 | 3.056018 | 15.029 | 10 | sampled
watch/medium/5000/inprocess | paired_null | update_to_searchable_ms_tantivy_over_tantivy (ratio) | 0.974307 | [0.885976, 1.495022] | 1.843351 | 1.843351 | 36.361 | 10 | sampled
watch/medium/5000/freshprocess | quill | updates_per_second (docs/s) | 12886.478331 | [12415.060501, 13436.245114] | 14436.229175 | 14436.229175 | 5.167 | 10 | sampled
watch/medium/5000/freshprocess | tantivy | updates_per_second (docs/s) | 36742.420045 | [33892.956726, 38838.710138] | 41051.037597 | 41051.037597 | 11.605 | 10 | sampled
watch/medium/5000/freshprocess | paired_ab | updates_per_second_quill_over_tantivy (ratio) | 0.351510 | [0.338055, 0.385906] | 0.491188 | 0.491188 | 12.962 | 10 | sampled
watch/medium/5000/freshprocess | paired_null | updates_per_second_tantivy_over_tantivy (ratio) | 1.007225 | [0.903046, 1.090036] | 1.171235 | 1.171235 | 11.063 | 10 | sampled
watch/medium/5000/freshprocess | quill | update_to_searchable_ms (ms) | 398.792190 | [391.042816, 440.591075] | 516.244037 | 516.244037 | 10.134 | 10 | sampled
watch/medium/5000/freshprocess | tantivy | update_to_searchable_ms (ms) | 142.719954 | [132.675053, 149.815779] | 153.062458 | 153.062458 | 6.696 | 10 | sampled
watch/medium/5000/freshprocess | paired_ab | update_to_searchable_ms_quill_over_tantivy (ratio) | 2.916745 | [2.644519, 3.143026] | 4.052986 | 4.052986 | 14.254 | 10 | sampled
watch/medium/5000/freshprocess | paired_null | update_to_searchable_ms_tantivy_over_tantivy (ratio) | 1.009328 | [0.878845, 1.096205] | 1.297479 | 1.297479 | 13.930 | 10 | sampled
