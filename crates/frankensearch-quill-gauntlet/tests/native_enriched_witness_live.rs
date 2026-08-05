//! bd-8nqz.4.1: run the native enriched witness against the REAL engines.
//!
//! The unit tests beside the oracle exercise the adjudicator on synthetic
//! observations, which proves the oracle is strict but proves nothing about
//! whether its committed expectations are TRUE of the shipping engines. That
//! is the gap this suite closes: it builds the committed corpus in a real
//! `QuillIndex` (and, under `tantivy-oracle`, a real `TantivyIndex`), drives
//! each engine's own native paginated API, and adjudicates the result against
//! the hand-derived expectations.
//!
//! An expectation that is wrong about BM25 fails HERE, loudly, instead of
//! sitting in a green unit suite describing an engine nobody ran.

use frankensearch_core::IndexableDocument;
use frankensearch_quill::{QuillConfig, QuillIndex};
use frankensearch_quill_gauntlet::native_enriched_witness::{
    EnrichedExpectationV1, FIXTURE_CORPUS, FIXTURE_ENRICHMENT_EXPECTATIONS, FIXTURE_EXPECTATIONS,
    FIXTURE_METADATA, UTF8_DOC_ID, UTF8_INTACT_TOKEN, adjudicate, adjudicate_enrichment,
    adjudicate_truncation_determinism, adjudicate_utf8_window, observe_quill,
    observe_quill_enrichment, truncation_probe_queries,
};

/// Build the committed corpus in a real Quill index.
async fn build_quill(cx: &asupersync::Cx, dir: &std::path::Path) -> QuillIndex {
    let index = QuillIndex::create(
        cx,
        dir,
        QuillConfig {
            bulk_load_mode: true,
            deterministic_ingest: true,
            max_ingest_shards: 1,
            ..QuillConfig::default()
        },
    )
    .await
    .expect("create the witness Quill index");
    for document in fixture_documents() {
        index
            .index_document(cx, &document)
            .await
            .expect("index a witness fixture document");
    }
    index
        .finish_bulk_load(cx)
        .await
        .expect("finalize the witness Quill index");
    index
}

/// The committed corpus as indexable documents, with the metadata the
/// enrichment expectations describe.
///
/// Both engines are fed EXACTLY this list, so a metadata or snippet
/// divergence is a divergence in the engines, not in how they were loaded.
fn fixture_documents() -> Vec<IndexableDocument> {
    FIXTURE_CORPUS
        .iter()
        .map(|(doc_id, body)| {
            let mut document = IndexableDocument::new(*doc_id, *body);
            if let Some((_, pairs)) = FIXTURE_METADATA.iter().find(|(id, _)| id == doc_id) {
                for (key, value) in *pairs {
                    document
                        .metadata
                        .insert((*key).to_owned(), (*value).to_owned());
                }
            }
            document
        })
        .collect()
}

/// The provenance a repaired campaign report carries. Only the SHAPE matters
/// here: `from_campaign_report` requires provenance to be present, and the
/// witness's own gates already prove each provenance axis separately.
fn baseline_campaign_provenance() -> frankensearch_quill_gauntlet::CampaignProvenance {
    frankensearch_quill_gauntlet::CampaignProvenance {
        producer_build_identity_sha256: "0".repeat(64),
        cargo_lock_sha256: "1".repeat(64),
        rustc_version_verbose: "rustc 0.0.0 (authorization baseline)".to_owned(),
        rust_toolchain_channel: "nightly-0000-00-00".to_owned(),
        unicode_version: "0.0.0".to_owned(),
        unicode_normalization_version: "0.0.0".to_owned(),
        unicode_normalization_table_version: "0.0.0".to_owned(),
        query_generator_id: "authorization-baseline".to_owned(),
        query_generator_schema_version: 1,
        query_seed: 0,
        query_source_identity_sha256: "2".repeat(64),
        query_profile_sha256: "3".repeat(64),
        analyzer_contract_hash: "4".repeat(64),
        schema_contract_hash: "5".repeat(64),
        corpus_manifest_hash: "6".repeat(64),
        query_manifest_hash: "7".repeat(64),
        corpus_seed: None,
    }
}

/// The complete frozen replacement campaign matrix.
///
/// bd-8nqz.4 slice 7: the CASS coverage slot takes CELLS and validates them
/// against the frozen completeness policy, so this builds the real matrix
/// rather than declaring a coverage class.
fn frozen_cass_cells() -> Vec<frankensearch_quill_gauntlet::CampaignCellEvidenceV1> {
    use frankensearch_quill_gauntlet::{
        BuiltInEvidenceBindingV1, CampaignCellEvidenceV1, CampaignContractModeV1,
        CampaignEvidenceRole, CampaignProfileV1, CampaignSha256V1, frozen_replacement_cell_keys,
        frozen_replacement_seed_bundle,
    };
    frozen_replacement_cell_keys()
        .into_iter()
        .map(|key| {
            let mode = match key.campaign_profile() {
                CampaignProfileV1::ShippingDefaultCoreV3 => CampaignContractModeV1::CoreLexicalV3,
                CampaignProfileV1::CassTotalV1 => CampaignContractModeV1::CassTotalV1,
            };
            let seeds = frozen_replacement_seed_bundle(key.seed_slot());
            let role = CampaignEvidenceRole::BuiltInEvidence(BuiltInEvidenceBindingV1::new(
                CampaignSha256V1::parse(&"a".repeat(64)).expect("strict lower-case hex"),
                CampaignSha256V1::parse(&"b".repeat(64)).expect("strict lower-case hex"),
            ));
            CampaignCellEvidenceV1::new(key, role, mode, seeds)
        })
        .collect()
}

/// Every committed expectation must hold against the REAL native Quill
/// paginated API — exact count, offset pagination, ordering and all.
#[test]
fn the_committed_expectations_hold_against_real_quill() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().expect("witness tempdir");
        let index = build_quill(&cx, dir.path()).await;

        let mut failures = Vec::new();
        for expectation in FIXTURE_EXPECTATIONS {
            let observed =
                observe_quill(&cx, &index, expectation).expect("observe native Quill page");
            let verdict = adjudicate(expectation, &observed);
            if !verdict.passed() {
                failures.push(format!(
                    "query={:?} limit={} offset={} -> {:?} (observed page {:?}, total {})",
                    expectation.query,
                    expectation.limit,
                    expectation.offset,
                    verdict.oracle_failures,
                    observed.page_doc_ids,
                    observed.total,
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "the committed expectations do not describe the shipping Quill engine:\n{}",
            failures.join("\n")
        );
    });
}

/// Every committed ENRICHMENT expectation must hold against real Quill:
/// configured highlight tags, hand-derived query classification, and metadata
/// semantics.
#[test]
fn the_committed_enrichment_expectations_hold_against_real_quill() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().expect("witness tempdir");
        let index = build_quill(&cx, dir.path()).await;

        let mut failures = Vec::new();
        for expectation in FIXTURE_ENRICHMENT_EXPECTATIONS {
            let observed = observe_quill_enrichment(&cx, &index, expectation)
                .expect("observe native Quill enrichment");
            let verdict = adjudicate_enrichment(expectation, &observed);
            if !verdict.passed() {
                failures.push(format!(
                    "query={:?} tags={}{} -> {:?} (hits {:?})",
                    expectation.query,
                    expectation.highlight_prefix,
                    expectation.highlight_postfix,
                    verdict.oracle_failures,
                    observed.hits,
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "the committed enrichment expectations do not describe the shipping Quill engine:\n{}",
            failures.join("\n")
        );
    });
}

/// bd-8nqz.4.1: the UTF-8 window boundary and deterministic long-query
/// truncation dimensions, against real Quill.
#[test]
fn utf8_windows_and_long_query_truncation_hold_against_real_quill() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().expect("witness tempdir");
        let index = build_quill(&cx, dir.path()).await;

        // UTF-8: the snippet window must consist of whole scalar values.
        let utf8_row = FIXTURE_ENRICHMENT_EXPECTATIONS
            .iter()
            .find(|row| row.subject_doc == UTF8_DOC_ID)
            .expect("utf8 enrichment row");
        let observed = observe_quill_enrichment(&cx, &index, utf8_row).expect("observe utf8");
        let verdict = adjudicate_utf8_window(&observed, UTF8_DOC_ID, UTF8_INTACT_TOKEN);
        assert!(
            verdict.passed(),
            "UTF-8 window boundary violated: {:?} (hits {:?})",
            verdict.oracle_failures,
            observed.hits
        );

        // Truncation: an over-length query must behave exactly like its
        // first-MAX_QUERY_LENGTH-character prefix.
        let (long, prefix, excluded_doc) = truncation_probe_queries();
        let row = |query: &str| EnrichedExpectationV1 {
            query: Box::leak(query.to_owned().into_boxed_str()),
            limit: 10,
            offset: 0,
            matching_docs: &[],
            total: 0,
            unambiguous_top: None,
        };
        let long_row = row(&long);
        let prefix_row = row(&prefix);
        let long_observed = observe_quill(&cx, &index, &long_row).expect("observe long query");
        let prefix_observed = observe_quill(&cx, &index, &prefix_row).expect("observe prefix");
        let verdict = adjudicate_truncation_determinism(&long_observed, &prefix_observed);
        assert!(
            verdict.passed(),
            "long-query truncation is not deterministic: {:?}",
            verdict.oracle_failures
        );
        // Both directions of the boundary, so this is a CUT-POINT test and
        // not merely a determinism test.
        assert!(
            prefix_observed
                .page_doc_ids
                .iter()
                .any(|id| id == "doc-beta"),
            "the term just INSIDE the cap must survive truncation; got {:?}",
            prefix_observed.page_doc_ids
        );
        assert!(
            !long_observed
                .page_doc_ids
                .iter()
                .any(|id| id == excluded_doc),
            "the term just BEYOND the cap must be truncated away, but {excluded_doc} matched: {:?}",
            long_observed.page_doc_ids
        );
    });
}

/// bd-8nqz.4.1 slice 4: the typed `QueryCapability` refusal on a real
/// positionless STORED schema, with its positionful control.
///
/// Both arms are real `QuillIndex` instances over the same committed corpus,
/// differing only in whether their stored schema indexes positions. The
/// refusal is therefore attributable to the missing capability rather than to
/// the query, the corpus, or the analyzer — and the served rows prove the
/// positionless index is otherwise usable, so a blanket refusal cannot pass.
#[test]
fn the_typed_capability_refusal_and_its_positionful_control_hold_against_real_quill() {
    use frankensearch_quill_gauntlet::native_enriched_witness::{
        CapabilitySchemaArmV1, FIXTURE_CAPABILITY_EXPECTATIONS, adjudicate_capability_probe,
        probe_quill_capability,
    };

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        // One index per arm, built from the arm's own stored schema.
        let mut arms = Vec::new();
        for arm in [
            CapabilitySchemaArmV1::Positionless,
            CapabilitySchemaArmV1::Positioned,
        ] {
            let index = QuillIndex::in_memory_with_schema(
                arm.schema(),
                QuillConfig {
                    deterministic_ingest: true,
                    max_ingest_shards: 1,
                    ..QuillConfig::default()
                },
            )
            .expect("build the witness capability index for this schema arm");
            index
                .index_documents(&cx, &fixture_documents())
                .await
                .expect("index the witness corpus into this schema arm");
            index
                .commit(&cx)
                .await
                .expect("commit the witness corpus in this schema arm");
            arms.push((arm, index));
        }

        let mut failures = Vec::new();
        for expectation in FIXTURE_CAPABILITY_EXPECTATIONS {
            let (_, index) = arms
                .iter()
                .find(|(arm, _)| *arm == expectation.arm)
                .expect("both schema arms are built");
            let outcome = probe_quill_capability(&cx, index, expectation.query, 10);
            let verdict = adjudicate_capability_probe(expectation, &outcome);
            if !verdict.passed() {
                failures.push(format!(
                    "row {} (query {:?}, {} schema) -> {:?} (observed {outcome:?})",
                    expectation.label,
                    expectation.query,
                    expectation.arm.code(),
                    verdict.oracle_failures,
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "the committed capability expectations do not describe the shipping Quill engine:\n{}",
            failures.join("\n")
        );
    });
}

/// bd-8nqz.4.1 slice 8: assemble a receipt from a run that really drove BOTH
/// engines, then verify it replays.
///
/// Every earlier live test adjudicates observations as they are produced. This
/// one closes the loop: the observations become a receipt, the receipt is
/// serialized, and the serialized bytes are loaded back through
/// `load_canonical`, which re-derives the manifests, the engine identities, the
/// producer identity and every verdict. A receipt that could not survive its
/// own round trip would be a record of nothing.
#[cfg(feature = "tantivy-oracle")]
#[test]
fn a_both_engines_receipt_assembles_and_replays_against_real_engines() {
    use frankensearch_core::traits::LexicalWrite;
    use frankensearch_lexical::TantivyIndex;
    use frankensearch_quill_gauntlet::native_enriched_witness::{
        CapabilitySchemaArmV1, NativeEngineV1, NativeEnrichedReceiptV1, NativeEnrichedRunV1,
        observe_quill_capabilities, observe_quill_enrichments, observe_quill_pages,
        observe_tantivy_enrichments, observe_tantivy_pages,
    };

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().expect("witness tempdir");
        let quill = build_quill(&cx, dir.path()).await;

        let tantivy_dir = tempfile::tempdir().expect("witness tempdir");
        let tantivy =
            TantivyIndex::create(tantivy_dir.path()).expect("create the witness Tantivy index");
        for document in fixture_documents() {
            tantivy
                .index_document(&cx, &document)
                .await
                .expect("index a witness fixture document");
        }
        tantivy.commit(&cx).await.expect("commit the Tantivy index");

        // The capability table spans two schema arms and is Quill-only.
        let mut capability_arms = Vec::new();
        for arm in [
            CapabilitySchemaArmV1::Positionless,
            CapabilitySchemaArmV1::Positioned,
        ] {
            let index = QuillIndex::in_memory_with_schema(
                arm.schema(),
                QuillConfig {
                    deterministic_ingest: true,
                    max_ingest_shards: 1,
                    ..QuillConfig::default()
                },
            )
            .expect("build the witness capability index for this schema arm");
            index
                .index_documents(&cx, &fixture_documents())
                .await
                .expect("index the witness corpus into this schema arm");
            index
                .commit(&cx)
                .await
                .expect("commit the witness corpus in this schema arm");
            capability_arms.push(index);
        }

        let mut observations = observe_quill_pages(&cx, &quill).expect("Quill pages");
        observations.extend(observe_tantivy_pages(&cx, &tantivy).expect("Tantivy pages"));
        let mut enriched_observations =
            observe_quill_enrichments(&cx, &quill).expect("Quill enrichments");
        enriched_observations
            .extend(observe_tantivy_enrichments(&cx, &tantivy).expect("Tantivy enrichments"));

        let run = NativeEnrichedRunV1 {
            observations,
            enriched_observations,
            capability_outcomes: observe_quill_capabilities(
                &cx,
                &capability_arms[0],
                &capability_arms[1],
            ),
            both_engines_observed: true,
        };
        let receipt =
            NativeEnrichedReceiptV1::assemble_for_this_build(&run).expect("assemble the receipt");

        // The receipt records what BOTH shipping engines actually did, and
        // every verdict was derived from the committed tables, not supplied.
        assert!(receipt.both_engines_observed);
        assert_eq!(receipt.engine_identities.len(), 2);
        // THE GUARD THIS BEAD EXISTS FOR: `engine_identities.len() == 2` is
        // derived from the FEATURE set, not from what was actually driven, and
        // the row counts are satisfied by any two runs. Driving Quill twice —
        // the failure mode the facade `lexical` alias makes easy since
        // bd-8nqz.4.2 flipped it to Quill — would satisfy both and report
        // itself as cross-engine coverage. Only these assertions see it.
        for engine in [NativeEngineV1::Quill, NativeEngineV1::Tantivy] {
            assert!(
                receipt
                    .observations
                    .iter()
                    .any(|observation| observation.engine == engine),
                "the receipt claims both engines but carries no {engine:?} page observation"
            );
            assert!(
                receipt
                    .enriched_observations
                    .iter()
                    .any(|observation| observation.engine == engine),
                "the receipt claims both engines but carries no {engine:?} enriched observation"
            );
        }
        assert!(
            receipt.all_verdicts_passed(),
            "both shipping engines must satisfy every committed expectation: {:?}",
            receipt
                .verdicts
                .iter()
                .filter(|verdict| !verdict.passed())
                .collect::<Vec<_>>()
        );

        // Round trip through the canonical loader.
        let address = receipt.receipt_hash().expect("address");
        let bytes = serde_json::to_vec(&receipt).expect("canonical body");
        let verified = NativeEnrichedReceiptV1::load_canonical(&bytes, &address)
            .expect("a receipt this build produced must replay in this build");
        assert_eq!(verified.receipt(), &receipt);
        assert!(!verified.authorizes_replacement());

        // Admissibility is a SEPARATE question, and in a shared working tree
        // the honest answer is no. Asserting the reason rather than the
        // outcome keeps this from silently becoming a green "it passed".
        //
        // The verdict is EMITTED either way (visible under --nocapture), because
        // a clean-tree run whose whole purpose is to produce an admissible
        // receipt must leave the address behind. Proving admissibility and
        // recording nothing would make the run unharvestable, and re-running it
        // later on a different tree would answer a different question.
        let admissibility = verified.require_release_admissible();
        println!("NATIVE_ENRICHED_RECEIPT_ADDRESS={address}");
        println!(
            "NATIVE_ENRICHED_RECEIPT_PRODUCER_REVISION={}",
            receipt.producer.source_git_revision
        );
        println!(
            "NATIVE_ENRICHED_RECEIPT_PRODUCER_DIRTY={}",
            receipt.producer.source_git_dirty
        );
        println!(
            "NATIVE_ENRICHED_RECEIPT_SOURCE_VERIFICATION={:?}",
            receipt.producer.source_verification
        );
        match &admissibility {
            Ok(()) => println!("NATIVE_ENRICHED_RECEIPT_ADMISSIBLE=yes"),
            Err(error) => println!("NATIVE_ENRICHED_RECEIPT_ADMISSIBLE=no reason={error}"),
        }

        match admissibility {
            Ok(()) => assert!(
                !receipt.producer.source_git_dirty,
                "a dirty producer must never be admissible"
            ),
            Err(error) => assert!(
                error.to_string().contains("dirty"),
                "the only expected inadmissibility here is a dirty tree, got {error}"
            ),
        }
    });
}

/// The same committed expectations must hold against the REAL Tantivy
/// incumbent. Running both arms against ONE independent oracle is what makes
/// a common-mode defect visible: neither engine is the other's reference.
#[cfg(feature = "tantivy-oracle")]
#[test]
fn the_committed_expectations_hold_against_real_tantivy() {
    use frankensearch_core::traits::LexicalWrite;
    use frankensearch_lexical::TantivyIndex;
    use frankensearch_quill_gauntlet::native_enriched_witness::{
        observe_tantivy, observe_tantivy_enrichment,
    };

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().expect("witness tempdir");
        let index = TantivyIndex::create(dir.path()).expect("create the witness Tantivy index");
        for document in fixture_documents() {
            index
                .index_document(&cx, &document)
                .await
                .expect("index a witness fixture document");
        }
        index.commit(&cx).await.expect("commit the Tantivy index");

        // The ENRICHED arm runs against the same index and the same committed
        // expectations, so a divergence between the engines shows up as one
        // of them failing the SHARED oracle rather than as a diff.
        let mut enrichment_failures = Vec::new();
        for expectation in FIXTURE_ENRICHMENT_EXPECTATIONS {
            let observed = observe_tantivy_enrichment(&cx, &index, expectation)
                .expect("observe native Tantivy enrichment");
            let verdict = adjudicate_enrichment(expectation, &observed);
            if !verdict.passed() {
                enrichment_failures.push(format!(
                    "query={:?} tags={}{} -> {:?} (hits {:?})",
                    expectation.query,
                    expectation.highlight_prefix,
                    expectation.highlight_postfix,
                    verdict.oracle_failures,
                    observed.hits,
                ));
            }
        }
        assert!(
            enrichment_failures.is_empty(),
            "the committed enrichment expectations do not describe the shipping Tantivy \
             engine:\n{}",
            enrichment_failures.join("\n")
        );

        // UTF-8 window boundaries, same oracle as the Quill arm.
        let utf8_row = FIXTURE_ENRICHMENT_EXPECTATIONS
            .iter()
            .find(|row| row.subject_doc == UTF8_DOC_ID)
            .expect("utf8 enrichment row");
        let utf8_observed =
            observe_tantivy_enrichment(&cx, &index, utf8_row).expect("observe utf8");
        let utf8_verdict = adjudicate_utf8_window(&utf8_observed, UTF8_DOC_ID, UTF8_INTACT_TOKEN);
        assert!(
            utf8_verdict.passed(),
            "Tantivy UTF-8 window boundary violated: {:?} (hits {:?})",
            utf8_verdict.oracle_failures,
            utf8_observed.hits
        );

        // Deterministic long-query truncation, with the cut point observable
        // in both directions.
        let (long, prefix, excluded_doc) = truncation_probe_queries();
        let row = |query: &str| EnrichedExpectationV1 {
            query: Box::leak(query.to_owned().into_boxed_str()),
            limit: 10,
            offset: 0,
            matching_docs: &[],
            total: 0,
            unambiguous_top: None,
        };
        let long_observed = observe_tantivy(&cx, &index, &row(&long)).expect("observe long query");
        let prefix_observed = observe_tantivy(&cx, &index, &row(&prefix)).expect("observe prefix");
        let truncation_verdict =
            adjudicate_truncation_determinism(&long_observed, &prefix_observed);
        assert!(
            truncation_verdict.passed(),
            "Tantivy long-query truncation is not deterministic: {:?}",
            truncation_verdict.oracle_failures
        );
        assert!(
            prefix_observed
                .page_doc_ids
                .iter()
                .any(|id| id == "doc-beta"),
            "the term just INSIDE the cap must survive truncation; got {:?}",
            prefix_observed.page_doc_ids
        );
        assert!(
            !long_observed
                .page_doc_ids
                .iter()
                .any(|id| id == excluded_doc),
            "the term just BEYOND the cap must be truncated away, but {excluded_doc} matched: {:?}",
            long_observed.page_doc_ids
        );

        let mut failures = Vec::new();
        for expectation in FIXTURE_EXPECTATIONS {
            let observed =
                observe_tantivy(&cx, &index, expectation).expect("observe native Tantivy page");
            let verdict = adjudicate(expectation, &observed);
            if !verdict.passed() {
                failures.push(format!(
                    "query={:?} limit={} offset={} -> {:?} (observed page {:?}, total {})",
                    expectation.query,
                    expectation.limit,
                    expectation.offset,
                    verdict.oracle_failures,
                    observed.page_doc_ids,
                    observed.total,
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "the committed expectations do not describe the shipping Tantivy engine:\n{}",
            failures.join("\n")
        );
    });
}

/// bd-8nqz.4 slice 2: an INADMISSIBLE enriched receipt can never authorize a
/// replacement, and the refusal must come from admissibility rather than from
/// a hole elsewhere in the bundle.
///
/// This lives here rather than beside the validator because a REAL
/// `VerifiedNativeEnrichedReceiptV1` cannot be minted from synthetic parts --
/// it has a private field and no public constructor, so the only way to hold
/// one is to assemble a real run and load it canonically. A unit test beside
/// the validator could only have used a stand-in, which would prove nothing
/// about the object the flip actually rests on.
///
/// The assertion is by REASON and is DETERMINED, not either/or: the expected
/// refusal is selected by an observable property of the receipt's own
/// producer, so exactly one outcome is pinned in any given environment.
#[test]
fn an_inadmissible_enriched_receipt_can_never_authorize_a_replacement() {
    use frankensearch_quill_gauntlet::native_enriched_witness::{
        CapabilitySchemaArmV1, NativeEnrichedReceiptV1, NativeEnrichedRunV1,
        observe_quill_capabilities, observe_quill_enrichments, observe_quill_pages,
    };
    use frankensearch_quill_gauntlet::replacement_authorization::{
        ReplacementEvidenceBundleV1, authorize,
    };
    use frankensearch_quill_gauntlet::{
        CampaignLexicalCoverageSummary, LexicalSideCoverageCounts, SemanticContract,
        load_pinned_campaign_report_v8, observe_live_quill_cancellation_receipt,
    };

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().expect("witness tempdir");
        let quill = build_quill(&cx, dir.path()).await;

        let mut capability_arms = Vec::new();
        for arm in [
            CapabilitySchemaArmV1::Positionless,
            CapabilitySchemaArmV1::Positioned,
        ] {
            let index = QuillIndex::in_memory_with_schema(
                arm.schema(),
                QuillConfig {
                    deterministic_ingest: true,
                    max_ingest_shards: 1,
                    ..QuillConfig::default()
                },
            )
            .expect("build the capability index for this schema arm");
            index
                .index_documents(&cx, &fixture_documents())
                .await
                .expect("index the witness corpus into this schema arm");
            index
                .commit(&cx)
                .await
                .expect("commit the witness corpus in this schema arm");
            capability_arms.push(index);
        }

        // A Quill-only run: exactly what the default feature lane can observe,
        // and therefore not release evidence no matter how green it is.
        let run = NativeEnrichedRunV1 {
            observations: observe_quill_pages(&cx, &quill).expect("Quill pages"),
            enriched_observations: observe_quill_enrichments(&cx, &quill)
                .expect("Quill enrichments"),
            capability_outcomes: observe_quill_capabilities(
                &cx,
                &capability_arms[0],
                &capability_arms[1],
            ),
            both_engines_observed: false,
        };
        let receipt =
            NativeEnrichedReceiptV1::assemble_for_this_build(&run).expect("assemble the receipt");
        let address = receipt.receipt_hash().expect("address");
        let bytes = serde_json::to_vec(&receipt).expect("canonical body");
        let verified = NativeEnrichedReceiptV1::load_canonical(&bytes, &address)
            .expect("a receipt this build produced must load");

        // EVERY OTHER SLOT IS PRESENT AND VALID, which is what makes this a
        // test of admissibility rather than of slot presence.
        let candidate = receipt.producer.source_git_revision.clone();
        // The core slot takes the REPORT and derives its binding, so the
        // coverage class cannot be asserted by this test either. The pinned
        // fixture is repaired along exactly the axes from_campaign_report
        // gates, and its producer revision is overridden to the candidate this
        // run actually observed.
        let mut core = load_pinned_campaign_report_v8().expect("pinned V8 campaign report");
        core.lexical_coverage = CampaignLexicalCoverageSummary::CoreLexicalV3 {
            subject: Box::new(LexicalSideCoverageCounts::default()),
            oracle: Box::new(LexicalSideCoverageCounts::default()),
            admissible: true,
        };
        core.semantic_contract = SemanticContract::shipping_default();
        core.provenance = Some(baseline_campaign_provenance());
        core.producer_build_identity.source_git_dirty = false;
        core.producer_build_identity.source_git_revision = candidate.clone();
        let cass = frozen_cass_cells();
        let cancellation = observe_live_quill_cancellation_receipt(&cx)
            .await
            .expect("observe the live Quill cancellation matrix");
        let census: frankensearch_quill_gauntlet::DivergenceRegisterLedger =
            serde_json::from_str(include_str!("../fixtures/divergence-register-v2-live.json"))
                .expect("the registered divergence census parses");

        let bundle = ReplacementEvidenceBundleV1 {
            candidate_source_revision: &candidate,
            core_lexical_v3: Some(&core),
            cass_total: Some(&cass),
            native_enriched: Some(&verified),
            cancellation: Some(&cancellation),
            divergence_census: Some(&census),
        };

        let error = authorize(&bundle)
            .expect_err("a single-engine receipt must never authorize a replacement")
            .to_string();

        if receipt.producer.source_git_dirty {
            assert!(
                error.contains("dirty"),
                "a dirty producer must refuse for provenance, got: {error}"
            );
        } else {
            assert!(
                error.contains("single-engine"),
                "a clean single-engine receipt must refuse for coverage, got: {error}"
            );
        }

        // In BOTH environments the refusal must be an ADMISSIBILITY refusal.
        // Without this the test would pass in a tree where some unrelated slot
        // had quietly become unsatisfiable -- the exact masking defect the
        // validator's own check ordering was repaired for.
        assert!(
            !error.contains("missing required evidence"),
            "the refusal must come from admissibility, not a missing slot: {error}"
        );
    });
}

/// bd-8nqz.4 slice 2 (happy path): a complete, current, admissible bundle
/// DOES authorize -- and the grant is emitted so a clean run is harvestable.
///
/// A validator proved only by its refusals is indistinguishable from one that
/// refuses everything. This is the positive control, and it is deliberately
/// the same shape as bd-8nqz.4.1's emitter: the outcome is asserted by REASON,
/// so the honest answer in a dirty tree ("inadmissible because dirty") is
/// recorded as such rather than being allowed to stand in for a clean grant.
#[cfg(feature = "tantivy-oracle")]
#[test]
fn a_complete_admissible_bundle_authorizes_and_emits_its_grant() {
    use frankensearch_core::traits::LexicalWrite;
    use frankensearch_lexical::TantivyIndex;
    use frankensearch_quill_gauntlet::native_enriched_witness::{
        CapabilitySchemaArmV1, NativeEnrichedReceiptV1, NativeEnrichedRunV1,
        observe_quill_capabilities, observe_quill_enrichments, observe_quill_pages,
        observe_tantivy_enrichments, observe_tantivy_pages,
    };
    use frankensearch_quill_gauntlet::replacement_authorization::{
        ReplacementEvidenceBundleV1, authorize,
    };
    use frankensearch_quill_gauntlet::{
        CampaignLexicalCoverageSummary, LexicalSideCoverageCounts, SemanticContract,
        load_pinned_campaign_report_v8, observe_live_quill_cancellation_receipt,
    };

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().expect("witness tempdir");
        let quill = build_quill(&cx, dir.path()).await;

        let tantivy_dir = tempfile::tempdir().expect("witness tempdir");
        let tantivy =
            TantivyIndex::create(tantivy_dir.path()).expect("create the witness Tantivy index");
        for document in fixture_documents() {
            tantivy
                .index_document(&cx, &document)
                .await
                .expect("index a witness fixture document");
        }
        tantivy.commit(&cx).await.expect("commit the Tantivy index");

        let mut capability_arms = Vec::new();
        for arm in [
            CapabilitySchemaArmV1::Positionless,
            CapabilitySchemaArmV1::Positioned,
        ] {
            let index = QuillIndex::in_memory_with_schema(
                arm.schema(),
                QuillConfig {
                    deterministic_ingest: true,
                    max_ingest_shards: 1,
                    ..QuillConfig::default()
                },
            )
            .expect("build the capability index for this schema arm");
            index
                .index_documents(&cx, &fixture_documents())
                .await
                .expect("index the witness corpus into this schema arm");
            index
                .commit(&cx)
                .await
                .expect("commit the witness corpus in this schema arm");
            capability_arms.push(index);
        }

        let mut observations = observe_quill_pages(&cx, &quill).expect("Quill pages");
        observations.extend(observe_tantivy_pages(&cx, &tantivy).expect("Tantivy pages"));
        let mut enriched_observations =
            observe_quill_enrichments(&cx, &quill).expect("Quill enrichments");
        enriched_observations
            .extend(observe_tantivy_enrichments(&cx, &tantivy).expect("Tantivy enrichments"));

        let run = NativeEnrichedRunV1 {
            observations,
            enriched_observations,
            capability_outcomes: observe_quill_capabilities(
                &cx,
                &capability_arms[0],
                &capability_arms[1],
            ),
            both_engines_observed: true,
        };
        let receipt =
            NativeEnrichedReceiptV1::assemble_for_this_build(&run).expect("assemble the receipt");
        let address = receipt.receipt_hash().expect("address");
        let bytes = serde_json::to_vec(&receipt).expect("canonical body");
        let verified = NativeEnrichedReceiptV1::load_canonical(&bytes, &address)
            .expect("a receipt this build produced must load");

        let candidate = receipt.producer.source_git_revision.clone();
        // The core slot takes the REPORT and derives its binding, so the
        // coverage class cannot be asserted by this test either. The pinned
        // fixture is repaired along exactly the axes from_campaign_report
        // gates, and its producer revision is overridden to the candidate this
        // run actually observed.
        let mut core = load_pinned_campaign_report_v8().expect("pinned V8 campaign report");
        core.lexical_coverage = CampaignLexicalCoverageSummary::CoreLexicalV3 {
            subject: Box::new(LexicalSideCoverageCounts::default()),
            oracle: Box::new(LexicalSideCoverageCounts::default()),
            admissible: true,
        };
        core.semantic_contract = SemanticContract::shipping_default();
        core.provenance = Some(baseline_campaign_provenance());
        core.producer_build_identity.source_git_dirty = false;
        core.producer_build_identity.source_git_revision = candidate.clone();
        let cass = frozen_cass_cells();
        let cancellation = observe_live_quill_cancellation_receipt(&cx)
            .await
            .expect("observe the live Quill cancellation matrix");
        let census: frankensearch_quill_gauntlet::DivergenceRegisterLedger =
            serde_json::from_str(include_str!("../fixtures/divergence-register-v2-live.json"))
                .expect("the registered divergence census parses");

        let bundle = ReplacementEvidenceBundleV1 {
            candidate_source_revision: &candidate,
            core_lexical_v3: Some(&core),
            cass_total: Some(&cass),
            native_enriched: Some(&verified),
            cancellation: Some(&cancellation),
            divergence_census: Some(&census),
        };

        let granted = authorize(&bundle);
        println!("REPLACEMENT_AUTHORIZATION_CANDIDATE={candidate}");
        println!(
            "REPLACEMENT_AUTHORIZATION_PRODUCER_DIRTY={}",
            receipt.producer.source_git_dirty
        );
        match &granted {
            Ok(authorization) => {
                println!("REPLACEMENT_AUTHORIZATION_GRANTED=yes");
                println!(
                    "REPLACEMENT_AUTHORIZATION_ENRICHED_ADDRESS={}",
                    authorization.native_enriched_receipt_address
                );
                println!(
                    "REPLACEMENT_AUTHORIZATION_CANCELLATION_BODY_SHA256={}",
                    authorization.cancellation_body_sha256
                );
                println!(
                    "REPLACEMENT_AUTHORIZATION_ENGINE_REVISION={}",
                    authorization.cancellation_engine_revision
                );
            }
            Err(error) => println!("REPLACEMENT_AUTHORIZATION_GRANTED=no reason={error}"),
        }

        // Asserted by REASON, and DETERMINED by the receipt's own producer
        // state so exactly one outcome is pinned in any environment.
        match granted {
            Ok(authorization) => {
                assert!(
                    !receipt.producer.source_git_dirty,
                    "a dirty producer must never yield a grant"
                );
                assert!(authorization.authorizes_replacement());
                assert_eq!(authorization.candidate_source_revision, candidate);
                assert_eq!(authorization.native_enriched_receipt_address, address);
            }
            Err(error) => assert!(
                error.to_string().contains("dirty"),
                "the only expected refusal for a complete both-engines bundle is a dirty \
                 producer, got: {error}"
            ),
        }
    });
}
