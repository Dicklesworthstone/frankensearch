use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StartupPolicy {
    pub require_attestation: bool,
    pub require_signature: bool,
    pub on_attestation_missing: String,
    pub on_signature_missing: String,
    pub on_signature_invalid: String,
    pub on_hash_mismatch: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProvenanceContractDefinition {
    pub kind: String, // "fsfs_provenance_contract"
    pub schema_version: u32,
    pub required_attestation_fields: Vec<String>,
    pub startup_policy: StartupPolicy,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BuildProvenance {
    pub source_commit: String,
    pub build_profile: String,
    pub rustc_version: String,
    pub target_triple: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeProvenance {
    pub binary_hash_sha256: String,
    pub config_hash_sha256: String,
    pub index_manifest_hash_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArtifactHash {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Signature {
    pub algorithm: String,
    pub key_id: String,
    pub signature_b64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProvenanceAttestationManifest {
    pub kind: String, // "fsfs_provenance_attestation"
    pub schema_version: u32,
    pub attestation_id: String,
    pub generated_at: String,
    pub build: BuildProvenance,
    pub runtime: RuntimeProvenance,
    pub artifact_hashes: Vec<ArtifactHash>,
    pub signature: Option<Signature>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StartupChecks {
    pub attestation_present: bool,
    pub attestation_parsed: bool,
    pub signature_present: bool,
    pub signature_valid: bool,
    pub binary_hash_match: bool,
    pub config_hash_match: bool,
    pub index_manifest_hash_match: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProvenanceAlert {
    pub reason_code: String,
    pub severity: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProvenanceStartupCheck {
    pub kind: String, // "fsfs_provenance_startup_check"
    pub schema_version: u32,
    pub trace_id: String,
    pub attestation_id: String,
    pub status: String,
    pub action: String,
    pub checks: StartupChecks,
    pub alerts: Vec<ProvenanceAlert>,
}
