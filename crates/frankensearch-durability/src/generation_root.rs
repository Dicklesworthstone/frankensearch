//! Pure route validation for capability-based generation roots.
//!
//! This module deliberately separates route syntax from filesystem admission.
//! A later descriptor-relative traversal layer may only receive a
//! [`GenerationRootRouteV1`] produced here, so it never needs to reinterpret
//! ambient, absolute, or traversal-bearing paths.

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::path::{Component, Path};

/// Bounded reason why a generation-root route cannot enter capability-based
/// filesystem traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationRootRouteErrorV1 {
    /// The route has no component.
    Empty,
    /// The route contains a separator, dot component, parent component, or
    /// platform prefix that would make its interpretation non-canonical.
    UnsafeComponent,
    /// The route contains an embedded NUL byte.
    NulByte,
}

impl fmt::Display for GenerationRootRouteErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Empty => "generation-root route is empty",
            Self::UnsafeComponent => "generation-root route has an unsafe component",
            Self::NulByte => "generation-root route contains a NUL byte",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for GenerationRootRouteErrorV1 {}

/// One canonical relative route to be resolved from an already trusted
/// directory descriptor.
///
/// The type owns only normal path components. It does not touch the
/// filesystem, follow symlinks, consult the current directory, or normalize a
/// caller-controlled route. Those operations are deferred to the
/// descriptor-relative admission layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationRootRouteV1 {
    components: Vec<OsString>,
}

impl GenerationRootRouteV1 {
    /// Parse one canonical, non-empty relative route.
    ///
    /// # Errors
    ///
    /// Returns a bounded error for every ambiguous route shape before any
    /// filesystem operation can occur.
    pub fn parse(route: &Path) -> Result<Self, GenerationRootRouteErrorV1> {
        #[cfg(unix)]
        validate_unix_route_bytes(route.as_os_str())?;

        let mut components = Vec::new();
        for component in route.components() {
            match component {
                Component::Normal(component) => components.push(component.to_owned()),
                Component::CurDir
                | Component::ParentDir
                | Component::RootDir
                | Component::Prefix(_) => return Err(GenerationRootRouteErrorV1::UnsafeComponent),
            }
        }
        if components.is_empty() {
            return Err(GenerationRootRouteErrorV1::Empty);
        }
        Ok(Self { components })
    }

    /// Construct a route containing exactly one normal component.
    ///
    /// # Errors
    ///
    /// Returns the same bounded validation error as [`Self::parse`].
    pub fn from_component(component: &OsStr) -> Result<Self, GenerationRootRouteErrorV1> {
        let route = Self::parse(Path::new(component))?;
        if route.components.len() != 1 {
            return Err(GenerationRootRouteErrorV1::UnsafeComponent);
        }
        Ok(route)
    }

    /// Components to resolve in order from an explicit trusted descriptor.
    #[must_use]
    pub fn components(&self) -> &[OsString] {
        &self.components
    }
}

#[cfg(unix)]
fn validate_unix_route_bytes(route: &OsStr) -> Result<(), GenerationRootRouteErrorV1> {
    use std::os::unix::ffi::OsStrExt as _;

    let bytes = route.as_bytes();
    if bytes.is_empty() {
        return Err(GenerationRootRouteErrorV1::Empty);
    }
    if bytes.contains(&0) {
        return Err(GenerationRootRouteErrorV1::NulByte);
    }
    if bytes
        .split(|byte| *byte == b'/')
        .any(|component| component.is_empty() || matches!(component, b"." | b".."))
    {
        return Err(GenerationRootRouteErrorV1::UnsafeComponent);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{GenerationRootRouteErrorV1, GenerationRootRouteV1};
    use std::ffi::OsStr;
    use std::path::Path;

    #[test]
    fn route_preserves_normal_component_order_without_filesystem_access() {
        let route = GenerationRootRouteV1::parse(Path::new("generations/0007/AUTHORITY"))
            .expect("normal relative route");
        let names: Vec<_> = route
            .components()
            .iter()
            .map(|component| component.to_string_lossy())
            .collect();
        assert_eq!(names, ["generations", "0007", "AUTHORITY"]);
    }

    #[test]
    fn route_rejects_ambiguous_components_before_descriptor_open() {
        assert_eq!(
            GenerationRootRouteV1::parse(Path::new("")),
            Err(GenerationRootRouteErrorV1::Empty)
        );
        for route in [".", "..", "a/./b", "a/../b", "/absolute", "a//b", "a/"] {
            assert_eq!(
                GenerationRootRouteV1::parse(Path::new(route)),
                Err(GenerationRootRouteErrorV1::UnsafeComponent),
                "route {route:?} must never be normalized into an admitted capability route"
            );
        }
    }

    #[test]
    fn single_component_constructor_refuses_separators() {
        assert_eq!(
            GenerationRootRouteV1::from_component(OsStr::new("AUTHORITY/next")),
            Err(GenerationRootRouteErrorV1::UnsafeComponent)
        );
    }

    #[cfg(unix)]
    #[test]
    fn route_rejects_embedded_nul_without_lossy_conversion() {
        use std::os::unix::ffi::OsStrExt as _;

        let route = OsStr::from_bytes(b"AUTHORITY\0shadow");
        assert_eq!(
            GenerationRootRouteV1::from_component(route),
            Err(GenerationRootRouteErrorV1::NulByte)
        );
    }
}
