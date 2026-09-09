//! Library for the `md` project.

/// Returns the greeting printed by the binary.
///
/// Kept in the library so it can be unit-tested directly.
pub fn greeting() -> &'static str {
    "Hello, world!"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_is_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }
}
