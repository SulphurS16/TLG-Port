# LibAtrac9 (vendored)

ATRAC9 decoder by Alex Barney, MIT license (see `LICENSE` in this directory).

- Source: `https://codeload.github.com/Thealexbarney/LibAtrac9/zip/refs/heads/master`,
  downloaded 2026-10-11, archive SHA-256
  `52758fa71b7fd94e735edee5ca5364028ad299ea7e98deee0c15061758e44f95`.
- Vendored: the `C/src` directory, unmodified. Built by `crates/runtime/build.rs` (cc crate)
  and used by the NGS2 sampler voices for ATRAC9 waveforms.
