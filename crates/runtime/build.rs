fn main() {
    // Keep bottom-up allocations (heaps, stacks) in the low 4 GiB so the PS4 guest
    // address range (16 GiB .. 1 TiB) can be reserved at startup. Image ASLR stays on.
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bins=/HIGHENTROPYVA:NO");
    } else {
        println!("cargo:rustc-link-arg-bins=-Wl,--disable-high-entropy-va");
    }
    // LibAtrac9 (vendored, MIT): ATRAC9 decoding for NGS2 sampler voices.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../third_party/libatrac9/src");
    let mut build = cc::Build::new();
    for e in std::fs::read_dir(&dir).expect("third_party/libatrac9/src") {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "c") {
            println!("cargo:rerun-if-changed={}", p.display());
            build.file(&p);
        }
    }
    build.include(&dir).opt_level(2).warnings(false).compile("atrac9");
}
