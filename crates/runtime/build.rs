fn main() {
    // Keep bottom-up allocations (heaps, stacks) in the low 4 GiB so the PS4 guest
    // address range (16 GiB .. 1 TiB) can be reserved at startup. Image ASLR stays on.
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bins=/HIGHENTROPYVA:NO");
    } else {
        println!("cargo:rustc-link-arg-bins=-Wl,--disable-high-entropy-va");
    }
}
