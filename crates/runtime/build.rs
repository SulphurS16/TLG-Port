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
    embed_resources();
}

/// res/app.rc (manifest: Common Controls 6 for message boxes, per-monitor
/// DPI awareness; icon) through the llvm-mingw windres (tools/env.ps1 puts
/// it on PATH; WINDRES overrides). Without windres tlg still builds.
fn embed_resources() {
    use std::path::Path;
    let res = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../res");
    for f in ["app.rc", "app.manifest", "app.ico"] {
        println!("cargo:rerun-if-changed={}", res.join(f).display());
    }
    println!("cargo:rerun-if-env-changed=WINDRES");
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join("app_res.o");
    let windres = std::env::var("WINDRES").unwrap_or_else(|_| "x86_64-w64-mingw32-windres".into());
    match std::process::Command::new(&windres).current_dir(&res).arg("-i").arg("app.rc").arg("-o").arg(&out).status() {
        Ok(s) if s.success() => println!("cargo:rustc-link-arg-bins={}", out.display()),
        r => println!("cargo:warning=resources not embedded ({windres}: {r:?})"),
    }
}
