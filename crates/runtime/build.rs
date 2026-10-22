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
    // PGO training builds (`--cfg tlg_pgo_gen` beside `-Cprofile-generate`, tools/pgo.ps1).
    println!("cargo:rustc-check-cfg=cfg(tlg_pgo_gen)");
    // Symbol names for log lines (data/ps4-names.txt) where the checkout has
    // them - not in release builds (TLG_RELEASE_BUILD): the public tree has no
    // such file, and a PGO profile trained in this checkout must fit the
    // release build's code (the cfg changes how the crate is split up and
    // inlined, and with it the profiled functions' shapes).
    println!("cargo:rustc-check-cfg=cfg(nid_names)");
    println!("cargo:rerun-if-env-changed=TLG_RELEASE_BUILD");
    let names = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/ps4-names.txt");
    println!("cargo:rerun-if-changed={}", names.display());
    if names.is_file() && std::env::var_os("TLG_RELEASE_BUILD").is_none() {
        println!("cargo:rustc-cfg=nid_names");
    }
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
