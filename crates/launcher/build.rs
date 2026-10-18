// Embeds res/app.rc (manifest: Common Controls 6, per-monitor DPI awareness;
// icon) with the llvm-mingw windres (tools/env.ps1 puts it on PATH; WINDRES
// overrides). Without windres the program still builds, unstyled.
use std::path::Path;
use std::process::Command;

fn main() {
    let res = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../res");
    for f in ["app.rc", "app.manifest", "app.ico"] {
        println!("cargo:rerun-if-changed={}", res.join(f).display());
    }
    println!("cargo:rerun-if-env-changed=WINDRES");
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join("app_res.o");
    let windres = std::env::var("WINDRES").unwrap_or_else(|_| "x86_64-w64-mingw32-windres".into());
    match Command::new(&windres).current_dir(&res).arg("-i").arg("app.rc").arg("-o").arg(&out).status() {
        Ok(s) if s.success() => println!("cargo:rustc-link-arg-bins={}", out.display()),
        r => println!("cargo:warning=resources not embedded ({windres}: {r:?})"),
    }
}
