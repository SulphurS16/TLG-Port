# Third-party components

## Rust crates (fetched by Cargo, not included; versions pinned in `Cargo.lock`)

| Crate | Use | License |
| --- | --- | --- |
| windows-sys / windows | Win32 API bindings (memory, threads, WASAPI/COM, Media Foundation; the launcher's window, controls and folder picker) | MIT OR Apache-2.0 |
| iced-x86 | x86-64 decoding and encoding for load-time patching (TLS, SSE4a trampolines) | MIT |
| parking_lot, smallvec | synchronization primitives, inline vectors | MIT OR Apache-2.0 |
| ash | Vulkan bindings (loads `vulkan-1.dll` at run time; the runtime and the launcher's GPU list) | MIT OR Apache-2.0 |
| rspirv / spirv | SPIR-V module builder for the GCN shader translator | Apache-2.0 |
| miniz_oxide / adler2 | zlib for PNG screenshots | MIT OR Zlib OR Apache-2.0 / 0BSD OR MIT OR Apache-2.0 |
| cc (build dependency) | compiles the vendored LibAtrac9 C sources | MIT OR Apache-2.0 |

Their dependencies are MIT and/or Apache-2.0, except libloading (ISC; it loads `vulkan-1.dll`
for ash) and unicode-ident (also Unicode-3.0; used only while building).

## In this repository

| Component | Location | Use | License |
| --- | --- | --- | --- |
| LibAtrac9 (Alex Barney), C implementation | `third_party/libatrac9` (source, licence and origin with SHA-256 in its README), unmodified | ATRAC9 decoding for NGS2 sampler voices | MIT |
| vgmstream, the HE-VAG predictor table | `crates/runtime/src/hle/libs/hevag.rs` (the table of `src/coding/psv_decoder.c`, with the notice below) | HE-VAG decoding for NGS2 sampler voices | ISC-style (below) |
| License texts of crates that ship none | `third_party/licenses` | read by `tools/notices.py` | iced-x86: MIT (its repository's text); rspirv, spirv: Apache-2.0 (the license text) |

vgmstream's notice (its `COPYING`), for the HE-VAG predictor table:

    Copyright (c) 2008-2025 Adam Gashlin, Fastelbja, Ronny Elfert, bnnm,
                            Christopher Snowhill, NicknineTheEagle, bxaimc,
                            Thealexbarney, CyberBotX, EdnessP, et al

    Portions Copyright (c) 2004-2008, Marko Kreen
    Portions Copyright 2001-2007  jagarl / Kazunori Ueno <jagarl@creator.club.ne.jp>
    Portions Copyright (c) 1998, Justin Frankel/Nullsoft Inc.
    Portions Copyright (C) 2006 Nullsoft, Inc.
    Portions Copyright (c) 2005-2007 Paul Hsieh
    Portions Copyright (C) 2000-2004 Leshade Entis, Entis-soft.
    Portions Public Domain originating with Sun Microsystems

    Permission to use, copy, modify, and distribute this software for any
    purpose with or without fee is hereby granted, provided that the above
    copyright notice and this permission notice appear in all copies.

    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
    ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

## Build tools (downloaded by `tools/bootstrap.ps1`, not included)

| Tool | Version | License |
| --- | --- | --- |
| Rust (rustup, `x86_64-pc-windows-gnullvm`) | 1.99.0 | MIT OR Apache-2.0 |
| llvm-mingw (github.com/mstorsjo/llvm-mingw): LLVM clang, lld, libunwind, compiler-rt; mingw-w64 headers and runtime | 20260908, UCRT, x86_64 | LLVM: Apache-2.0 WITH LLVM-exception; mingw-w64: its own licenses (mostly public domain, ZPL 2.1 and MIT) |
| Python embeddable (for `tools/make_icon.py` and `tools/notices.py`) | 3.13.9 | PSF License |

## In the programs built from this repository

`tlg.exe` and `tlg-launcher.exe` contain the Rust standard library (MIT OR Apache-2.0), the
crates and components above, LLVM's libunwind and compiler-rt builtins (Apache-2.0 WITH
LLVM-exception) and the mingw-w64 runtime startup code. Their notices ship next to the
programs: `THIRD_PARTY_NOTICES.txt` (the crates, LibAtrac9, vgmstream, LLVM and the mingw-w64
runtime; made by `tools/notices.py`) and `RUST_STD_NOTICES.html` (the Rust standard library's
notices, from the toolchain).

## Used from Windows at run time (not redistributed)

Vulkan (`vulkan-1.dll`, installed with the graphics driver), WASAPI, Media Foundation, XInput,
HID, the common controls and the Universal C Runtime.

## Own artwork

The application icon (`res/app.ico`, a feather) is drawn by `tools/make_icon.py` (this
project, MIT); no game assets are used.

## The game

The Last Guardian is © Sony Interactive Entertainment Inc. The port contains none of its data
or code; each player provides their own copy.
