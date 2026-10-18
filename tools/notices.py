r"""Writes THIRD_PARTY_NOTICES.txt: the license notices of the third-party code
compiled into tlg.exe and tlg-launcher.exe (this project's own code: LICENSE).

    . .\tools\env.ps1
    python tools\notices.py

The crates are the ones linked into the two programs for the Windows target
(`cargo metadata` with Cargo.lock: normal dependencies, without proc-macros and
build-time crates), each with the license file it ships; where a crate offers a
choice, MIT is used. Then LibAtrac9's license, vgmstream's notice (from
hevag.rs) and the LLVM and MinGW-w64 runtime notices of the llvm-mingw toolchain.
tools\package.ps1 adds the Rust standard library's notices (the toolchain's
COPYRIGHT-library.html) next to the programs as RUST_STD_NOTICES.html.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'THIRD_PARTY_NOTICES.txt'
TARGET = 'x86_64-pc-windows-gnullvm'
PROGRAMS = ('tlg', 'tlg-launcher')
# Where a crate offers several licenses, the first of these it offers is used.
PREFERENCE = ('MIT', 'ISC', 'Zlib', '0BSD', 'Apache-2.0')
EXT = r'(\.(md|txt))?'
LICENSE_FILES = {
    'MIT': [rf'licen[cs]e[-_.]?mit{EXT}', rf'mit[-_.]licen[cs]e{EXT}'],
    'ISC': [rf'licen[cs]e[-_.]?isc{EXT}'],
    'Zlib': [rf'licen[cs]e[-_.]?zlib{EXT}'],
    '0BSD': [rf'licen[cs]e[-_.]?0bsd{EXT}'],
    'Apache-2.0': [rf'licen[cs]e[-_.]?apache(-2\.0)?{EXT}'],
}
# Accepted as well when the crate has a single license.
SINGLE_FILES = [rf'licen[cs]e{EXT}', rf'copying{EXT}']
NOTICE_FILE = rf'notice{EXT}'
# License texts of crates that ship none: iced's from its repository; rspirv and
# spirv are Apache-2.0 with no NOTICE file upstream, so the license text itself.
UPSTREAM_TEXTS = {
    'iced-x86': ('MIT', 'third_party/licenses/iced-x86.txt'),
    'rspirv': ('Apache-2.0', 'third_party/licenses/Apache-2.0.txt'),
    'spirv': ('Apache-2.0', 'third_party/licenses/Apache-2.0.txt'),
}
WIDTH = 78


def fail(msg):
    sys.exit(f'notices.py: {msg}')


def read(path):
    text = path.read_text(encoding='utf-8').lstrip('﻿').replace('\r\n', '\n')
    return text.strip('\n') + '\n'


def linked_crates():
    meta = json.loads(subprocess.run(
        ['cargo', 'metadata', '--format-version', '1', '--locked', '--filter-platform', TARGET,
         '--manifest-path', str(ROOT / 'Cargo.toml')],
        capture_output=True, check=True).stdout)
    pkgs = {p['id']: p for p in meta['packages']}
    nodes = {n['id']: n for n in meta['resolve']['nodes']}

    def proc_macro(p):
        return any('proc-macro' in t['kind'] for t in p['targets'])

    todo = [i for i, p in pkgs.items() if p['name'] in PROGRAMS and p['source'] is None]
    if len(todo) != len(PROGRAMS):
        fail('the programs are not in this workspace')
    seen = set()
    while todo:
        i = todo.pop()
        if i in seen:
            continue
        seen.add(i)
        for d in nodes[i]['deps']:
            if any(k['kind'] is None for k in d['dep_kinds']) and not proc_macro(pkgs[d['pkg']]):
                todo.append(d['pkg'])
    crates = [pkgs[i] for i in seen if pkgs[i]['source'] is not None]
    return sorted(crates, key=lambda p: (p['name'], p['version']))


def license_text(p):
    expr = (p.get('license') or '').strip()
    name = f"{p['name']} {p['version']}"
    if not expr:
        fail(f'{name}: no license field')
    if ' AND ' in expr or ' WITH ' in expr:
        fail(f"{name}: '{expr}' needs a decision by hand")
    offered = [o.strip(' ()') for o in re.split(r'\s+OR\s+|/', expr)]
    lic = next((x for x in PREFERENCE if x in offered), None)
    if lic is None:
        fail(f"{name}: none of {PREFERENCE} in '{expr}'")
    d = Path(p['manifest_path']).parent
    files = sorted(f.name for f in d.iterdir() if f.is_file())
    patterns = LICENSE_FILES[lic] + (SINGLE_FILES if len(offered) == 1 else [])
    found = [f for f in files if any(re.fullmatch(x, f, re.I) for x in patterns)]
    if found:
        text = read(d / found[0])
    elif UPSTREAM_TEXTS.get(p['name'], (None,))[0] == lic:
        text = read(ROOT / UPSTREAM_TEXTS[p['name']][1])
    else:
        fail(f'{name}: no {lic} license file among {files}')
    if lic == 'Apache-2.0':
        for f in files:
            if re.fullmatch(NOTICE_FILE, f, re.I):
                text += '\n' + read(d / f)
    return lic, text


def crates_section(crates):
    groups = {}
    for p in crates:
        lic, text = license_text(p)
        groups.setdefault((lic, text), []).append(p)
    parts = []
    for (lic, text), members in groups.items():
        names = '\n'.join(f"{p['name']} {p['version']} ({p['license']})" for p in members)
        head = names + f'\nLicense used: {lic}'
        parts.append(f"{'-' * WIDTH}\n{head}\n{'-' * WIDTH}\n\n{text}")
    return '\n'.join(parts)


def vgmstream_notice():
    lines = []
    for line in (ROOT / 'crates/runtime/src/hle/libs/hevag.rs').read_text(encoding='utf-8').splitlines():
        if not line.startswith('//!'):
            break
        lines.append(line[4:] if line.startswith('//! ') else line[3:])
    start = next((i for i, l in enumerate(lines) if l.startswith('Copyright')), None)
    end = next((i for i in range(start or 0, len(lines)) if lines[i].endswith('THIS SOFTWARE.')), None)
    if start is None or end is None:
        fail("vgmstream's notice not found in hevag.rs")
    return '\n'.join(lines[start:end + 1]) + '\n'


def llvm_mingw():
    clang = shutil.which('x86_64-w64-mingw32-clang')
    if not clang:
        fail('llvm-mingw is not on PATH (run tools\\env.ps1 first)')
    return Path(clang).resolve().parent.parent


INTRO = """\
Third-party notices: tlg.exe and tlg-launcher.exe
=================================================

The Last Guardian unofficial PC port is MIT licensed (LICENSE). Besides its own
code, the two programs contain the third-party code below, used under the
licenses shown. Where a component offers a choice of licenses, the one used is
named. The Rust standard library's notices are in RUST_STD_NOTICES.html, next
to the programs.

This file is generated by tools/notices.py from Cargo.lock and the components'
own license files.
"""


def main():
    tc = llvm_mingw()
    sections = [
        ('Rust crates', crates_section(linked_crates())),
        ('LibAtrac9 (third_party/libatrac9): ATRAC9 decoding', read(ROOT / 'third_party/libatrac9/LICENSE')),
        ("vgmstream's HE-VAG predictor table (crates/runtime/src/hle/libs/hevag.rs)", vgmstream_notice()),
        ('LLVM: libunwind and the compiler-rt builtins (Apache-2.0 WITH LLVM-exception)',
         read(tc / 'LICENSE.TXT')),
        ('MinGW-w64 runtime: program startup and support code',
         read(tc / 'x86_64-w64-mingw32/share/mingw32/COPYING.MinGW-w64-runtime.txt')),
    ]
    out = [INTRO]
    for n, (title, body) in enumerate(sections, 1):
        out.append(f"\n{'=' * WIDTH}\n{n}. {title}\n{'=' * WIDTH}\n\n{body}")
    OUT.write_text(''.join(out), encoding='utf-8', newline='\n')
    print(f'{OUT.name}: {OUT.stat().st_size} bytes')


if __name__ == '__main__':
    main()
