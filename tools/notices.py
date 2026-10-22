r"""Writes THIRD_PARTY_NOTICES.txt: the license notices of the third-party code
compiled into tlg.exe and tlg-launcher.exe (this project's own code: LICENSE).

    . .\tools\env.ps1
    python tools\notices.py

The crates are the ones linked into the two programs for the Windows target
(`cargo metadata` with Cargo.lock: normal dependencies, without proc-macros and
build-time crates), each with the license file it ships; where a crate offers a
choice, MIT is used. Then LibAtrac9's license, vgmstream's notice (from
hevag.rs) and the LLVM and MinGW-w64 runtime notices of the llvm-mingw toolchain.
Last, the Rust standard library's notices (the toolchain's
COPYRIGHT-library.html), each distinct notice text once.
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


def rust_std_section():
    """The Rust standard library's notices from the toolchain that builds the
    programs (its COPYRIGHT-library.html): the in-tree entries, then each
    out-of-tree component once, then each distinct notice text once with the
    components that ship it. Only the components of the target's standard
    library (its sysroot's rlibs: the file also covers crates that build
    the toolchain). As for the crates, where a component offers MIT among
    its licenses, MIT is used and only its MIT notice is kept; texts that
    differ only in line breaks and spacing count as one."""
    import html
    import os
    home, toolchain = os.environ.get('RUSTUP_HOME'), os.environ.get('RUSTUP_TOOLCHAIN')
    if not home or not toolchain:
        fail('RUSTUP_HOME / RUSTUP_TOOLCHAIN not set (run tools\\env.ps1 first)')
    src = Path(home) / 'toolchains' / toolchain / 'share/doc/rust/COPYRIGHT-library.html'
    if not src.is_file():
        fail(f'missing {src}')
    doc = src.read_text(encoding='utf-8')
    sysroot = Path(home) / 'toolchains' / toolchain / 'lib/rustlib' / TARGET / 'lib'
    linked = {re.sub(r'-[0-9a-f]+$', '', f.stem[3:]).replace('-', '_') for f in sysroot.glob('lib*.rlib')}
    if not linked:
        fail(f'no rlibs in {sysroot}')

    def plain(s):
        # (Tags dropped, entities decoded, printable ASCII and newlines kept.)
        return re.sub(r'[^\x20-\x7e\n]', '', html.unescape(re.sub(r'<[^>]+>', '', s)))

    start = doc.find('id="in-tree-files"')
    split = doc.find('id="out-of-tree-dependencies"')
    if start < 0 or split < 0:
        fail(f'unexpected layout of {src}')
    in_tree = [l.strip() for l in plain(doc[doc.find('>', start) + 1:split]).splitlines() if l.strip()]
    in_tree = [l for l in in_tree if l != 'In-tree files' and not l.startswith('<h2')]
    comps, texts = [], {}
    for block in doc[split:].split('<h3>')[1:]:
        name = plain(block[:block.find('</h3>')]).strip()
        if name.rsplit('-', 1)[0].replace('-', '_') not in linked:
            continue

        def field(k):
            m = re.search(rf'<b>{k}:</b>(.*?)</p>', block, re.S)
            return ' '.join(plain(m.group(1)).split()) if m else ''

        lic = field('License') or '?'
        notices = [(plain(f), html.unescape(b).strip('\n') + '\n') for f, b in re.findall(r'<summary><code>(.*?)</code></summary>\s*<pre>(.*?)</pre>', block, re.S)]
        mit_option = re.search(r'\bMIT\b', lic) and re.search(r'\bOR\b|/', lic)
        if mit_option:
            notices = [n for n in notices if 'mit' in n[0].lower() or 'Permission is hereby granted' in n[1]]
        used = ' (MIT used)' if mit_option else ''
        # (Without the authors' addresses: the notices carry the copyrights.)
        comps.append(f'{name} - {lic}{used}')
        for fname, text in notices:
            key = ' '.join(text.split())
            texts.setdefault(key, [text, []])[1].append(f'{name} ({fname})')
    if not comps:
        fail(f'no components found in {src}')
    parts = [
        'The Rust standard library is dual-licensed under Apache-2.0 and MIT (MIT is',
        'used); copyright the Rust Project Developers (https://thanks.rust-lang.org).',
        'Its in-tree files:',
        '',
        *in_tree,
        '',
        'The out-of-tree components it was built with:',
        '',
        *comps,
    ]
    for text, members in texts.values():
        parts.append('\n' + '-' * WIDTH + '\n' + '\n'.join(members) + '\n' + '-' * WIDTH + '\n\n' + text)
    return '\n'.join(parts)


def llvm_mingw():
    clang = shutil.which('x86_64-w64-mingw32-clang')
    if not clang:
        fail('llvm-mingw is not on PATH (run tools\\env.ps1 first)')
    return Path(clang).resolve().parent.parent


INTRO = """\
Third-party notices: tlg.exe and tlg-launcher.exe
=================================================

The Last Guardian unofficial PC port is free software under the GNU General
Public License, version 3 or (at your option) any later version (LICENSE).
Besides its own code, the two programs contain the third-party code below, used
under the licenses shown, all of which allow it in a program under the GPL
version 3. Where a component offers a choice of licenses, the one used is named.

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
        ('The Rust standard library', rust_std_section()),
    ]
    out = [INTRO]
    for n, (title, body) in enumerate(sections, 1):
        out.append(f"\n{'=' * WIDTH}\n{n}. {title}\n{'=' * WIDTH}\n\n{body}")
    OUT.write_text(''.join(out), encoding='utf-8', newline='\n')
    print(f'{OUT.name}: {OUT.stat().st_size} bytes')


if __name__ == '__main__':
    main()
