//! `name = value` text files with `#` comment lines. Lines are kept as they
//! are; `set` replaces the value of the first line with that name (later
//! duplicates are dropped) or appends a line, so comments and order survive
//! a rewrite.

use std::path::Path;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KvFile {
    lines: Vec<String>,
}

/// The name and raw value of a setting line (None for blank and comment
/// lines). The value is everything after the first `=`, trimmed.
pub fn split_line(line: &str) -> Option<(String, &str)> {
    let t = line.trim();
    if t.is_empty() || t.starts_with('#') {
        return None;
    }
    let (k, v) = t.split_once('=')?;
    Some((k.trim().to_ascii_lowercase(), v.trim()))
}

/// A value without a trailing ` # comment` (for values that are never
/// paths).
pub fn strip_comment(v: &str) -> &str {
    match v.find(" #").or_else(|| v.find("\t#")) {
        Some(i) => v[..i].trim_end(),
        None if v.starts_with('#') => "",
        None => v,
    }
}

impl KvFile {
    pub fn parse(text: &str) -> KvFile {
        KvFile { lines: text.lines().map(|l| l.trim_end_matches('\r').to_string()).collect() }
    }

    /// The file's lines, or an empty file if it does not exist.
    pub fn load(path: &Path) -> std::io::Result<KvFile> {
        match std::fs::read_to_string(path) {
            Ok(t) => Ok(KvFile::parse(&t)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(KvFile::default()),
            Err(e) => Err(e),
        }
    }

    /// Setting lines in file order: (line index, lower-case name, raw value).
    pub fn entries(&self) -> impl Iterator<Item = (usize, String, &str)> {
        self.lines.iter().enumerate().filter_map(|(i, l)| split_line(l).map(|(k, v)| (i, k, v)))
    }

    /// The raw value of the first line named `key`.
    pub fn get(&self, key: &str) -> Option<&str> {
        let key = key.to_ascii_lowercase();
        self.entries().find(|(_, k, _)| *k == key).map(|(_, _, v)| v)
    }

    pub fn set(&mut self, key: &str, value: &str) {
        let lk = key.to_ascii_lowercase();
        let idx: Vec<usize> = self.entries().filter(|(_, k, _)| *k == lk).map(|(i, _, _)| i).collect();
        let line = if value.is_empty() { format!("{key} =") } else { format!("{key} = {value}") };
        match idx.first() {
            Some(&first) => {
                self.lines[first] = line;
                for &i in idx[1..].iter().rev() {
                    self.lines.remove(i);
                }
            }
            None => self.lines.push(line),
        }
    }

    pub fn remove(&mut self, key: &str) {
        let lk = key.to_ascii_lowercase();
        let idx: Vec<usize> = self.entries().filter(|(_, k, _)| *k == lk).map(|(i, _, _)| i).collect();
        for &i in idx.iter().rev() {
            self.lines.remove(i);
        }
    }

    pub fn text(&self) -> String {
        let mut s = self.lines.join("\r\n");
        s.push_str("\r\n");
        s
    }

    /// Writes the file atomically (a temporary file renamed over it).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        write_atomic(path, self.text().as_bytes())
    }
}

/// Replaces `path` with `data` through a temporary file in the same
/// directory, so a crash leaves either the old or the new file.
pub fn write_atomic(path: &Path, data: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_keeps_comments_and_order() {
        let mut f = KvFile::parse("# head\nvsync = on\n\n# mode\nDisplay_Mode = windowed # note\nvsync = off\n");
        assert_eq!(f.get("display_mode"), Some("windowed # note"));
        assert_eq!(strip_comment(f.get("display_mode").unwrap()), "windowed");
        assert_eq!(f.get("vsync"), Some("on"));
        f.set("vsync", "fast");
        f.set("monitor", r"\\.\DISPLAY2");
        assert_eq!(f.text(), "# head\r\nvsync = fast\r\n\r\n# mode\r\nDisplay_Mode = windowed # note\r\nmonitor = \\\\.\\DISPLAY2\r\n");
        f.remove("display_mode");
        assert_eq!(f.get("display_mode"), None);
        f.set("game_dir", "");
        assert_eq!(f.get("game_dir"), Some(""));
    }

    #[test]
    fn comments_stripped_only_after_space() {
        assert_eq!(strip_comment("2.5 # faster"), "2.5");
        assert_eq!(strip_comment(r"C:\Games\#1"), r"C:\Games\#1");
        assert_eq!(strip_comment("# all comment"), "");
    }
}
