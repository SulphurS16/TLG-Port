//! libSceSysmodule: system module loading. All system services used by the
//! title are provided by HLE, so loading only records the request.

use crate::hle::Registry;
use crate::{reg, sysv};
use parking_lot::Mutex;
use std::collections::BTreeSet;

static LOADED: Mutex<BTreeSet<u32>> = Mutex::new(BTreeSet::new());

sysv! {
    fn sceSysmoduleLoadModule(id: u32) -> i32 {
        crate::info!("sceSysmoduleLoadModule({id:#x}) -> provided by HLE");
        LOADED.lock().insert(id);
        0
    }
    fn sceSysmoduleUnloadModule(id: u32) -> i32 {
        LOADED.lock().remove(&id);
        0
    }
    fn sceSysmoduleIsLoaded(id: u32) -> i32 {
        if LOADED.lock().contains(&id) { 0 } else { 0x8012_0002u32 as i32 }
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceSysmodule", [sceSysmoduleLoadModule, sceSysmoduleUnloadModule, sceSysmoduleIsLoaded]);
}
