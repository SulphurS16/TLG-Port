//! HLE for PS4 system libraries other than libkernel.

pub mod ajm;
pub mod app;
pub mod atrac9;
pub mod avplayer;
pub mod devices;
pub mod extra;
pub mod fiber;
pub mod gnm;
pub mod ngs2;
pub mod ime_dialog;
pub mod online;
pub mod rtc;
pub mod savedata;
pub mod sysmodule;
pub mod videoout;

use crate::hle::Registry;

pub fn register(r: &mut Registry) {
    sysmodule::register(r);
    app::register(r);
    online::register(r);
    ime_dialog::register(r);
    devices::register(r);
    ajm::register(r);
    rtc::register(r);
    savedata::register(r);
    avplayer::register(r);
    gnm::register(r);
    videoout::register(r);
    extra::register(r);
    fiber::register(r);
    ngs2::register(r);
}
