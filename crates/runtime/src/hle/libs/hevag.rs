//! HE-VAG ("High Efficiency VAG"), the PS4's and Vita's extension of PS-ADPCM
//! (VAG): the same 16-byte frames of 28 samples (byte 0: predictor index low
//! nibble and shift; byte 1: predictor index high nibble and flags; then 14
//! bytes of 4-bit codes, low nibble first), with predictor indexes up to 127
//! over the last four samples. Indexes 0-4 are VAG's own filters, so plain VAG
//! decodes as before. TLG's sound effects (footsteps, Trico, chains) use the
//! extended indexes in about a quarter of their frames (docs/LOG.md: Audio:
//! HE-VAG sound effects); read as VAG they had decoded to noise.
//!
//! The predictor table is from vgmstream (`src/coding/psv_decoder.c`), under
//! this notice:
//!
//! Copyright (c) 2008-2025 Adam Gashlin, Fastelbja, Ronny Elfert, bnnm,
//!                         Christopher Snowhill, NicknineTheEagle, bxaimc,
//!                         Thealexbarney, CyberBotX, EdnessP, et al
//!
//! Portions Copyright (c) 2004-2008, Marko Kreen
//! Portions Copyright 2001-2007  jagarl / Kazunori Ueno <jagarl@creator.club.ne.jp>
//! Portions Copyright (c) 1998, Justin Frankel/Nullsoft Inc.
//! Portions Copyright (C) 2006 Nullsoft, Inc.
//! Portions Copyright (c) 2005-2007 Paul Hsieh
//! Portions Copyright (C) 2000-2004 Leshade Entis, Entis-soft.
//! Portions Public Domain originating with Sun Microsystems
//!
//! Permission to use, copy, modify, and distribute this software for any
//! purpose with or without fee is hereby granted, provided that the above
//! copyright notice and this permission notice appear in all copies.
//!
//! THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
//! WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
//! MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
//! ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
//! WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
//! ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
//! OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

/// Bytes and samples of one frame (per channel).
pub const FRAME_BYTES: usize = 16;
pub const FRAME_SAMPLES: usize = 28;

/// Prediction coefficients per index, for the last four samples (newest
/// first).
#[rustfmt::skip]
const COEF: [[f32; 4]; 128] = [
    [0.0, 0.0, 0.0, 0.0],
    [0.9375, 0.0, 0.0, 0.0],
    [1.796875, -0.8125, 0.0, 0.0],
    [1.53125, -0.859375, 0.0, 0.0],
    [1.90625, -0.9375, 0.0, 0.0],
    [1.7982178, -0.86169434, 0.0, 0.0],
    [1.770874, -0.89916992, 0.0, 0.0],
    [1.6992188, -0.91821289, 0.0, 0.0],
    [1.6031494, -0.9375, 0.0, 0.0],
    [1.4682617, -0.9375, 0.0, 0.0],
    [1.3139648, -0.9375, 0.0, 0.0],
    [1.1424561, -0.9375, 0.0, 0.0],
    [0.95605469, -0.9375, 0.0, 0.0],
    [0.75695801, -0.9375, 0.0, 0.0],
    [0.54785156, -0.9375, 0.0, 0.0],
    [0.33166504, -0.9375, 0.0, 0.0],
    [0.11108398, -0.9375, 0.0, 0.0],
    [-0.11108398, -0.9375, 0.0, 0.0],
    [-0.33166504, -0.9375, 0.0, 0.0],
    [-0.54785156, -0.9375, 0.0, 0.0],
    [-0.75695801, -0.9375, 0.0, 0.0],
    [-0.95605469, -0.9375, 0.0, 0.0],
    [-1.1424561, -0.9375, 0.0, 0.0],
    [-1.3139648, -0.9375, 0.0, 0.0],
    [-1.4682617, -0.9375, 0.0, 0.0],
    [-1.6031494, -0.9375, 0.0, 0.0],
    [-1.6992188, -0.91821289, 0.0, 0.0],
    [-1.770874, -0.89916992, 0.0, 0.0],
    [-1.7982178, -0.86169434, 0.0, 0.0],
    [0.65625, -1.125, 0.40625, -0.375],
    [-0.78125, -0.875, -0.40625, -0.28125],
    [-1.28125, -0.90625, -0.4375, -0.125],
    [-0.020385742, -0.33227539, -0.060302734, -0.066040039],
    [-0.90698242, -0.27111816, -0.28051758, 0.051757812],
    [-0.97668457, -0.38647461, -0.34350586, 0.03527832],
    [0.73461914, -0.57983398, 0.32336426, -0.15844727],
    [0.46362305, -0.84790039, 0.47302246, -0.1484375],
    [-1.0054932, -0.31689453, -0.25280762, 0.027709961],
    [1.1229248, 0.24194336, -0.16870117, -0.28271484],
    [1.5894775, -0.37158203, -0.46289062, 0.15466309],
    [1.6005859, -0.54772949, -0.2746582, 0.20324707],
    [-0.20361328, -0.45703125, -0.78808594, 0.10253906],
    [0.95446777, -0.52832031, 0.25769043, -0.061767578],
    [1.168335, -0.16308594, -0.092407227, 0.059448242],
    [1.2246094, -0.31274414, 0.036621094, 0.024291992],
    [-0.57922363, -0.50317383, -0.66967773, -0.18225098],
    [-0.71972656, 0.2902832, -0.58435059, -0.84802246],
    [-0.14562988, -1.112915, -0.15100098, -0.38012695],
    [0.33972168, -0.86767578, -0.19226074, -0.17663574],
    [-0.89526367, -0.25170898, -0.27001953, 0.054443359],
    [0.7479248, -0.3145752, -0.038452148, -0.0021972656],
    [1.1544189, -0.22680664, 0.012451172, 0.031494141],
    [0.96142578, -0.54724121, 0.25952148, -0.065673828],
    [-0.87548828, -0.21911621, -0.25256348, 0.058837891],
    [-0.89819336, -0.2565918, -0.27258301, 0.053710938],
    [-1.1193848, -0.42834473, -0.32641602, -0.047729492],
    [-0.32202148, -0.32312012, -0.23547363, -0.1998291],
    [0.2286377, 1.1209717, 0.22705078, -0.70141602],
    [1.1247559, 0.22692871, -0.13720703, -0.29626465],
    [1.6118164, -0.36767578, -0.50524902, 0.16723633],
    [1.5181885, -0.58496094, -0.03125, 0.075927734],
    [-0.32385254, -0.13964844, -0.38842773, -0.83959961],
    [1.1390381, -0.12792969, -0.10107422, 0.061889648],
    [0.20043945, -0.075683594, -0.11547852, -0.51623535],
    [0.51831055, -0.92590332, -0.065063477, -0.27575684],
    [-1.097168, -0.47497559, -0.34265137, 0.0053710938],
    [-0.31274414, -0.3338623, -0.21118164, -0.23181152],
    [0.38842773, -0.058959961, -0.087158203, -0.17346191],
    [0.96887207, -0.46923828, 0.34436035, -0.12438965],
    [1.229126, -0.31848145, 0.038330078, 0.023803711],
    [1.0253906, -0.40246582, 0.18933105, -0.018920898],
    [-1.0411377, -0.33874512, -0.296875, -0.041015625],
    [1.1568604, -0.22973633, 0.013183594, 0.03125],
    [0.0091552734, -0.27355957, -0.036376953, -0.84680176],
    [-1.1160889, -0.5078125, -0.36169434, 0.00061035156],
    [-0.88745117, -0.23901367, -0.26318359, 0.056152344],
    [-0.33447266, 0.45715332, 0.72460938, -0.13293457],
    [1.0977783, 0.23779297, -0.083374023, -0.33007812],
    [1.5992432, -0.34606934, -0.47045898, 0.12878418],
    [1.164917, -0.23937988, 0.015869141, 0.030517578],
    [0.64355469, -0.52124023, 0.38134766, -0.38537598],
    [-0.93945312, -0.41296387, -0.3548584, -0.055664062],
    [0.89221191, 0.3079834, 0.052978516, -0.30041504],
    [1.2542725, -0.34997559, 0.047729492, 0.020996094],
    [1.3354492, -0.45422363, 0.081176758, 0.01184082],
    [0.0029296875, -0.037841797, -0.15405273, 0.0390625],
    [-0.99145508, -0.29431152, -0.28210449, -0.033081055],
    [-1.0389404, -0.37438965, -0.28527832, 0.019897461],
    [0.039794922, -0.46948242, 0.051147461, -0.1138916],
    [1.0858154, 0.26782227, -0.066040039, -0.3515625],
    [1.4737549, -0.22900391, -0.24621582, -0.073364258],
    [1.0655518, -0.41784668, 0.2043457, -0.020629883],
    [1.5808105, -0.46960449, -0.36706543, 0.23754883],
    [1.2253418, -0.3137207, 0.036865234, 0.024169922],
    [1.1456299, -0.33654785, 0.12304688, 0.0050048828],
    [-0.57617188, -0.61108398, -0.34814453, -0.14172363],
    [0.96057129, -0.52807617, 0.26062012, -0.061157227],
    [0.29907227, -1.0494385, 0.15856934, -0.33935547],
    [1.2441406, -0.33728027, 0.043945312, 0.022094727],
    [1.3809814, -0.51428223, 0.10168457, 0.0064697266],
    [1.239502, -0.33154297, 0.042114258, 0.022583008],
    [1.1765137, -0.17297363, -0.08996582, 0.058837891],
    [0.47045898, -0.5559082, 0.3470459, -0.41467285],
    [0.81774902, -0.6907959, 0.27453613, -0.13110352],
    [1.3527832, -0.47705078, 0.088867188, 0.009765625],
    [-0.12524414, -1.1975098, -0.098266602, -0.42260742],
    [1.269043, -0.45727539, 0.16687012, -0.01171875],
    [1.2557373, 0.12060547, -0.23376465, -0.17541504],
    [0.9708252, 0.47338867, -0.093261719, -0.39831543],
    [1.5489502, -0.4119873, -0.40942383, 0.25378418],
    [0.81066895, 0.38647461, 0.028198242, -0.25500488],
    [-0.28662109, -0.89770508, -0.23730469, -0.50317383],
    [1.1340332, -0.49304199, 0.23010254, -0.030029297],
    [0.56555176, -0.78161621, 0.21337891, -0.19763184],
    [1.3729248, -0.50354004, 0.097900391, 0.0074462891],
    [1.1971436, -0.27880859, 0.026733398, 0.027099609],
    [1.1884766, -0.1875, -0.086181641, 0.057739258],
    [1.0302734, -0.41943359, 0.19067383, -0.021484375],
    [1.1361084, -0.12463379, -0.10192871, 0.062133789],
    [0.20727539, -1.1016846, 0.083984375, -0.37072754],
    [1.2468262, -0.34069824, 0.044921875, 0.021850586],
    [1.0241699, 0.39648438, -0.092529297, -0.36486816],
    [0.87902832, 0.40478516, 0.0056152344, -0.3190918],
    [-0.010742188, -0.95324707, -0.065673828, -0.5579834],
    [0.75598145, -0.63342285, 0.33691406, -0.15197754],
    [1.5045166, -0.1574707, -0.40087891, 0.030883789],
    [1.5947266, -0.49743652, -0.34472656, 0.22912598],
    [0.65100098, 0.36608887, 0.094604492, -0.13818359],
];

/// Decodes one frame of one channel into 28 samples (full scale 1.0).
/// `hist` holds the channel's last four decoded samples, newest first, at
/// 16-bit scale; the decoder keeps them unclamped (only the output is
/// clamped). A frame with flag 7 (the end marker) decodes to silence.
pub fn decode_frame(fr: &[u8], hist: &mut [f32; 4]) -> [f32; FRAME_SAMPLES] {
    let shift = (fr[0] & 0x0F) as u32;
    // (Indexes above 127 do not occur; as in vgmstream, they predict nothing.)
    let c = COEF.get(((fr[1] & 0xF0) | (fr[0] >> 4)) as usize).copied().unwrap_or([0.0; 4]);
    let silent = fr[1] & 0x0F >= 7;
    let mut out = [0f32; FRAME_SAMPLES];
    for (i, o) in out.iter_mut().enumerate() {
        let s = if silent {
            0.0
        } else {
            let byte = fr[2 + i / 2];
            let nib = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            let code = ((((nib as i32) << 28) >> 28) << 12) >> shift;
            code as f32 + hist[0] * c[0] + hist[1] * c[1] + hist[2] * c[2] + hist[3] * c[3]
        };
        *hist = [s, hist[0], hist[1], hist[2]];
        *o = s.clamp(-32768.0, 32767.0) / 32768.0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame of `codes` (28 signed 4-bit values) with predictor `index`,
    /// `shift` and `flag`.
    fn frame(index: u8, shift: u8, flag: u8, codes: &[i8; 28]) -> [u8; 16] {
        let mut fr = [0u8; 16];
        fr[0] = ((index & 0x0F) << 4) | (shift & 0x0F);
        fr[1] = (index & 0xF0) | (flag & 0x0F);
        for (i, &c) in codes.iter().enumerate() {
            let n = (c as u8) & 0x0F;
            fr[2 + i / 2] |= if i % 2 == 0 { n } else { n << 4 };
        }
        fr
    }

    #[test]
    fn vag_predictors_decode_as_vag() {
        // Indexes 0-4 with VAG's two-tap filters (x / 64), history at 16-bit
        // scale: the PS-ADPCM recurrence.
        let codes: [i8; 28] = std::array::from_fn(|i| ((i * 5) % 15) as i8 - 7);
        let vag = [(0.0, 0.0), (60.0, 0.0), (115.0, -52.0), (98.0, -55.0), (122.0, -60.0)];
        for (index, &(a, b)) in vag.iter().enumerate() {
            let mut hist = [1000.0, -500.0, 0.0, 0.0];
            let out = decode_frame(&frame(index as u8, 8, 0, &codes), &mut hist);
            let (mut h1, mut h2) = (1000.0f32, -500.0f32);
            for (i, &c) in codes.iter().enumerate() {
                let s = ((c as i32) << 12 >> 8) as f32 + h1 * (a / 64.0) + h2 * (b / 64.0);
                (h2, h1) = (h1, s);
                assert!((out[i] - s.clamp(-32768.0, 32767.0) / 32768.0).abs() < 1e-6, "index {index} sample {i}");
            }
        }
    }

    #[test]
    fn extended_predictors_use_four_taps_and_byte_1() {
        // Index 125 = 0x7d: low nibble 0xd in byte 0, high nibble 0x70 in
        // byte 1 (VAG would read predictor 13 and clamp it to 4).
        let codes = [0i8; 28];
        let mut hist = [100.0, 50.0, -20.0, 10.0];
        let out = decode_frame(&frame(125, 0, 0, &codes), &mut hist);
        let c = COEF[125];
        let want = 100.0 * c[0] + 50.0 * c[1] - 20.0 * c[2] + 10.0 * c[3];
        assert!((out[0] * 32768.0 - want).abs() < 1e-3, "{} vs {want}", out[0] * 32768.0);
        // The history moved along by 28 samples, newest first.
        assert!((hist[3] - out[24] * 32768.0).abs() < 1e-2);
        assert!((hist[0] - out[27] * 32768.0).abs() < 1e-2);
    }

    #[test]
    fn end_flag_frames_are_silent() {
        let codes = [7i8; 28];
        let mut hist = [1000.0, 1000.0, 1000.0, 1000.0];
        let out = decode_frame(&frame(1, 0, 7, &codes), &mut hist);
        assert!(out.iter().all(|&x| x == 0.0));
        assert_eq!(hist, [0.0; 4]);
    }

    #[test]
    fn output_clamps_but_history_does_not() {
        // Code 7 at shift 0 is 28672; with predictor 1 (0.9375) the second
        // sample would be 28672 + 26880 = 55552: output at full scale, the
        // history keeps the value.
        let mut codes = [0i8; 28];
        codes[0] = 7;
        codes[1] = 7;
        let mut hist = [0.0; 4];
        let out = decode_frame(&frame(1, 0, 0, &codes), &mut hist);
        assert_eq!(out[0], 28672.0 / 32768.0);
        assert!((out[1] - 32767.0 / 32768.0).abs() < 1e-6);
        // The last sample decays from the unclamped 55552 (from 32767 if the
        // history were clamped).
        let want = 55552.0 * 0.9375f32.powi(26);
        assert!((hist[0] - want).abs() < 0.5, "{} vs {want}", hist[0]);
    }
}
