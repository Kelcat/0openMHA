// This file is part of the HörTech Open Master Hearing Aid (openMHA)
// Copyright © 2024 Hörzentrum Oldenburg gGmbH
//
// openMHA is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, version 3 of the License.
//
// openMHA is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License, version 3 for more details.
//
// You should have received a copy of the GNU Affero General Public License,
// version 3 along with openMHA.  If not, see <http://www.gnu.org/licenses/>.

/**
 * \file wola_filterbank.cpp
 * \brief WOLA (Weighted OverLap-Add) Filterbank Plugin
 *
 * Implements a WOLA filterbank that decomposes a wideband signal into
 * frequency subbands. Each subband output is a time-domain waveform
 * obtained by applying WOLA analysis (windowing + FFT), bandpass
 * filtering (rectangular band assignment in the frequency domain),
 * and WOLA synthesis (IFFT + overlap-add).
 *
 * The number of output channels is num_bands × num_input_channels.
 * For input channel ch and frequency band b, the output channel index
 * is ch * num_bands + b.
 *
 * Summing all band outputs reconstructs the original signal (perfect
 * reconstruction for COLA-compliant analysis windows such as Hann with
 * 50 % overlap).
 */

#include "mha_plugin.hh"
#include "mha_signal.hh"
#include "mha_utils.hh"
#include "windowselector.h"
#include <math.h>
#include <vector>
#include <stdexcept>

namespace wola_filterbank {

/** Runtime configuration for the WOLA filterbank.  Created during
 * prepare() and whenever configuration variables change.  Holds all
 * per-processing-block state. */
class wola_filterbank_cfg_t {
public:
    /** Construct runtime configuration.
     * @param fftlen_   FFT length (must be >= wndlen_)
     * @param wndlen_   Analysis window length in samples
     * @param fragsize_ Hop size (block size) in samples
     * @param nch_      Number of input audio channels
     * @param wndpos_   Position of the analysis window within the FFT
     *                  buffer: 0 = beginning, 0.5 = centred, 1 = end
     * @param window    Analysis window shape (wndlen_ samples, single channel)
     * @param freq_edges_hz  Vector of band-edge frequencies in Hz,
     *                       length must be num_bands + 1 and be monotonically
     *                       increasing.  Values are clamped to [0, srate/2].
     * @param srate     Sampling rate in Hz */
    wola_filterbank_cfg_t(unsigned fftlen_,
                          unsigned wndlen_,
                          unsigned fragsize_,
                          unsigned nch_,
                          mha_real_t wndpos_,
                          const MHAWindow::base_t& window,
                          const std::vector<mha_real_t>& freq_edges_hz,
                          mha_real_t srate);

    /** Destructor: frees per-band OLA buffers and the FFT handle. */
    ~wola_filterbank_cfg_t();

    /** Perform WOLA analysis, per-band synthesis, and OLA.
     * @param wave_in Input waveform block (fragsize_ × nch_).
     * @return Pointer to output waveform (fragsize_ × num_bands_×nch_).
     *         The output is valid until the next call to process(). */
    mha_wave_t* process(mha_wave_t* wave_in);

    /** @return Number of frequency bands. */
    unsigned num_bands() const { return nbands; }

private:
    unsigned nbands;    //!< number of frequency bands
    unsigned fftlen;    //!< FFT length
    unsigned wndlen;    //!< analysis window length
    unsigned fragsize;  //!< hop size
    unsigned nch;       //!< number of input channels
    unsigned npad1;     //!< zero-padding samples before window
    unsigned npad2;     //!< zero-padding samples after window
    mha_fft_t fft;      //!< FFT / IFFT handle

    /** Analysis window, pre-normalised so that OLA reconstruction of
     *  a COLA-compliant window gives unity gain. */
    MHAWindow::base_t analysis_window;

    /** Scale factor applied to each IFFT output before OLA.
     *  Chosen so that the sum of all band outputs equals the input. */
    mha_real_t synthesis_scale;

    MHASignal::waveform_t in_buf;        //!< wndlen × nch  : sliding input ring
    MHASignal::waveform_t fft_in_buf;    //!< fftlen × nch  : windowed + zero-padded
    MHASignal::spectrum_t spec;          //!< (fftlen/2+1) × nch : analysis spectrum
    MHASignal::spectrum_t band_spec_tmp; //!< (fftlen/2+1) × nch : per-band spectrum
    MHASignal::waveform_t band_wave_tmp; //!< fftlen × nch  : IFFT output (reused)

    /** Per-band OLA accumulation buffers, each fftlen × nch.
     *  Owned by this object; deleted in destructor. */
    std::vector<MHASignal::waveform_t*> ola_bufs;

    MHASignal::waveform_t write_buf; //!< fragsize × (nbands×nch) : output

    std::vector<unsigned> band_bin_start; //!< first FFT bin of each band
    std::vector<unsigned> band_bin_end;   //!< one past last FFT bin of each band
};

// ---------------------------------------------------------------------------

wola_filterbank_cfg_t::wola_filterbank_cfg_t(
    unsigned fftlen_,
    unsigned wndlen_,
    unsigned fragsize_,
    unsigned nch_,
    mha_real_t wndpos_,
    const MHAWindow::base_t& window,
    const std::vector<mha_real_t>& freq_edges_hz,
    mha_real_t srate)
    : nbands(static_cast<unsigned>(freq_edges_hz.size()) - 1u),
      fftlen(fftlen_),
      wndlen(wndlen_),
      fragsize(fragsize_),
      nch(nch_),
      npad1(static_cast<unsigned>(floor(wndpos_ * (fftlen_ - wndlen_)))),
      npad2(fftlen_ - wndlen_ - npad1),
      fft(mha_fft_new(fftlen_)),
      analysis_window(window),
      synthesis_scale(0.0f),
      in_buf(wndlen_, nch_),
      fft_in_buf(fftlen_, nch_),
      spec(fftlen_ / 2 + 1, nch_),
      band_spec_tmp(fftlen_ / 2 + 1, nch_),
      band_wave_tmp(fftlen_, nch_),
      write_buf(fragsize_, nbands * nch_)
{
    if (freq_edges_hz.size() < 2)
        throw MHA_Error(__FILE__, __LINE__,
                        "wola_filterbank: freq_bands must have at least 2 entries"
                        " (1 band), got %zu entries.",
                        freq_edges_hz.size());

    // --- Compute analysis window normalization ----------------------------
    // Normalise so that physical level is preserved (same convention as
    // wave2spec): divide by RMS of the raw window and compensate for the
    // energy dilution caused by zero-padding.
    const mha_real_t rms_of_window =
        sqrtf(analysis_window.sumsqr() / static_cast<float>(wndlen));
    const mha_real_t zeropadding_compensation =
        sqrtf(static_cast<float>(fftlen) / static_cast<float>(wndlen));
    const mha_real_t norm_factor = zeropadding_compensation / rms_of_window;
    analysis_window *= norm_factor;

    // --- Compute synthesis scale from the COLA sum -----------------------
    // For any steady-state output sample the COLA sum is
    //   cola_sum = sum_m raw_window[t_ref - m*fragsize]
    // where the sum is over all frames m that cover t_ref.  With the
    // normalised analysis window the COLA sum equals norm_factor * cola_sum.
    // The synthesis scale is chosen so that norm_factor * cola_sum * sc = 1.
    const unsigned t_ref = wndlen - 1u; // representative steady-state position
    mha_real_t cola_sum = 0.0f;
    {
        // Obtain raw (un-normalised) window values for the COLA computation.
        MHAWindow::base_t raw_window(window);
        for (unsigned m = 0; m * fragsize <= t_ref; ++m)
            cola_sum += raw_window[t_ref - m * fragsize];
    }
    if (cola_sum == 0.0f)
        throw MHA_Error(__FILE__, __LINE__,
                        "wola_filterbank: COLA sum of analysis window is zero;"
                        " check window type and hop size.");
    synthesis_scale = 1.0f / (norm_factor * cola_sum);

    // --- Compute FFT bin boundaries from frequency edges -----------------
    const unsigned num_spec_bins = fftlen / 2 + 1;
    band_bin_start.resize(nbands);
    band_bin_end.resize(nbands);
    for (unsigned b = 0; b < nbands; ++b) {
        // Map Hz to FFT bin index: bin = round(f / srate * fftlen)
        // Clamp to valid range [0, num_spec_bins).
        auto hz_to_bin = [&](mha_real_t f_hz) -> unsigned {
            float bin_f = f_hz / srate * static_cast<float>(fftlen);
            int bin = static_cast<int>(bin_f + 0.5f);
            if (bin < 0) bin = 0;
            if (static_cast<unsigned>(bin) >= num_spec_bins)
                bin = static_cast<int>(num_spec_bins) - 1;
            return static_cast<unsigned>(bin);
        };
        band_bin_start[b] = hz_to_bin(freq_edges_hz[b]);
        band_bin_end[b]   = hz_to_bin(freq_edges_hz[b + 1]);
        if (band_bin_start[b] >= band_bin_end[b])
            throw MHA_Error(__FILE__, __LINE__,
                            "wola_filterbank: band %u has no FFT bins"
                            " (start bin %u >= end bin %u); check freq_bands"
                            " values relative to fftlen and srate.",
                            b, band_bin_start[b], band_bin_end[b]);
    }

    // --- Allocate per-band OLA buffers -----------------------------------
    ola_bufs.resize(nbands, nullptr);
    for (unsigned b = 0; b < nbands; ++b) {
        ola_bufs[b] = new MHASignal::waveform_t(fftlen, nch);
        memset(ola_bufs[b]->buf, 0,
               sizeof(mha_real_t) * fftlen * nch);
    }
}

wola_filterbank_cfg_t::~wola_filterbank_cfg_t()
{
    for (unsigned b = 0; b < nbands; ++b)
        delete ola_bufs[b];
    mha_fft_free(fft);
}

mha_wave_t* wola_filterbank_cfg_t::process(mha_wave_t* wave_in)
{
    // --- Step 1: Update sliding input buffer ----------------------------
    // Shift in_buf left by fragsize, copy wave_in to the right part.
    in_buf.copy_from_at(0, wndlen - fragsize, in_buf, fragsize);
    in_buf.copy_from_at(wndlen - fragsize, fragsize, *wave_in, 0);

    // --- Step 2: Apply analysis window and zero-padding -----------------
    for (unsigned ch = 0; ch < nch; ++ch) {
        // Zero-padding before window
        for (unsigned k = 0; k < npad1; ++k)
            value(&fft_in_buf, k, ch) = 0.0f;
        // Windowed signal
        for (unsigned k = 0; k < wndlen; ++k)
            value(&fft_in_buf, npad1 + k, ch) =
                value(&in_buf, k, ch) * analysis_window[k];
        // Zero-padding after window
        for (unsigned k = npad1 + wndlen; k < fftlen; ++k)
            value(&fft_in_buf, k, ch) = 0.0f;
    }

    // --- Step 3: Forward FFT ----------------------------------------
    mha_fft_wave2spec(fft, &fft_in_buf, &spec);

    const unsigned num_spec_bins = fftlen / 2 + 1;

    // --- Step 4: Per-band IFFT + OLA ------------------------------------
    for (unsigned b = 0; b < nbands; ++b) {
        // 4a: Copy analysis spectrum to band_spec_tmp, zero out-of-band bins
        for (unsigned ch = 0; ch < nch; ++ch) {
            for (unsigned k = 0; k < num_spec_bins; ++k) {
                if (k >= band_bin_start[b] && k < band_bin_end[b]) {
                    value(&band_spec_tmp, k, ch) = value(&spec, k, ch);
                } else {
                    value(&band_spec_tmp, k, ch).re = 0.0f;
                    value(&band_spec_tmp, k, ch).im = 0.0f;
                }
            }
        }

        // 4b: Inverse FFT
        mha_fft_spec2wave(fft, &band_spec_tmp, &band_wave_tmp);

        // 4c: Scale IFFT output and OLA
        MHASignal::waveform_t& ola = *ola_bufs[b];
        // Shift OLA buffer left by fragsize
        ola.copy_from_at(0, fftlen - fragsize, ola, fragsize);
        // Zero the right part
        for (unsigned k = fftlen - fragsize; k < fftlen; ++k)
            ola.assign_frame(k, 0.0f);
        // Overlap-add: accumulate scaled IFFT output
        for (unsigned fr = 0; fr < fftlen; ++fr)
            for (unsigned ch = 0; ch < nch; ++ch)
                value(&ola, fr, ch) +=
                    value(&band_wave_tmp, fr, ch) * synthesis_scale;

        // 4d: Copy first fragsize samples to output channels ch*nbands+b
        for (unsigned fr = 0; fr < fragsize; ++fr)
            for (unsigned ch = 0; ch < nch; ++ch)
                value(&write_buf, fr, ch * nbands + b) =
                    value(&ola, fr, ch);
    }

    return &write_buf;
}

// ---------------------------------------------------------------------------

/** Plugin interface class for the WOLA filterbank. */
class wola_filterbank_if_t
    : public MHAPlugin::plugin_t<wola_filterbank_cfg_t> {
public:
    wola_filterbank_if_t(MHA_AC::algo_comm_t& iac,
                         const std::string& configured_name);

    /** Validate signal dimensions, compute output channel count, and create
     * the runtime configuration. */
    void prepare(mhaconfig_t&);

    /** Release runtime resources. */
    void release();

    /** Process one block: delegate to runtime configuration. */
    mha_wave_t* process(mha_wave_t*);

private:
    /** (Re-)create runtime configuration from current parser variables. */
    void update();

    /** Lock / unlock all configuration variables.
     * @param b True to lock (called from prepare), false to unlock (release). */
    void setlock(bool b);

    MHAParser::int_t   nfft;   //!< FFT length
    MHAParser::int_t   nwnd;   //!< Analysis window length
    MHAParser::float_t wndpos; //!< Window position within FFT buffer

    windowselector_t window_config; //!< Analysis window type selector

    /** Band-edge frequencies in Hz.  Must be a monotonically increasing
     *  vector with at least 2 entries.  The number of bands is
     *  freq_bands.data.size() - 1. */
    MHAParser::vfloat_t freq_bands;

    /** When true, disallow window sizes that are not a power-of-two
     *  multiple of the hop size. */
    MHAParser::bool_t strict_window_ratio;

    mhaconfig_t tftype_in; //!< input signal dimensions (stored in prepare)
};

wola_filterbank_if_t::wola_filterbank_if_t(MHA_AC::algo_comm_t& iac,
                                           const std::string& /*configured_name*/)
    : MHAPlugin::plugin_t<wola_filterbank_cfg_t>(
          "WOLA (Weighted OverLap-Add) analysis-synthesis filterbank.\n\n"
          "Decomposes a multi-channel waveform signal into frequency subbands "
          "using Weighted OverLap-Add (WOLA) analysis, rectangular bandpass "
          "filtering in the FFT domain, and WOLA synthesis.\n\n"
          "Each output block contains fragsize samples for every "
          "(input_channel, band) pair. Output channel ch*num_bands+b "
          "carries the subband signal of input channel ch filtered to band b.\n\n"
          "Summing all band outputs per input channel reconstructs the "
          "original signal with unity gain for COLA-compliant windows "
          "(e.g. Hann at 50 % overlap).",
          iac),
      nfft("FFT length", "512", "[1,]"),
      nwnd("Analysis window length in samples", "400", "[1,]"),
      wndpos("Position of analysis window in FFT buffer\n"
             "(0 = beginning, 0.5 = centred, 1 = end)",
             "0.5", "[0,1]"),
      window_config("hanning"),
      freq_bands("Band-edge frequencies in Hz (monotonically increasing).\n"
                 "Number of bands = size - 1.\n"
                 "Example for 4 bands with 16 kHz sample rate:\n"
                 "  [0 1000 2000 4000 8000]",
                 "[0 1000 2000 4000 8000]"),
      strict_window_ratio(
          "Disallow window sizes that are not a power-of-two multiple of "
          "the hop size (fragsize).",
          "yes")
{
    insert_item("fftlen", &nfft);
    insert_item("wndlen", &nwnd);
    insert_item("wndpos", &wndpos);
    window_config.insert_items(this);
    insert_item("freq_bands", &freq_bands);
    insert_item("strict_window_ratio", &strict_window_ratio);
}

void wola_filterbank_if_t::prepare(mhaconfig_t& t)
{
    try {
        setlock(true);

        if (t.domain != MHA_WAVEFORM)
            throw MHA_ErrorMsg(
                "wola_filterbank: waveform input is required.");

        if (nfft.data < static_cast<int>(nwnd.data))
            throw MHA_Error(__FILE__, __LINE__,
                            "wola_filterbank: FFT length (%d) must be >= "
                            "window length (%d).",
                            nfft.data, nwnd.data);

        if (static_cast<unsigned>(nwnd.data) < t.fragsize)
            throw MHA_Error(__FILE__, __LINE__,
                            "wola_filterbank: window length (%d) must be >= "
                            "fragsize (%u).",
                            nwnd.data, t.fragsize);

        if (strict_window_ratio.data) {
            if (static_cast<unsigned>(nwnd.data) == t.fragsize ||
                !MHAUtils::is_multiple_of_by_power_of_two(
                    static_cast<unsigned>(nwnd.data), t.fragsize))
                throw MHA_Error(
                    __FILE__, __LINE__,
                    "wola_filterbank: The ratio between the hop size"
                    " (\"fragsize\", %u) and the window length (%d)"
                    " must be a power of two (2, 4, 8, ...).",
                    t.fragsize, nwnd.data);
        }

        if (freq_bands.data.size() < 2)
            throw MHA_Error(__FILE__, __LINE__,
                            "wola_filterbank: freq_bands must have at least "
                            "2 entries (defining 1 band), has %zu entries.",
                            freq_bands.data.size());

        tftype_in = t;
        t.fftlen = static_cast<unsigned>(nfft.data);
        t.wndlen = static_cast<unsigned>(nwnd.data);

        // Output: fragsize samples per (channel × band)
        unsigned nbands = static_cast<unsigned>(freq_bands.data.size()) - 1u;
        t.channels *= nbands;

        // Domain stays MHA_WAVEFORM
        update();
    }
    catch (MHA_Error&) {
        setlock(false);
        throw;
    }
}

void wola_filterbank_if_t::release()
{
    setlock(false);
}

void wola_filterbank_if_t::update()
{
    if (is_prepared()) {
        const MHAWindow::base_t& win =
            window_config.get_window_data(
                static_cast<unsigned>(nwnd.data));

        std::vector<mha_real_t> edges(freq_bands.data.begin(),
                                       freq_bands.data.end());

        push_config(new wola_filterbank_cfg_t(
            static_cast<unsigned>(nfft.data),
            static_cast<unsigned>(nwnd.data),
            tftype_in.fragsize,
            tftype_in.channels,
            wndpos.data,
            win,
            edges,
            static_cast<mha_real_t>(tftype_in.srate)));
    }
}

mha_wave_t* wola_filterbank_if_t::process(mha_wave_t* wave_in)
{
    poll_config();
    return cfg->process(wave_in);
}

void wola_filterbank_if_t::setlock(bool b)
{
    nfft.setlock(b);
    nwnd.setlock(b);
    wndpos.setlock(b);
    freq_bands.setlock(b);
    strict_window_ratio.setlock(b);
    window_config.setlock(b);
}

} // namespace wola_filterbank

MHAPLUGIN_CALLBACKS(wola_filterbank,
                    wola_filterbank::wola_filterbank_if_t,
                    wave, wave)
MHAPLUGIN_DOCUMENTATION
(wola_filterbank,
 "filterbank signal-transformation overlap-add",
 "The plugin {\\tt wola\\_filterbank} implements a Weighted OverLap-Add"
 " (WOLA) analysis-synthesis filterbank.\n\n"

 "The filterbank decomposes a multi-channel waveform signal into"
 " frequency subbands.  For each processing block the following steps"
 " are performed:\n\n"

 "\\begin{enumerate}\n"
 "\\item {\\bf Analysis:} The input samples are accumulated in a sliding"
 " buffer of length {\\tt wndlen}.  The buffer is multiplied sample-wise"
 " by the analysis window (default: Hann), zero-padded to {\\tt fftlen}"
 " samples, and transformed by a forward FFT.\n\n"
 "\\item {\\bf Band filtering:} For each band $b$ the full analysis"
 " spectrum is copied and all FFT bins outside the band are set to zero."
 " The band edges are computed from the {\\tt freq\\_bands} configuration"
 " variable, which holds the edge frequencies in Hz.\n\n"
 "\\item {\\bf Synthesis:} Each band spectrum is inverse-FFT transformed."
 " The resulting waveform is scaled and added to the per-band"
 " overlap-add accumulation buffer.  The first {\\tt fragsize} samples"
 " of each band buffer are written to the output.\n"
 "\\end{enumerate}\n\n"

 "The number of output channels is $\\mathrm{num\\_bands} \\times"
 " \\mathrm{num\\_input\\_channels}$.  Output channel"
 " $ch \\cdot \\mathrm{num\\_bands} + b$ carries the subband signal of"
 " input channel $ch$, filtered to band $b$.\n\n"

 "For a Hann analysis window at 50\\% overlap"
 " (i.e.\\ {\\tt fragsize} $=$ {\\tt wndlen}$/2$),"
 " summing all band outputs per input channel reconstructs the original"
 " signal with unity gain.  Other COLA-compliant windows and overlap"
 " ratios are also supported.\n\n"

 "The synthesis scale is computed numerically from the COLA sum of the"
 " analysis window at the configured overlap ratio, so the plugin is"
 " self-calibrating and does not require manual gain adjustments when"
 " window type or overlap ratio is changed.\n\n"

 "Configuration example (4 bands, 16 kHz, 50\\% overlap):\n"
 "\\begin{verbatim}\n"
 "srate = 16000\n"
 "fragsize = 200\n"
 "nchannels_in = 1\n"
 "mhalib = wola_filterbank\n"
 "mha.fftlen = 512\n"
 "mha.wndlen = 400\n"
 "mha.freq_bands = [0 1000 2000 4000 8000]\n"
 "\\end{verbatim}\n"
)

// Local Variables:
// compile-command: "make"
// c-basic-offset: 4
// indent-tabs-mode: nil
// coding: utf-8-unix
// End:
