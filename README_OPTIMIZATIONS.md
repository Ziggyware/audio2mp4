# win_loopback_mp3_mux — Optimization Notes

## v5.1.0 — Buffered streaming MP4 from disk + buffered encoding to disk

Everything below replaced a whole-file-in-RAM design with a bounded, buffered
streaming design. One constant controls every buffer in the pipeline:

```rust
pub const IO_BUF_BYTES: usize = 1 << 20; // 1 MiB
```

### 1. MP4 input: clips are streamed from disk, never held in RAM

| Before | After |
| --- | --- |
| `decode_video_clip_to_canvases()` decoded up to `min(duration*fps, 900)` RGB frames into a `Vec<RgbCanvas>` — up to ~1 GB of RGB24 at 640x640, and the frame ceiling had to be cut down by a `768 MiB` budget so it would not OOM. | `decode_video_clip_to_disk_frames()` decodes **BGRA → RGB one frame at a time** and appends it to a raw-RGB cache file through a 1 MiB `BufWriter`. Only two canvases (source + target) are ever resident. |
| Encoder indexed `scenes[i].frames[n]` directly (random access into RAM). | `VisualScene` is now `SceneStorage::Memory(Arc<Vec<RgbCanvas>>)` for images/procedural scenes, or `SceneStorage::Disk(Arc<DiskFrameSet>)` for video clips. `DiskFrameSet` reads frames through a `BufReader::with_capacity(IO_BUF_BYTES, File)`. |

Notes:

* The frame cache is per-process: `%TEMP%\win_loopback_mp4_cache\pid<N>\clip<M>.rgb`
  (override with `--frame-cache-dir <DIR>`).
* Reads are sequential (`cursor.next_index == local_idx`), so the `BufReader`
  never seeks in the common path; a wrap-around or a jump (scene switch) issues a
  single `seek`.
* `DiskFrameSet` deletes its cache file on `Drop`, and `main()` removes the whole
  directory when the session ends. Stale directories from a reused PID are purged
  on startup.
* `decode_video_clip_to_disk_frames` no longer needs a memory cap (the old
  `768 MiB / bytes_per_frame` limit is gone); the cap is now purely the requested
  duration (`--seconds` × `--fps`).
* Decoding itself was already streamed by the Media Foundation `IMFSourceReader`
  (it reads the clip from disk on demand); that behaviour is unchanged.
* The PPM image path also streams now: the header is scanned from a `BufReader`
  and the pixels go straight into the canvas with `read_exact` — no whole-file
  `Vec<u8>` and no second copy.

### 2. MP4 output: buffered encoding, direct-to-MF NV12, interleaved A/V

* **No scratch buffer copy.** `rgb24_to_nv12()` now writes *directly* into the
  locked `IMFMediaBuffer`. Previously the frame was converted into
  `nv12_scratch: Vec<u8>` and then `copy_nonoverlapping`'d into the MF buffer — a
  full extra frame copy (≈614 KB at 640x640) per video frame.
* **Reusable read buffers.** Transition frames are read into two long-lived
  `RgbCanvas` buffers (`read_a`, `read_b`); heap churn is zero per frame.
  RAM-backed scenes are still converted zero-copy via `memory_frame()`.
* **Interleaved writes.** Video and audio samples are now written
  *interleaved* (each video frame is followed by the audio slice that belongs to
  it) instead of "all video frames, then all audio in 4096-frame chunks". The
  SinkWriter therefore streams `mdat` to disk progressively rather than holding a
  long backlog and flushing it at `Finalize()`.
* **PCM goes straight into the MF buffer** (`write_pcm_chunk`) — same as before,
  but now factored out and driven by the interleave schedule; the f32→i16
  staging `Vec` is not used.
* `nv12_cache` still short-circuits fully static scenes (one NV12 frame cached per
  static scene), so a 4-scene procedural session converts each image once.

### 3. Writing to disk: one 1 MiB buffered, atomic writer for every artifact

`AtomicFileWriter` replaces all the ad-hoc `File::create` + `write_all` paths:

```
1 MiB BufWriter  →  <name>.<ext>.part  →  fsync  →  rename() over the target
```

* Previously `write_atomic_file()` wrote the whole in-RAM `Vec<u8>` through an
  **unbuffered** `File` (one syscall per internal chunk, no fsync, no atomicity).
* MP3: the `ID3v2.3 tag + LAME frames` concat `Vec` is gone — the tag and the
  encoder output are streamed straight into the buffered writer.
* M4A: `serialize_m4a_container()` used to build a compacted copy of the MP3
  stream (`clean_stream`), then `ftyp + moov + mdat` as three `Vec`s, then
  `concat()` them into a fourth. New `write_m4a_container_stream()` emits
  `ftyp`/`moov` (small) and copies `mdat` payload frame-by-frame from the MP3
  buffer, coalescing ~1 MiB writes. `parse_mp3_frame_ranges()` returns
  `(offset, len)` pairs instead of copying the stream.
* MP4 metadata injection: `inject_udta_metadata_into_mp4_file()` used to
  `fs::read()` the entire MP4, splice the `udta` box into a new `Vec` of the same
  size, and rewrite the file. It now:
  1. indexes the top-level boxes by **seek + 16-byte header reads** only,
  2. stream-copies `[0, moov_start)` → patched `moov` size → the rest of the old
     header + `moov` body → the new `udta` box → the file tail, all through a
     1 MiB buffer into the `.part` file, then `fsync` + `rename`.

## Measurement tips (Windows)

```powershell
# Peak working set before/after (no clip vs. a long clip):
Get-Process win_loopback_mp3_mux | Select-Object PeakWorkingSet64

# Confirm the frame cache is being used and cleaned up:
dir $env:TEMP\win_loopback_mp4_cache -Recurse
```

Long clips (`--seconds 300`) with `--res 1280x720` are the interesting case: the
old build needed ~2.5 GB just for decoded frames, the new one stays flat (a few
frame buffers + the audio PCM that capture itself has to keep).

## Previous optimizations (v5.0.0, unchanged)

1. Cargo: native CPU rustflags, `panic=abort`, `strip`, thin LTO (see `cargo.toml`, `.cargo/config.toml`).
2. LAME `Quality::Worst` (~4x faster than `Best`, inaudible at 192k for loopback).
3. sRGB gamma LUT (4096 entries) + cached sin/cos LUT.
4. `sample_bilinear` fast path (single clamp, integer indexing).
5. `render_physics_transition_frame`: row-parallel `std::thread::scope`.
6. `rgb24_to_nv12`: 256-entry Y/U/V LUTs instead of multiplies + clamps.
7. NV12 cache for static scenes.
8. MF H.264 knobs: CBR, GOP = fps, quality 85, CAVLC.
9. CRC32 table + deferred Adler-32 modulo (PNG writer).
10. FFT: precomputed band bins, magnitude-squared, bands updated every 4th ingest.
11. K-weighting: f32 biquads.
12. Dissolve: 256-entry sqrt LUT.
13. `resize_to`: nearest-neighbour for >2x downscale; new allocation-free `resize_into`.
