// Shared pure-JS streaming GIF89a encoder with shared 256-color Global Color Table
// and LSB-first LZW compression. Used by both playground-worker.js (off-thread with
// cooperative cancellation and progress callbacks) and playground.js (main-thread fallback).

/**
 * Encodes a sequence of RGBA frames into a looping GIF89a Uint8Array.
 *
 * @param {object} opts
 * @param {number} opts.frameCount Number of frames to encode.
 * @param {(frameIdx: number) => Uint8Array} opts.getFrameRgba Callback returning RGBA bytes for a frame.
 * @param {number} opts.width Frame width in pixels.
 * @param {number} opts.height Frame height in pixels.
 * @param {number} [opts.delayMs=150] Delay between frames in milliseconds.
 * @param {string[]} [opts.levelColors=[]] Hex palette colors to seed first.
 * @param {() => boolean} [opts.isCancelled] Optional cooperative cancellation predicate.
 * @param {(fraction: number, message: string) => void} [opts.onProgress] Optional progress callback.
 * @param {() => Promise<void>} [opts.yieldTick] Optional async yield hook for responsive workers.
 * @returns {Promise<Uint8Array | null>} Encoded GIF89a bytes, or null if cancelled.
 */
export async function encodeGif89a({
  frameCount,
  getFrameRgba,
  width,
  height,
  delayMs = 150,
  levelColors = [],
  isCancelled,
  onProgress,
  yieldTick,
}) {
  if (!Number.isFinite(frameCount) || frameCount <= 0) {
    throw new Error("No frames supplied for GIF encoding.");
  }
  const nPix = width * height;
  const delayCs = Math.max(2, Math.round((delayMs || 150) / 10));

  if (onProgress) {
    onProgress(0.48, "Quantizing depression tree GIF palette...");
  }

  // 1. Sample up to 16 evenly spaced frames to build a 5-5-5 RGB frequency histogram.
  const counts = new Uint32Array(32768);
  const sumR = new Float64Array(32768);
  const sumG = new Float64Array(32768);
  const sumB = new Float64Array(32768);
  const sampleCount = Math.min(frameCount, 16);
  for (let s = 0; s < sampleCount; s++) {
    if (isCancelled && isCancelled()) return null;
    const fi = sampleCount === 1 ? 0 : Math.round((s * (frameCount - 1)) / (sampleCount - 1));
    const rgba = getFrameRgba(fi);
    for (let i = 0; i < rgba.length; i += 4) {
      const r = rgba[i];
      const g = rgba[i + 1];
      const b = rgba[i + 2];
      const key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
      counts[key]++;
      sumR[key] += r;
      sumG[key] += g;
      sumB[key] += b;
    }
    if (yieldTick && (s & 3) === 3) {
      await yieldTick();
    }
  }

  const buckets = [];
  for (let k = 0; k < 32768; k++) {
    if (counts[k] > 0) buckets.push(k);
  }
  buckets.sort((a, b) => counts[b] - counts[a]);

  const palette = new Uint8Array(256 * 3);
  let palLen = 0;
  // Seed exact hierarchy level colors first so node fills never drift during quantization.
  for (const hex of levelColors) {
    if (palLen >= 256) break;
    const h = String(hex).replace("#", "");
    palette[palLen * 3] = parseInt(h.slice(0, 2), 16) || 0;
    palette[palLen * 3 + 1] = parseInt(h.slice(2, 4), 16) || 0;
    palette[palLen * 3 + 2] = parseInt(h.slice(4, 6), 16) || 0;
    palLen++;
  }
  // Cap total palette entries (seeded level colors + data buckets) at 192 so all 64 colors
  // of the 4x4x4 fallback cube fit in 256 slots.
  const maxDataColors = 192;
  for (let i = 0; i < buckets.length && palLen < maxDataColors; i++) {
    const k = buckets[i];
    const c = counts[k];
    palette[palLen * 3] = Math.round(sumR[k] / c);
    palette[palLen * 3 + 1] = Math.round(sumG[k] / c);
    palette[palLen * 3 + 2] = Math.round(sumB[k] / c);
    palLen++;
  }
  const levels = [0, 85, 170, 255];
  for (let r = 0; r < 4 && palLen < 256; r++) {
    for (let g = 0; g < 4 && palLen < 256; g++) {
      for (let b = 0; b < 4 && palLen < 256; b++) {
        palette[palLen * 3] = levels[r];
        palette[palLen * 3 + 1] = levels[g];
        palette[palLen * 3 + 2] = levels[b];
        palLen++;
      }
    }
  }

  // Lazy O(1) 15-bit RGB -> palette index lookup cache.
  const lut = new Int16Array(32768).fill(-1);
  function nearestColor(key, r, g, b) {
    const cached = lut[key];
    if (cached >= 0) return cached;
    let bestIdx = 0;
    let bestDist = Infinity;
    for (let p = 0; p < palLen; p++) {
      const dr = r - palette[p * 3];
      const dg = g - palette[p * 3 + 1];
      const db = b - palette[p * 3 + 2];
      const d = dr * dr + dg * dg + db * db;
      if (d < bestDist) {
        bestDist = d;
        bestIdx = p;
        if (d === 0) break;
      }
    }
    lut[key] = bestIdx;
    return bestIdx;
  }

  // Growable byte writer.
  const chunks = [];
  let buf = new Uint8Array(65536);
  let pos = 0;
  function ensure(extra) {
    if (pos + extra > buf.length) {
      chunks.push(buf.subarray(0, pos));
      buf = new Uint8Array(Math.max(65536, extra * 2));
      pos = 0;
    }
  }
  function writeByte(b) {
    ensure(1);
    buf[pos++] = b & 0xff;
  }
  function writeU16(v) {
    ensure(2);
    buf[pos++] = v & 0xff;
    buf[pos++] = (v >> 8) & 0xff;
  }
  function writeBytes(arr) {
    ensure(arr.length);
    buf.set(arr, pos);
    pos += arr.length;
  }

  // Header + Logical Screen Descriptor + Global Color Table + NETSCAPE2.0 loop.
  writeBytes([0x47, 0x49, 0x46, 0x38, 0x39, 0x61]); // GIF89a
  writeU16(width);
  writeU16(height);
  writeByte(0xf7); // GCT flag=1, 8-bit color res, 256-entry GCT
  writeByte(0);
  writeByte(0);
  writeBytes(palette);
  writeBytes([
    0x21, 0xff, 0x0b, 0x4e, 0x45, 0x54, 0x53, 0x43, 0x41, 0x50, 0x45, 0x32, 0x2e, 0x30,
    0x03, 0x01, 0x00, 0x00, 0x00,
  ]);

  // Reusable LZW open-addressed hash table (zero per-frame GC).
  const HT_SIZE = 16384;
  const HT_MASK = HT_SIZE - 1;
  const htKey = new Int32Array(HT_SIZE);
  const htVal = new Int32Array(HT_SIZE);
  const htStamp = new Uint32Array(HT_SIZE);
  let stamp = 1;
  const subBlock = new Uint8Array(255);
  const indexed = new Uint8Array(nPix);

  for (let f = 0; f < frameCount; f++) {
    if (isCancelled && isCancelled()) return null;

    const rgba = getFrameRgba(f);
    for (let i = 0, p = 0; i < nPix; i++, p += 4) {
      const r = rgba[p];
      const g = rgba[p + 1];
      const b = rgba[p + 2];
      const key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
      indexed[i] = nearestColor(key, r, g, b);
    }

    // Graphic Control Extension
    writeBytes([0x21, 0xf9, 0x04, 0x04, delayCs & 0xff, (delayCs >> 8) & 0xff, 0x00, 0x00]);
    // Image Descriptor
    writeByte(0x2c);
    writeU16(0);
    writeU16(0);
    writeU16(width);
    writeU16(height);
    writeByte(0x00);
    // LZW minimum code size = 8
    writeByte(8);

    const clearCode = 256;
    const eoiCode = 257;
    let nextCode = 258;
    let codeSize = 9;
    let bitBuf = 0;
    let bitCount = 0;
    let subLen = 0;

    function flushSub() {
      if (subLen > 0) {
        ensure(subLen + 1);
        buf[pos++] = subLen;
        buf.set(subBlock.subarray(0, subLen), pos);
        pos += subLen;
        subLen = 0;
      }
    }
    function emit(code) {
      bitBuf |= code << bitCount;
      bitCount += codeSize;
      while (bitCount >= 8) {
        subBlock[subLen++] = bitBuf & 0xff;
        if (subLen === 255) flushSub();
        bitBuf >>>= 8;
        bitCount -= 8;
      }
    }

    stamp++;
    if (stamp === 0xffffffff) {
      htStamp.fill(0);
      stamp = 1;
    }
    emit(clearCode);

    let prefix = indexed[0];
    for (let i = 1; i < nPix; i++) {
      const k = indexed[i];
      const combined = (prefix << 8) | k;
      let slot = ((combined * 2654435761) >>> 18) & HT_MASK;
      let found = -1;
      while (htStamp[slot] === stamp) {
        if (htKey[slot] === combined) {
          found = htVal[slot];
          break;
        }
        slot = (slot + 1) & HT_MASK;
      }
      if (found >= 0) {
        prefix = found;
      } else {
        emit(prefix);
        if (nextCode < 4096) {
          htStamp[slot] = stamp;
          htKey[slot] = combined;
          htVal[slot] = nextCode;
          if (nextCode === 1 << codeSize && codeSize < 12) {
            codeSize++;
          }
          nextCode++;
        } else {
          emit(clearCode);
          stamp++;
          if (stamp === 0xffffffff) {
            htStamp.fill(0);
            stamp = 1;
          }
          nextCode = 258;
          codeSize = 9;
        }
        prefix = k;
      }
    }

    emit(prefix);
    if (nextCode === 1 << codeSize && codeSize < 12) {
      codeSize++;
    }
    emit(eoiCode);
    if (bitCount > 0) {
      subBlock[subLen++] = bitBuf & 0xff;
    }
    flushSub();
    writeByte(0x00); // block terminator

    if ((f & 3) === 3 || f === frameCount - 1) {
      if (onProgress) {
        const frac = 0.5 + 0.48 * ((f + 1) / frameCount);
        onProgress(frac, `Encoding depression tree GIF frame ${f + 1}/${frameCount}...`);
      }
      if (yieldTick) {
        await yieldTick();
      }
    }
  }

  writeByte(0x3b); // GIF trailer
  chunks.push(buf.subarray(0, pos));
  const totalLen = chunks.reduce((s, c) => s + c.length, 0);
  const out = new Uint8Array(totalLen);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}
