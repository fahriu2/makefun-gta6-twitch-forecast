// Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun
// gta6-twitch-ccu-forecast-01). Node 22, standard library only, deterministic,
// no network.
//
// Idea
// ----
// Every big game launch follows the same daily rhythm in UTC, because the
// unlock rolls at local midnight: New Zealand first, then Europe, then the
// US East (05:00 UTC) and West (08:00 UTC) coasts. The day starts low
// (hour 1 is 00:00-01:00 UTC, when Europe is asleep and the US cannot play
// yet), dips further in hours 2-4, jumps when the US East coast unlocks,
// lulls around midday, then climbs to a Europe-evening hump and a US-evening
// peak before decaying late at night.
//
// The one number we get to anchor on is hour 1's print. Across past
// launches, how high the day climbs relative to hour 1 depends strongly on
// how big hour 1 itself is: a launch whose hour 1 is already huge (the
// whole world stayed up for it) barely grows afterwards, while a launch
// with a small hour 1 can grow 3-9x by US prime time. So for each hour we
// take the comparables' log ratios ln(viewers[h] / viewers[1]) and combine
// them with a Gaussian kernel over ln(hour-1 viewers): launches whose
// hour 1 was a similar size to this case's count most. The median of that
// weighted sample is the forecast's centre; its weighted spread (MAD)
// is the quantile spread, with a floor so an unprecedented launch never
// gets an over-confident narrow band, and a widening when this case's
// hour-1 count sits outside the range the comparables cover.
//
// Nothing here is tuned to any single launch: the comparables are read
// from the case file at run time, so the final case's extra comparables
// (including the four practice launches) are used automatically.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Kernel bandwidth in ln(viewers) units for weighting comparables by hour-1 size. */
const BANDWIDTH = 0.9;
/** Minimum log-spread of any hour's quantiles. */
const SIGMA_FLOOR = 0.3;
/** Extra log-spread per unit of ln(hour-1) outside the comparables' range. */
const EXTRAPOLATION_WIDENING = 0.35;
/** Fallback spread (house baseline) when no comparable is usable. */
const FALLBACK_SIGMA = 0.15;

interface KnownRow {
  hour: number;
  viewers: number;
}

interface ForecastRow {
  hour: number;
}

interface HourlyPoint {
  from: string;
  viewers: number;
}

interface Comparable {
  name: string;
  day_start: string;
  hourly: HourlyPoint[];
}

interface Case {
  game: { name: string };
  levels: number[];
  known: KnownRow[];
  forecast: ForecastRow[];
  comparables: Comparable[];
}

/** The median of a list of numbers. */
function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

/** The weighted median of [value, weight] pairs. */
function weightedMedian(pairs: Array<[number, number]>): number {
  const s = [...pairs].sort((a, b) => a[0] - b[0]);
  const total = s.reduce((acc, [, w]) => acc + w, 0);
  let acc = 0;
  for (const [v, w] of s) {
    acc += w;
    if (acc >= total / 2) return v;
  }
  return s[s.length - 1][0];
}

/** Median absolute deviation of xs around m, scaled to a normal sigma. */
function madSigma(xs: number[], m: number): number {
  return 1.4826 * median(xs.map((x) => Math.abs(x - m)));
}

/** Weighted MAD of pairs around m, scaled to a normal sigma. */
function weightedMadSigma(pairs: Array<[number, number]>, m: number): number {
  return 1.4826 * weightedMedian(pairs.map(([v, w]) => [Math.abs(v - m), w] as [number, number]));
}

/**
 * One comparable's 24 hourly viewer counts for its release day
 * (day_start + 0h .. +23h), or null when the day is incomplete, has a
 * non-positive count, or is not a launch at all (the target game's own
 * pre-release category history, which shares the game's name).
 */
function releaseDayCurve(c: Comparable, gameName: string): number[] | null {
  if (c.name === gameName) return null;
  const start = Date.parse(c.day_start);
  if (!Number.isFinite(start)) return null;
  const byHour = new Map<number, number>();
  for (const pt of c.hourly) {
    const t = Date.parse(pt.from);
    if (!Number.isFinite(t)) continue;
    const h = Math.floor((t - start) / 3_600_000);
    if (h >= 0 && h < 24) byHour.set(h, pt.viewers);
  }
  if (byHour.size < 24) return null;
  const curve: number[] = [];
  for (let h = 0; h < 24; h++) {
    const v = byHour.get(h);
    if (v === undefined || !(v > 0) || !Number.isFinite(v)) return null;
    curve.push(v);
  }
  return curve;
}

/** x rounded to 2 decimals (half up), clamped to the contest's bounds. */
function round2Clamped(x: number): number {
  const r = Math.round(x * 100) / 100;
  if (!Number.isFinite(r)) return 1;
  return Math.min(1_000_000_000, Math.max(0.01, r));
}

/**
 * Forecast one case: for each hour 2..24, five quantiles of that hour's
 * viewer count, as multiples of hour 1's print.
 */
function forecast(c: Case): Array<{ hour: number; q: number[] }> {
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error('case has no hour-1 print');
  const p1 = Math.max(1, first.viewers);
  const logP1 = Math.log(p1);

  const curves = c.comparables
    .map((comp) => releaseDayCurve(comp, c.game.name))
    .filter((curve): curve is number[] => curve !== null);

  // Fallback: no usable comparable (should not happen) -> persistence.
  if (curves.length === 0) {
    return c.forecast.map(({ hour }) => ({
      hour,
      q: Z.map((z) => round2Clamped(p1 * Math.exp(z * FALLBACK_SIGMA * Math.sqrt(hour - 1)))),
    }));
  }

  const logV1 = curves.map((curve) => Math.log(curve[0]));
  const lo = Math.min(...logV1);
  const hi = Math.max(...logV1);
  const outside = Math.max(0, lo - logP1, logP1 - hi);
  const sigmaExtra = 1 + EXTRAPOLATION_WIDENING * outside;

  return c.forecast.map(({ hour }) => {
    const h = hour - 1; // 0-based index into the release-day curve
    const pairs: Array<[number, number]> = curves.map((curve) => {
      const w = Math.exp(-(((Math.log(curve[0]) - logP1) / BANDWIDTH) ** 2));
      return [Math.log(curve[h] / curve[0]), w] as [number, number];
    });
    const med = weightedMedian(pairs);
    const sigma = Math.max(SIGMA_FLOOR, weightedMadSigma(pairs, med)) * sigmaExtra;
    const q = Z.map((z) => round2Clamped(p1 * Math.exp(med + z * sigma)));
    // Keep the row non-decreasing (rounding can only tie, never invert, but be safe).
    for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
    return { hour, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const hours = forecast(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
