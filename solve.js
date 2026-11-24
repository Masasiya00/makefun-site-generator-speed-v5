import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const addon = require('./addon/engine.node');

/** Answer one case, exactly as makefun-site 1.0.0's build does. */
export function run(input) {
  return JSON.parse(addon.runOne(JSON.stringify(input)));
}

/** Fast path used by main.ts: render every case in <inputDir>/cases.json
 *  and write <outputDir>/answers.json, entirely inside the native engine. */
export function runAll(inputDir, outputDir) {
  addon.runAll(inputDir, outputDir);
}
