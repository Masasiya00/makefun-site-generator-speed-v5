import { setFlagsFromString } from 'node:v8';

// V8's incremental marking off, set before solve.js is imported.
setFlagsFromString('--no-incremental-marking');

const { runAll } = (await import(new URL('./solve.js', import.meta.url).href)) as {
  runAll: (inputDir: string, outputDir: string) => void;
};

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

runAll(INPUT_DIR, OUTPUT_DIR);
