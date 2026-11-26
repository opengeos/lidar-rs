export * from "./lidar_js_direct.d.ts";

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;
export type SyncInitInput = BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
}

/** Synchronously initializes the WebAssembly module (browser / bundler target). */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/** Asynchronously loads and initializes the WebAssembly module (browser ES module target). */
export default function init(
  module_or_path?:
    | { module_or_path: InitInput | Promise<InitInput> }
    | InitInput
    | Promise<InitInput>,
): Promise<InitOutput>;
