/**
 * RI-Bench workspace guard (pi extension, loaded with -e by ribench_agentic.pi_runner).
 *
 * The senior agent may only read/ls/grep inside its case workspace (the pi cwd). Any path that resolves outside it
 * (../, absolute, ~, symlink) is refused before the tool runs, so nothing outside the workspace -- in particular the
 * episode's context.json one level up, which names the arm and condition -- can ever reach the model. The model gets
 * an error result carrying GUARD_MARK; pi_runner.parse_trace records such refused calls as warnings instead of
 * workspace escapes. Added 2026-09-25 after DeepSeek-V4.1-Flash repeatedly listed/read the parent directory.
 */
import { existsSync, realpathSync } from "node:fs";
import { isAbsolute, resolve, sep } from "node:path";

const GUARD_MARK = "RIBENCH_WORKSPACE_GUARD";

function inside(root: string, target: string): boolean {
	return target === root || target.startsWith(root + sep);
}

export default function (pi: any) {
	pi.on("tool_call", async (event: any, ctx: any) => {
		const root = realpathSync(ctx.cwd);
		const raw = (event.input ?? {}).path;
		const path = raw === undefined || raw === null || raw === "" ? "." : raw;
		let ok = typeof path === "string" && !path.startsWith("~");
		if (ok) {
			let target = isAbsolute(path) ? resolve(path) : resolve(root, path);
			if (existsSync(target)) target = realpathSync(target);
			ok = inside(root, target);
		}
		if (!ok) {
			return { block: true, reason: `${GUARD_MARK}: path "${String(path)}" is outside the case directory; only files under the current directory are available.` };
		}
		return undefined;
	});
}
