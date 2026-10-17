#!/usr/bin/env bash
# scripts/spirv_stats.sh <blob.spv>...
#
# Static SPIR-V metrics of compiled shaders (the dust shaders in particular): one line per blob,
# for `definite_particles.md`. Counts, from `spirv-dis`:
#   funcs     functions (OpFunction)                calls    OpFunctionCall (inlining state)
#   fnvars    OpVariable of Function storage: glslang makes one per by-value argument and per
#             local; struct / array / mat ones are the copies this codebase removes by hand
#   structs   of those, struct-typed              arrays   of those, array-typed
#   mats      of those, matrix-typed
#   dynidx    OpAccessChain into a Function-storage variable with a non-constant index: the
#             sites that stay in local memory whatever the optimiser does
#   clocks    OpReadClockKHR (the profiler variant's probes)
#   words     the blob size in 32-bit words
# Needs spirv-dis (VULKAN_SDK/bin or PATH) and python3.
set -euo pipefail

DIS="${SPIRV_DIS:-}"
if [ -z "$DIS" ]; then
    if [ -n "${VULKAN_SDK:-}" ] && [ -x "$VULKAN_SDK/bin/spirv-dis" ]; then
        DIS="$VULKAN_SDK/bin/spirv-dis"
    else
        DIS="$(command -v spirv-dis || true)"
    fi
fi
if [ -z "$DIS" ]; then
    echo "spirv-dis not found (set VULKAN_SDK or SPIRV_DIS)" >&2
    exit 1
fi

[ $# -ge 1 ] || { echo "usage: $0 <blob.spv>..." >&2; exit 1; }

PY=$(cat <<'PYEOF'
import sys
name, words = sys.argv[1], int(sys.argv[2])
funcs = calls = clocks = 0
ptr_fn = {}      # pointer type id -> pointee type id (Function storage only)
kind = {}        # type id -> 'struct' | 'array' | 'mat'
consts = set()
fnvars = {}      # variable id -> pointee type id
chains = []      # (base id, [index ids])
for line in sys.stdin:
    t = line.split()
    if not t:
        continue
    if len(t) >= 3 and t[1] == '=':
        res, op, ops = t[0], t[2], t[3:]
    else:
        res, op, ops = None, t[0], t[1:]
    if op == 'OpFunction':
        funcs += 1
    elif op == 'OpFunctionCall':
        calls += 1
    elif op == 'OpReadClockKHR':
        clocks += 1
    elif op == 'OpTypePointer' and len(ops) >= 2 and ops[0] == 'Function':
        ptr_fn[res] = ops[1]
    elif op == 'OpTypeStruct':
        kind[res] = 'struct'
    elif op == 'OpTypeArray':
        kind[res] = 'array'
    elif op == 'OpTypeMatrix':
        kind[res] = 'mat'
    elif op.startswith('OpConstant') or op.startswith('OpSpecConstant'):
        consts.add(res)
    elif op == 'OpVariable' and len(ops) >= 2 and ops[1] == 'Function':
        fnvars[res] = ptr_fn.get(ops[0])
    elif op in ('OpAccessChain', 'OpInBoundsAccessChain'):
        # %res = OpAccessChain %type %base %idx...
        chains.append((ops[1], ops[2:]))
structs = sum(1 for v in fnvars.values() if kind.get(v) == 'struct')
arrays = sum(1 for v in fnvars.values() if kind.get(v) == 'array')
mats = sum(1 for v in fnvars.values() if kind.get(v) == 'mat')
dynidx = sum(1 for base, idx in chains if base in fnvars and any(i not in consts for i in idx))
print('%-34s %6d %6d %7d %7d %7d %5d %7d %6d %7d' % (name, funcs, calls, len(fnvars), structs, arrays, mats, dynidx, clocks, words))
PYEOF
)

printf '%-34s %6s %6s %7s %7s %7s %5s %7s %6s %7s\n' blob funcs calls fnvars structs arrays mats dynidx clocks words
for spv in "$@"; do
    words=$(( $(stat -c %s "$spv") / 4 ))
    "$DIS" --raw-id "$spv" | python3 -I -c "$PY" "$(basename "$spv")" "$words"
done
