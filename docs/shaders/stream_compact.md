# `stream_compact.comp`

## Purpose
Packs a sparse array of collision flags into a dense, contiguous array of active collision pairs.

## Mathematical Foundation
Many physics pipelines execute broadphase collision checks resulting in sparse arrays (where index $i$ corresponds to an object or bounding box check, but most checks do not yield actual overlaps). Sorting or solving LCP directly over this sparse array is inefficient. Stream compaction guarantees that subsequent narrowphase and LCP solver phases operate strictly over dense data.

## Implementation Details
- **Subgroup Parallel Prefix Sum**: Uses `subgroupAdd` and `subgroupExclusiveAdd` (from `GL_KHR_shader_subgroup_arithmetic`) to efficiently determine the local offset of each active element within a subgroup.
- **Workgroup Offsets**: Writes subgroup sums into shared memory (`shared_sums`), performs a second prefix sum over the shared array to find the subgroup base offsets, and then uses a single `atomicAdd` to global memory (`workgroup_offset`) to reserve space for the entire workgroup in the dense output array.
