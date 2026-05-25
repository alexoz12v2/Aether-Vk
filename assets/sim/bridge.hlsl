#define ParticleData vk::BufferPointer<MegaParticleData>
#define RigidBodyArray vk::BufferPointer<RigidBody>
#define SparseCollisions vk::BufferPointer<SparseCollisionData>
#define CrossSparseCollisions vk::BufferPointer<CrossCollisionData>
#define PairBuffer vk::BufferPointer<PairBufferType>

#ifndef P_READ
#define P_READ(ptr, offset) vk::RawBufferLoad<float>((uint64_t)(ptr) + (offset)*4)
#define P_WRITE(ptr, offset, val) vk::RawBufferStore<float>((uint64_t)(ptr) + (offset)*4, val)
#endif

#ifndef BDA_LOAD
#define BDA_LOAD(type, addr) vk::RawBufferLoad<type>((uint64_t)(addr))
#define BDA_STORE(type, addr, val) vk::RawBufferStore<type>((uint64_t)(addr), val)
#endif

[[vk::ext_instruction(124)]] vk::BufferPointer<uint> cast_u64_ptr(uint64_t val);

// Valid atomic declarations without Int64Atomics capability mapping
[[vk::ext_instruction(234)]] uint spvAtomicIAdd_ref([[vk::ext_reference]] uint ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(237)]] uint spvAtomicUMin_ref([[vk::ext_reference]] uint ptr, uint scope, uint semantics, uint value);
[[vk::ext_instruction(230)]] uint spvAtomicCompareExchange_ref([[vk::ext_reference]] uint ptr, uint scope, uint semanticsEqual, uint semanticsUnequal, uint value, uint comparator);

// Intercept subagent calls that pass uint64_t directly
#define SPV_SCOPE_DEVICE 1
#define SPV_SEMANTICS_RELAXED 0
#define spvAtomicIAdd(addr, scope, sem, val) spvAtomicIAdd_ref(cast_u64_ptr(addr).Get(), scope, sem, val)
#define spvAtomicUMin(addr, scope, sem, val) spvAtomicUMin_ref(cast_u64_ptr(addr).Get(), scope, sem, val)
#define spvAtomicCompareExchange(addr, scope, semE, semU, val, comp) spvAtomicCompareExchange_ref(cast_u64_ptr(addr).Get(), scope, semE, semU, val, comp)

// GLSL compatibility macros for gjk_cta_utils.glsl
#define vec2 float2
#define vec3 float3
#define vec4 float4
#define mat3 float3x3
#define mat4 float4x4
