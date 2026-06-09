#[test]
fn print_sizes() {
  use aethervk_core_rlib::gpu::compute_push_constants::*;
  println!(
    "NarrowCcdPushConstants: {}",
    std::mem::size_of::<NarrowCcdPushConstants>()
  );
  println!(
    "NarrowCcdCrossLcaPushConstants: {}",
    std::mem::size_of::<NarrowCcdCrossLcaPushConstants>()
  );
  println!(
    "NarrowCcdParticlesPushConstants: {}",
    std::mem::size_of::<NarrowCcdParticlesPushConstants>()
  );
  println!(
    "P12PushConstants: {}",
    std::mem::size_of::<P12PushConstants>()
  );
  println!(
    "LbvhPushConstants: {}",
    std::mem::size_of::<LbvhPushConstants>()
  );
  println!(
    "CcdPushConstants: {}",
    std::mem::size_of::<CcdPushConstants>()
  );
  println!(
    "StreamCompactPushConstants: {}",
    std::mem::size_of::<StreamCompactPushConstants>()
  );
  println!(
    "ReduceToiPushConstants: {}",
    std::mem::size_of::<ReduceToiPushConstants>()
  );
  println!(
    "LcpPushConstants: {}",
    std::mem::size_of::<LcpPushConstants>()
  );
  println!(
    "BarnesHutPushConstants: {}",
    std::mem::size_of::<BarnesHutPushConstants>()
  );
  println!(
    "P5PushConstants: {}",
    std::mem::size_of::<P5PushConstants>()
  );
  println!(
    "P34PushConstants: {}",
    std::mem::size_of::<P34PushConstants>()
  );
  println!(
    "ImexParticlesP12PushConstants: {}",
    std::mem::size_of::<ImexParticlesP12PushConstants>()
  );
  println!(
    "ImexBodiesP3PushConstants: {}",
    std::mem::size_of::<ImexBodiesP3PushConstants>()
  );
  println!(
    "ImexParticlesP45PushConstants: {}",
    std::mem::size_of::<ImexParticlesP45PushConstants>()
  );
  println!(
    "RbForceAssignPushConstants: {}",
    std::mem::size_of::<RbForceAssignPushConstants>()
  );
  println!(
    "BpClearPushConstants: {}",
    std::mem::size_of::<BpClearPushConstants>()
  );
  println!(
    "BpBoundsGenPushConstants: {}",
    std::mem::size_of::<BpBoundsGenPushConstants>()
  );
  println!(
    "BpScenePushConstants: {}",
    std::mem::size_of::<BpScenePushConstants>()
  );
  println!(
    "BpClassifyPushConstants: {}",
    std::mem::size_of::<BpClassifyPushConstants>()
  );
  println!(
    "BpCrossLcaPushConstants: {}",
    std::mem::size_of::<BpCrossLcaPushConstants>()
  );
  println!(
    "BpParticleSelfPushConstants: {}",
    std::mem::size_of::<BpParticleSelfPushConstants>()
  );
  println!(
    "ApplyEmittersPushConstants: {}",
    std::mem::size_of::<ApplyEmittersPushConstants>()
  );
  println!(
    "EmitParticlesPushConstants: {}",
    std::mem::size_of::<EmitParticlesPushConstants>()
  );
}