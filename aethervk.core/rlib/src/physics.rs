//! physics module.

pub mod collision_pipeline;
pub mod cpu;
pub mod cpu_kernels;
pub mod handoff;
pub mod lca;
pub mod lcp_integration;
pub mod motion_bvh;
pub mod particle;
pub mod physics_scene;
pub mod tlas_builder;

#[cfg(test)]
mod integration_tests;

#[cfg(test)]
mod vulkan_integration_tests;

#[cfg(test)]
mod kernels_integration_test;

#[cfg(test)]
mod vulkan_math_tests;
