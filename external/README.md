# External, Locally downloaded, dependencies

## `vk-mem`

Release 0.5.0 for the `vk-mem` create doens't expose its `bindgen` generated `ffi` module, which is necessary for
some low level api usage, hence, the repository has been forked and slightly modified

Remember to do

```shell
git submodule update --init
```

when entering inside this submodule to download `Vulkan-Headers` and `VulkanMemoryAllocator`
