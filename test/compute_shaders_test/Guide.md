# Test Execution

First you need to generate `test_data/` by running, for a given test, its corresponding `gen*.py` test with a python environment whose version is greater than or equal 3.10, with packages
`PIL` and `numpy` installed.

Then you can proceed with the usual `cargo nextest run` or debugging with `rust-lldb`

