# Assmebly Notes

## x86_64

## x86

## ARMv7A 32 bit (armeabi-v7a)

When in this mode the generated code

- Android: Is by default Thumb-2 mode instructions (`BX` or `BLX` to change modes) (when in thumb mode, the program counter is odd and
  the `T` bit of the `CPSR` is set)

- iOS: the default mode is not specified, you need to specify the `--target` in your toolchain

- Android: uses soft float by default, `-mfloat-abi=soft`, meaning functions pass floating point numbers into integer registers,
  which isn't optimal

## ARMv8A 64 bit (armb64-v8a)

