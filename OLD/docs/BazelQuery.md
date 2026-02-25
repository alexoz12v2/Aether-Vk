# Bazel Query System

## Some quick commands

List Actions of a target

```sh
bazel aquery //src/launcher/windows:avk_windows_launcher
```

Compile in dbg

```sh
bazel build //src/launcher/windows:avk_windows_launcher --verbose_failures --compilation_mode=dbg
```

Compile in opt

```sh
bazel build //src/launcher/windows:avk_windows_launcher --verbose_failures --compilation_mode=opt
```

Compile in fastbuild (default)

```sh
bazel build //src/launcher/windows:avk_windows_launcher --verbose_failures --compilation_mode=fastbuild
```

Get Execution Root

```sh
bazel info execution_root
```
