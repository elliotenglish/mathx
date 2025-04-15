# Bazel hints

## Find all targets and information
```
$ bazel query ...
```

Note that "..." is literal as it is a wildcard matching all targets under the root (i.e. `//<DIR>/...`).

## See target attributes

```
$ bazel query --output=build <TARGET>
```

## Build a target
```
$ bazel build <TARGET>
```

## Output directory
```
bazel-bin/<TARGET>
```

## Build debug (default)
```
$ bazel build --compilation_mode=dbg <TARGET>
$ bazel build -c dbg <TARGET>
```

## Build optimized
```
$ bazel build --compilation_mode=opt
$ bazel build -c opt
```

## See build commands
```
$ bazel build --verbose_failures <TARGET>
$ bazel build -s <TARGET>
```
