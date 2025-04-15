load("@code//config:platform_config.bzl","platform_glob","platform_glob_lib","platform_include_prefix")

cc_library(
    name = "sdl2",
    hdrs = platform_glob(["include/SDL2/SDL.h"]),
    srcs = platform_glob_lib(["SDL2"]),
    strip_include_prefix = platform_include_prefix(),
    visibility = ["//visibility:public"],
)
