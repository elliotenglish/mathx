load("@code//config:platform_config.bzl","platform_glob","platform_glob_lib","platform_include_prefix")

cc_library(
    name = "eigen",
    hdrs = platform_glob(["include/eigen3/**"]),
    strip_include_prefix = platform_include_prefix("eigen3"),
    visibility = ["//visibility:public"],
)
