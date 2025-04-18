#load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")

cc_library(
    name = "mathx_visualization",
    srcs = glob(["src/**/*.cc","src/**/*.h"]),
    hdrs = glob(["src/**/*.h"]),
    deps = ["@vtk//:vtk",
            "@eigen//:eigen",
            "//mathx_core:mathx_core",
            "//mathx_numerics:mathx_numerics"],
    strip_include_prefix = "src",
    visibility = ["//visibility:public"]
)

cc_binary(
    name = "test",
    srcs = glob(["tests/test.cc","tests/test_utilities.*"]),
    deps = ["@vtk//:vtk","@eigen//:eigen",":mathx_visualization"],
)
