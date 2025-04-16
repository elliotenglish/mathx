#load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")

cc_library(
    name = "mathx_numerics",
    srcs = glob(["src/**/*.cc","src/**/*.h"]),
    hdrs = glob(["src/**/*.h"]),
    deps = ["//mathx_core:mathx_core",
            "@eigen//:eigen"],
    visibility = ["//visibility:public"],
    strip_include_prefix="src"
)

cc_binary(
    name = "test",
    srcs = glob(["tests/*.cc","tests/*.h"]),
    deps = ["mathx_numerics",
            "//mathx_testing:test_main",
            "@com_google_googletest//:gtest"],
)
