#load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")

cc_library(
    name = "test_main",
    srcs = glob(["src/**/*.cc"]),
    hdrs = glob(["src/**/*.h"]),
    deps = ["//mathx_core:mathx_core",
            "@com_google_googletest//:gtest"],
    visibility = ["//visibility:public"],
    strip_include_prefix="src",
    linkstatic=True
)
