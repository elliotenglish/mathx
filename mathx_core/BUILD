#load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")

cc_library(
    name = "mathx_core",
    srcs = glob(["src/**/*.cc","src/**/*.h"]),
    hdrs = glob(["src/**/*.h"]),
    deps = ["@eigen//:eigen",
            # "//PicoJSON:PicoJSON",
            # "//argh:argh",
            # "@sdl2//:sdl2",
            "@opencv//:opencv",
            "@nlohmann-json//:nlohmann-json"],
    strip_include_prefix = "src",
    visibility = ["//visibility:public"]
)
