cc_library(
    name = "ceres-solver",
    hdrs = glob(["include/**/*.hpp"]) + glob(["include/**/*.h"]),
    linkopts = [
        "-L /usr/local/opt/ceres-solver/lib -lceres"
    ],
    deps=[
        "@glog//:glog",
        "@eigen//:eigen"
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"]
)
